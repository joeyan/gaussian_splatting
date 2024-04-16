#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "checks.cuh"

template <unsigned int CHUNK_SIZE>
__global__ void render_depth_kernel(
    const float* __restrict__ xyz_camera_frame,
    const float* __restrict__ uvs,
    const float* __restrict__ opacity,
    const float* __restrict__ conic,
    const int* __restrict__ splat_start_end_idx_by_tile_idx,
    const int* __restrict__ gaussian_idx_by_splat_idx,
    const int image_width,
    const int image_height,
    const float alpha_threshold,
    float* __restrict__ depth_image
) {
    // grid = tiles, blocks = pixels within each tile
    const int u_splat = blockIdx.x * blockDim.x + threadIdx.x;
    const int v_splat = blockIdx.y * blockDim.y + threadIdx.y;
    const int tile_idx = blockIdx.x + blockIdx.y * gridDim.x;

    // keep threads around even if pixel is not valid for copying data
    bool valid_pixel = u_splat < image_width && v_splat < image_height;

    const int splat_idx_start = splat_start_end_idx_by_tile_idx[tile_idx];
    const int splat_idx_end = splat_start_end_idx_by_tile_idx[tile_idx + 1];
    int num_splats_this_tile = splat_idx_end - splat_idx_start;

    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;

    float alpha_accum = 0.0;
    float alpha_weight = 0.0;

    // shared memory copies of inputs
    __shared__ int _gaussian_idx_by_splat_idx[CHUNK_SIZE];
    __shared__ float _uvs[CHUNK_SIZE * 2];
    __shared__ float _opacity[CHUNK_SIZE];
    __shared__ float _conic[CHUNK_SIZE * 3];

    const int num_chunks = (num_splats_this_tile + CHUNK_SIZE - 1) / CHUNK_SIZE;
    bool found_depth = false;
    // copy chunks
    for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
        __syncthreads(); // make sure previous iteration is complete before
                         // modifying inputs
        for (int i = thread_id; i < CHUNK_SIZE; i += block_size) {
            const int tile_splat_idx = chunk_idx * CHUNK_SIZE + i;
            if (tile_splat_idx >= num_splats_this_tile) {
                break;
            }
            const int global_splat_idx = splat_idx_start + tile_splat_idx;

            const int gaussian_idx = gaussian_idx_by_splat_idx[global_splat_idx];
            _gaussian_idx_by_splat_idx[i] = gaussian_idx;
            _uvs[i * 2 + 0] = uvs[gaussian_idx * 2 + 0];
            _uvs[i * 2 + 1] = uvs[gaussian_idx * 2 + 1];
            _opacity[i] = opacity[gaussian_idx];

            #pragma unroll
            for (int j = 0; j < 3; j++) {
                _conic[i * 3 + j] = conic[gaussian_idx * 3 + j];
            }
        }
        __syncthreads(); // wait for copying to complete before attempting to
                         // use data
        if (valid_pixel && !found_depth) {
            int chunk_start = chunk_idx * CHUNK_SIZE;
            int chunk_end = min((chunk_idx + 1) * CHUNK_SIZE, num_splats_this_tile);
            int num_splats_this_chunk = chunk_end - chunk_start;
            for (int i = 0; i < num_splats_this_chunk; i++) {
                const float u_mean = _uvs[i * 2 + 0];
                const float v_mean = _uvs[i * 2 + 1];

                const float u_diff = __int2float_rn(u_splat) - u_mean;
                const float v_diff = __int2float_rn(v_splat) - v_mean;

                // 2d covariance matrix - add 0.25 to diagonal to make it positive definite rather
                // than semi-definite
                const float a = _conic[i * 3 + 0] + 0.25;
                const float b = _conic[i * 3 + 1] * 0.5;
                const float c = _conic[i * 3 + 2] + 0.25;
                const float det = a * c - b * b;

                float alpha = 0.0;
                // compute mahalanobis distance
                const float mh_sq =
                    (c * u_diff * u_diff - (b + b) * u_diff * v_diff + a * v_diff * v_diff) / det;
                if (mh_sq > 0.0) {
                    // probablity at this pixel normalized to have
                    // probability at the center of the gaussian to be 1.0
                    const float norm_prob = __expf(-0.5 * mh_sq);
                    alpha = _opacity[i] * norm_prob;
                }
                alpha_weight = 1.0 - alpha_accum;
                const float weight = alpha * (1.0 - alpha_accum);
                alpha_accum += weight;

                if (alpha_accum > alpha_threshold) {
                    // get depth from this gaussians
                    const int gaussian_idx = _gaussian_idx_by_splat_idx[i];
                    const float x = xyz_camera_frame[gaussian_idx * 3 + 0];
                    const float y = xyz_camera_frame[gaussian_idx * 3 + 1];
                    const float z = xyz_camera_frame[gaussian_idx * 3 + 2];
                    const float depth = sqrt(x * x + y * y + z * z);

                    depth_image[v_splat * image_width + u_splat] = depth;
                    found_depth = true;
                    break;
                }
            } // end splat loop
        }     // valid pixel check
    }         // end chunk loop
}

void render_depth_cuda(
    torch::Tensor xyz_camera_frame,
    torch::Tensor uvs,
    torch::Tensor opacity,
    torch::Tensor conic,
    torch::Tensor splat_start_end_idx_by_tile_idx,
    torch::Tensor gaussian_idx_by_splat_idx,
    const float alpha_threshold,
    torch::Tensor depth_image
) {
    CHECK_VALID_INPUT(xyz_camera_frame);
    CHECK_VALID_INPUT(uvs);
    CHECK_VALID_INPUT(opacity);
    CHECK_VALID_INPUT(conic);
    CHECK_VALID_INPUT(splat_start_end_idx_by_tile_idx);
    CHECK_VALID_INPUT(gaussian_idx_by_splat_idx);
    CHECK_VALID_INPUT(depth_image);

    int N = uvs.size(0);
    TORCH_CHECK(uvs.size(1) == 2, "uvs must be Nx2 (u, v)");
    TORCH_CHECK(
        xyz_camera_frame.size(0) == N,
        "xyz_camera_frame must have the same number of elements as uvs"
    );
    TORCH_CHECK(xyz_camera_frame.size(1) == 3, "xyz_camera_frame must be Nx3");
    TORCH_CHECK(opacity.size(0) == N, "Opacity must have the same number of elements as uvs");
    TORCH_CHECK(opacity.size(1) == 1, "Opacity must be Nx1");
    TORCH_CHECK(conic.size(0) == N, "Conic must have the same number of elements as uvs");
    TORCH_CHECK(conic.size(1) == 3, "Conic must be Nx3");
    int image_height = depth_image.size(0);
    int image_width = depth_image.size(1);
    TORCH_CHECK(depth_image.size(2) == 1, "Depth Image must be HxWx1");

    int num_tiles_x = (image_width + 16 - 1) / 16;
    int num_tiles_y = (image_height + 16 - 1) / 16;

    dim3 block_size(16, 16, 1);
    dim3 grid_size(num_tiles_x, num_tiles_y, 1);

    CHECK_FLOAT_TENSOR(xyz_camera_frame);
    CHECK_FLOAT_TENSOR(uvs);
    CHECK_FLOAT_TENSOR(opacity);
    CHECK_FLOAT_TENSOR(conic);
    CHECK_INT_TENSOR(splat_start_end_idx_by_tile_idx);
    CHECK_INT_TENSOR(gaussian_idx_by_splat_idx);
    CHECK_FLOAT_TENSOR(depth_image);

    render_depth_kernel<960><<<grid_size, block_size>>>(
        xyz_camera_frame.data_ptr<float>(),
        uvs.data_ptr<float>(),
        opacity.data_ptr<float>(),
        conic.data_ptr<float>(),
        splat_start_end_idx_by_tile_idx.data_ptr<int>(),
        gaussian_idx_by_splat_idx.data_ptr<int>(),
        image_width,
        image_height,
        alpha_threshold,
        depth_image.data_ptr<float>()
    );
    cudaDeviceSynchronize();
}
