#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "checks.cuh"

# define CHUNKSIZE 300

template<typename T>
__global__ void render_tiles_backward_kernel(
        const T* __restrict__ uvs,
        const T* __restrict__ opacity,
        const T* __restrict__ rgb,
        const T* __restrict__ sigma_image,
        const int* __restrict__ splat_start_end_idx_by_tile_idx,
        const int* __restrict__ gaussian_idx_by_splat_idx,
        const int* __restrict__ num_splats_per_pixel,
        const T* __restrict__ final_weight_per_pixel,
        const T* __restrict__ grad_image,
        const int image_width,
        const int image_height,
        bool use_fast_exp,
        T* __restrict__ grad_rgb, // N_gaussians x 3
        T* __restrict__ grad_opacity, // N_gaussians x 1
        T* __restrict__ grad_uv, // N_gaussians x 2
        T* __restrict__ grad_sigma_image // N_gaussians x 4
) {
    // grid = tiles, blocks = pixels within each tile
    const int u_splat = blockIdx.x * blockDim.x + threadIdx.x;
    const int v_splat = blockIdx.y * blockDim.y + threadIdx.y;

    // tile and splat info
    const int tile_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const int splat_idx_start = splat_start_end_idx_by_tile_idx[tile_idx];
    const int splat_idx_end = splat_start_end_idx_by_tile_idx[tile_idx + 1];
    int num_splats_tile = splat_idx_end - splat_idx_start;

    int num_splats_px;
    T grad_image_r;
    T grad_image_g;
    T grad_image_b;
    T weight;
    T color_accum[3] = {0.0, 0.0, 0.0};

    // keep threads around even if pixel is not valid for copying data
    bool valid_pixel = u_splat < image_width && v_splat < image_height;

    if (valid_pixel) {
        num_splats_px = num_splats_per_pixel[v_splat * image_width + u_splat];
        grad_image_r = grad_image[(v_splat * image_width + u_splat) * 3 + 0];
        grad_image_g = grad_image[(v_splat * image_width + u_splat) * 3 + 1];
        grad_image_b = grad_image[(v_splat * image_width + u_splat) * 3 + 2];
        weight = final_weight_per_pixel[u_splat + v_splat * image_width];
    }


    // shared memory copies of inputs
    __shared__ T _uvs[CHUNKSIZE * 2];
    __shared__ T _opacity[CHUNKSIZE];
    __shared__ T _rgb[CHUNKSIZE * 3];
    __shared__ T _sigma_image[CHUNKSIZE * 4];

    // shared_memory copies of outputs
    __shared__ T _grad_uv[CHUNKSIZE * 2];
    __shared__ T _grad_opacity[CHUNKSIZE];
    __shared__ T _grad_rgb[CHUNKSIZE * 3];
    __shared__ T _grad_sigma_image[CHUNKSIZE * 4];
    

    const int num_chunks = (num_splats_tile + CHUNKSIZE - 1) / CHUNKSIZE;
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;
    // copy chunks last to first
    for (int chunk_idx = num_chunks - 1; chunk_idx >= 0; chunk_idx--) {
        // copy gaussians in-order
        for (int i = thread_id; i < CHUNKSIZE; i += block_size) {
            const int tile_splat_idx = chunk_idx * CHUNKSIZE + i;
            if (tile_splat_idx >= num_splats_tile) {
                break;
            }
            const int global_splat_idx = splat_idx_start + tile_splat_idx;

            // copy gaussians in the order they are splatted
            const int gaussian_idx = gaussian_idx_by_splat_idx[global_splat_idx];
            _uvs[i * 2 + 0] = uvs[gaussian_idx * 2 + 0];
            _uvs[i * 2 + 1] = uvs[gaussian_idx * 2 + 1];
            _opacity[i] = opacity[gaussian_idx];
            _rgb[i * 3 + 0] = rgb[gaussian_idx * 3 + 0];
            _rgb[i * 3 + 1] = rgb[gaussian_idx * 3 + 1];
            _rgb[i * 3 + 2] = rgb[gaussian_idx * 3 + 2];
            _sigma_image[i * 4 + 0] = sigma_image[gaussian_idx * 4 + 0];
            _sigma_image[i * 4 + 1] = sigma_image[gaussian_idx * 4 + 1];
            _sigma_image[i * 4 + 2] = sigma_image[gaussian_idx * 4 + 2];
            _sigma_image[i * 4 + 3] = sigma_image[gaussian_idx * 4 + 3];

            // zero out shared memory gradients
            _grad_uv[i * 2 + 0] = 0.0;
            _grad_uv[i * 2 + 1] = 0.0;
            _grad_opacity[i] = 0.0;
            _grad_rgb[i * 3 + 0] = 0.0;
            _grad_rgb[i * 3 + 1] = 0.0;
            _grad_rgb[i * 3 + 2] = 0.0;
            _grad_sigma_image[i * 4 + 0] = 0.0;
            _grad_sigma_image[i * 4 + 1] = 0.0;
            _grad_sigma_image[i * 4 + 2] = 0.0;
            _grad_sigma_image[i * 4 + 3] = 0.0;
        }
        __syncthreads(); // wait for copying to complete before attempting to use data

        // compute gradients for this chunk
        if (valid_pixel) {
            int chunk_start = chunk_idx * CHUNKSIZE;
            int chunk_end = min((chunk_idx + 1) * CHUNKSIZE, num_splats_px);
            for (int i = chunk_end - chunk_start - 1; i >= 0; i--) {
                const T u_mean = _uvs[i * 2 + 0];
                const T v_mean = _uvs[i * 2 + 1];

                const T u_diff = T(u_splat) - u_mean;
                const T v_diff = T(v_splat) - v_mean;
                
                // 2d covariance matrix
                const T a = _sigma_image[i * 4 + 0];
                const T b = _sigma_image[i * 4 + 1];
                const T c = _sigma_image[i * 4 + 2];
                const T d = _sigma_image[i * 4 + 3];
                T det = a * d - b * c;

                T norm_prob = 0.0;
                T reciprocal_det = 1.0 / det;
                if (det > 0.0) {
                    if (det < 1e-14) {
                        det += 1e-14;
                        reciprocal_det = 1.0 / det;
                    }
                    // compute mahalanobis distance
                    const T mh_sq = (d * u_diff * u_diff - (b + c) * u_diff * v_diff + a * v_diff * v_diff) * reciprocal_det;
                    if (mh_sq > 0.0) {
                        if (use_fast_exp) {
                            norm_prob = __expf(-0.5 * mh_sq);
                        } else {
                            norm_prob = exp(-0.5 * mh_sq);
                        }
                    }
                }
                
                T alpha = _opacity[i] * norm_prob;
                if (abs(alpha - 1.0) < 1e-14) {
                    alpha -= 1e-14;
                }
                const T reciprocal_one_minus_alpha = 1.0 / (1.0 - alpha);
        
                // update weight
                if (i < num_splats_px - 1) {
                    weight = weight * reciprocal_one_minus_alpha;
                }
            
                atomicAdd(_grad_rgb + i * 3 + 0, alpha * weight * grad_image_r);
                atomicAdd(_grad_rgb + i * 3 + 1, alpha * weight * grad_image_g);
                atomicAdd(_grad_rgb + i * 3 + 2, alpha * weight * grad_image_b);
        
                T grad_alpha_r = (_rgb[i * 3 + 0] * weight - color_accum[0] * reciprocal_one_minus_alpha) * grad_image_r;
                T grad_alpha_g = (_rgb[i * 3 + 1] * weight - color_accum[1] * reciprocal_one_minus_alpha) * grad_image_g;
                T grad_alpha_b = (_rgb[i * 3 + 2] * weight - color_accum[2] * reciprocal_one_minus_alpha) * grad_image_b;
                T grad_alpha = grad_alpha_r + grad_alpha_g + grad_alpha_b;
        
                T grad_opa = norm_prob * grad_alpha;

                // update opacity gradient
                atomicAdd(_grad_opacity + i, grad_opa);
        
                // compute gradient for probability
                T grad_prob = _opacity[i] * grad_alpha;
                T grad_mh_sq = -0.5 * norm_prob * grad_prob;
        
                // compute gradient for projected mean
                T grad_u = -(-b * v_diff - c * v_diff + 2 * d * u_diff) * reciprocal_det * grad_mh_sq;
                T grad_v = -(2 * a * v_diff - b * u_diff - c * u_diff) * reciprocal_det * grad_mh_sq;
                atomicAdd(_grad_uv + i * 2 + 0, grad_u);
                atomicAdd(_grad_uv + i * 2 + 1, grad_v);
        
                const T common_frac = (a * v_diff * v_diff - b * u_diff * v_diff - c * u_diff * v_diff + d * u_diff * u_diff) * reciprocal_det * reciprocal_det;
                const T grad_a = (-d * common_frac + v_diff * v_diff * reciprocal_det) * grad_mh_sq;
                const T grad_b = (c * common_frac - u_diff * v_diff * reciprocal_det) * grad_mh_sq;
                const T grad_c = (b * common_frac - u_diff * v_diff * reciprocal_det) * grad_mh_sq;
                const T grad_d = (-a * common_frac + u_diff * u_diff * reciprocal_det) * grad_mh_sq;
                atomicAdd(_grad_sigma_image + i * 4 + 0, grad_a);
                atomicAdd(_grad_sigma_image + i * 4 + 1, grad_b);
                atomicAdd(_grad_sigma_image + i * 4 + 2, grad_c);
                atomicAdd(_grad_sigma_image + i * 4 + 3, grad_d);
        
                // update color_accum for next splat
                color_accum[0] += _rgb[i * 3 + 0] * alpha * weight;
                color_accum[1] += _rgb[i * 3 + 1] * alpha * weight;
                color_accum[2] += _rgb[i * 3 + 2] * alpha * weight;
            } // compute chunk grad
        } // valid pixel check
        __syncthreads(); // wait for all grad computations to complete 

        // copy gradients back to global memory
        for (int i = thread_id; i < CHUNKSIZE; i += block_size) {
            const int tile_splat_idx = chunk_idx * CHUNKSIZE + i;
            if (tile_splat_idx >= num_splats_tile) {
                break;
            }
            const int global_splat_idx = splat_idx_start + tile_splat_idx;
            if (isnan(_grad_opacity[i])) {
                printf("grad_opacity is nan in copy: block_x %d, block_y %d, thread: %d, chunk: %d, i: %d, tile_splat_idx: %d, num_splats_in_tile: %d, num_chunks: %d\n", blockIdx.x, blockIdx.y, thread_id, chunk_idx, i, tile_splat_idx, num_splats_tile, num_chunks);
            } 

            const int gaussian_idx = gaussian_idx_by_splat_idx[global_splat_idx];
            atomicAdd(grad_rgb + gaussian_idx * 3 + 0, _grad_rgb[i * 3 + 0]);
            atomicAdd(grad_rgb + gaussian_idx * 3 + 1, _grad_rgb[i * 3 + 1]);
            atomicAdd(grad_rgb + gaussian_idx * 3 + 2, _grad_rgb[i * 3 + 2]);
            atomicAdd(grad_opacity + gaussian_idx, _grad_opacity[i]);
            atomicAdd(grad_uv + gaussian_idx * 2 + 0, _grad_uv[i * 2 + 0]);
            atomicAdd(grad_uv + gaussian_idx * 2 + 1, _grad_uv[i * 2 + 1]);
            atomicAdd(grad_sigma_image + gaussian_idx * 4 + 0, _grad_sigma_image[i * 4 + 0]);
            atomicAdd(grad_sigma_image + gaussian_idx * 4 + 1, _grad_sigma_image[i * 4 + 1]);
            atomicAdd(grad_sigma_image + gaussian_idx * 4 + 2, _grad_sigma_image[i * 4 + 2]);
            atomicAdd(grad_sigma_image + gaussian_idx * 4 + 3, _grad_sigma_image[i * 4 + 3]);
        } // copy gradients back to global memory
        __syncthreads(); // make sure copy is complete before zeroing out gradients for next iteration
    } // loop over chunks
}


void render_tiles_backward_cuda(
    torch::Tensor uvs,
    torch::Tensor opacity,
    torch::Tensor rgb,
    torch::Tensor sigma_image,
    torch::Tensor splat_start_end_idx_by_tile_idx,
    torch::Tensor gaussian_idx_by_splat_idx,
    torch::Tensor num_splats_per_pixel,
    torch::Tensor final_weight_per_pixel,
    torch::Tensor grad_image,
    torch::Tensor grad_rgb,
    torch::Tensor grad_opacity,
    torch::Tensor grad_uv,
    torch::Tensor grad_sigma_image
) {
    CHECK_VALID_INPUT(uvs);
    CHECK_VALID_INPUT(opacity);
    CHECK_VALID_INPUT(rgb);
    CHECK_VALID_INPUT(sigma_image);
    CHECK_VALID_INPUT(splat_start_end_idx_by_tile_idx);
    CHECK_VALID_INPUT(gaussian_idx_by_splat_idx);
    CHECK_VALID_INPUT(num_splats_per_pixel);
    CHECK_VALID_INPUT(final_weight_per_pixel);
    CHECK_VALID_INPUT(grad_image);
    CHECK_VALID_INPUT(grad_rgb);
    CHECK_VALID_INPUT(grad_opacity);
    CHECK_VALID_INPUT(grad_uv);
    CHECK_VALID_INPUT(grad_sigma_image);

    int N = uvs.size(0);
    TORCH_CHECK(uvs.size(1) == 2, "uvs must have 2 channels");
    TORCH_CHECK(opacity.size(0) == N, "opacity must have the same size as uvs");
    TORCH_CHECK(rgb.size(0) == N, "rgb must have the same size as uvs");
    TORCH_CHECK(rgb.size(1) == 3, "rgb must have 3 channels");
    TORCH_CHECK(sigma_image.size(0) == N, "sigma_image must have the same size as uvs");
    TORCH_CHECK(sigma_image.size(1) == 2, "sigma_image must have 2x2 channels");
    TORCH_CHECK(sigma_image.size(2) == 2, "sigma_image must have 2x2 channels");
    
    int image_height = num_splats_per_pixel.size(0);
    int image_width = num_splats_per_pixel.size(1);

    int num_tiles_x = (image_width + 16 - 1) / 16;
    int num_tiles_y = (image_height + 16 - 1) / 16;

    TORCH_CHECK(splat_start_end_idx_by_tile_idx.size(0) == num_tiles_x * num_tiles_y + 1, "splat_start_end_idx_by_tile_idx ");
    TORCH_CHECK(num_splats_per_pixel.size(0) == image_height, "num_splats_per_pixel must have the same size as the image");
    TORCH_CHECK(num_splats_per_pixel.size(1) == image_width, "num_splats_per_pixel must have the same size as the image");
    TORCH_CHECK(final_weight_per_pixel.size(0) == image_height, "final_weight_per_pixel must have the same size as the image");
    TORCH_CHECK(final_weight_per_pixel.size(1) == image_width, "final_weight_per_pixel must have the same size as the image");
    TORCH_CHECK(grad_image.size(0) == image_height, "grad_image must have the same size as the image");
    TORCH_CHECK(grad_image.size(1) == image_width, "grad_image must have the same size as the image");
    TORCH_CHECK(grad_image.size(2) == 3, "grad_image must have 3 channels");

    dim3 block_size(16, 16, 1);
    dim3 grid_size(num_tiles_x, num_tiles_y, 1);

    if (uvs.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(opacity);
        CHECK_FLOAT_TENSOR(rgb);
        CHECK_FLOAT_TENSOR(sigma_image);
        CHECK_INT_TENSOR(splat_start_end_idx_by_tile_idx);
        CHECK_INT_TENSOR(gaussian_idx_by_splat_idx);
        CHECK_INT_TENSOR(num_splats_per_pixel);
        CHECK_FLOAT_TENSOR(final_weight_per_pixel);
        CHECK_FLOAT_TENSOR(grad_image);
        CHECK_FLOAT_TENSOR(grad_rgb);
        CHECK_FLOAT_TENSOR(grad_opacity);
        CHECK_FLOAT_TENSOR(grad_uv);
        CHECK_FLOAT_TENSOR(grad_sigma_image);
        render_tiles_backward_kernel<float><<<grid_size, block_size>>>(
            uvs.data_ptr<float>(),
            opacity.data_ptr<float>(),
            rgb.data_ptr<float>(),
            sigma_image.data_ptr<float>(),
            splat_start_end_idx_by_tile_idx.data_ptr<int>(),
            gaussian_idx_by_splat_idx.data_ptr<int>(),
            num_splats_per_pixel.data_ptr<int>(),
            final_weight_per_pixel.data_ptr<float>(),
            grad_image.data_ptr<float>(),
            image_width,
            image_height,
            true,
            grad_rgb.data_ptr<float>(),
            grad_opacity.data_ptr<float>(),
            grad_uv.data_ptr<float>(),
            grad_sigma_image.data_ptr<float>()
        );
    } else if (uvs.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(opacity);
        CHECK_DOUBLE_TENSOR(rgb);
        CHECK_DOUBLE_TENSOR(sigma_image);
        CHECK_INT_TENSOR(splat_start_end_idx_by_tile_idx);
        CHECK_INT_TENSOR(gaussian_idx_by_splat_idx);
        CHECK_INT_TENSOR(num_splats_per_pixel);
        CHECK_DOUBLE_TENSOR(final_weight_per_pixel);
        CHECK_DOUBLE_TENSOR(grad_image);
        CHECK_DOUBLE_TENSOR(grad_rgb);
        CHECK_DOUBLE_TENSOR(grad_opacity);
        CHECK_DOUBLE_TENSOR(grad_uv);
        CHECK_DOUBLE_TENSOR(grad_sigma_image);
        render_tiles_backward_kernel<double><<<grid_size, block_size>>>(
            uvs.data_ptr<double>(),
            opacity.data_ptr<double>(),
            rgb.data_ptr<double>(),
            sigma_image.data_ptr<double>(),
            splat_start_end_idx_by_tile_idx.data_ptr<int>(),
            gaussian_idx_by_splat_idx.data_ptr<int>(),
            num_splats_per_pixel.data_ptr<int>(),
            final_weight_per_pixel.data_ptr<double>(),
            grad_image.data_ptr<double>(),
            image_width,
            image_height,
            false,
            grad_rgb.data_ptr<double>(),
            grad_opacity.data_ptr<double>(),
            grad_uv.data_ptr<double>(),
            grad_sigma_image.data_ptr<double>()
        );
    } else {
        AT_ERROR("Inputs must be float32 or float64");
    }
    cudaDeviceSynchronize();
}
