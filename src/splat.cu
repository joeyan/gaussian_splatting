#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cuda_bf16.h>

#include "bfloat162_helpers.cuh"
#include "checks.cuh"
#include "spherical_harmonics.cuh"

template <
    typename T,
    bool use_fast_exp,
    unsigned int CHUNK_SIZE,
    unsigned int N_SH,
    unsigned int N_SH_PAIRS>
__global__ void render_tiles_kernel(
    const T* __restrict__ uvs,
    const T* __restrict__ opacity,
    const __nv_bfloat162* __restrict__ rgb,
    const T* __restrict__ conic,
    const T* __restrict__ view_dir_by_pixel,
    const int* __restrict__ splat_start_end_idx_by_tile_idx,
    const int* __restrict__ gaussian_idx_by_splat_idx,
    const int image_width,
    const int image_height,
    int* num_splats_per_pixel,
    T* __restrict__ final_weight_per_pixel,
    T* __restrict__ image
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

    T alpha_accum = 0.0;
    T alpha_weight = 0.0;
    int num_splats = 0;

    __nv_bfloat16 view_dir[3];
    __nv_bfloat162 sh_at_view_dir[N_SH_PAIRS];

    if (valid_pixel) {
        #pragma unroll
        for (int axis = 0; axis < 3; axis++) {
            view_dir[axis] =
                __float2bfloat16(view_dir_by_pixel[(v_splat * image_width + u_splat) * 3 + axis]);
        }

        // compute sh as bfloat16
        __nv_bfloat16 tmp_sh[N_SH];
        compute_sh_coeffs_for_view_dir<__nv_bfloat16, N_SH>(view_dir, tmp_sh);

        // convert sh to bfloat162
        #pragma unroll
        for (int sh_pair_idx = 0; sh_pair_idx < N_SH_PAIRS; sh_pair_idx++) {
            int sh_idx = sh_pair_idx * 2;
            // first value is valid in loop
            sh_at_view_dir[sh_pair_idx].x = tmp_sh[sh_idx];
            if (sh_idx + 1 < N_SH) {
                sh_at_view_dir[sh_pair_idx].y = tmp_sh[sh_idx + 1];
            } else {
                sh_at_view_dir[sh_pair_idx].y = __float2bfloat16(0.0f);
            }
        }
    }

    // shared memory copies of inputs
    __shared__ T _uvs[CHUNK_SIZE * 2];
    __shared__ T _opacity[CHUNK_SIZE];
    __shared__ __nv_bfloat162 _rgb[CHUNK_SIZE * 3 * N_SH_PAIRS];
    __shared__ T _conic[CHUNK_SIZE * 3];

    const int shared_image_size = 16 * 16 * 3;
    __shared__ T _image[shared_image_size];

    #pragma unroll
    for (int i = thread_id; i < shared_image_size; i += block_size) {
        _image[i] = 0.0;
    }

    const int num_chunks = (num_splats_this_tile + CHUNK_SIZE - 1) / CHUNK_SIZE;
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
            _uvs[i * 2 + 0] = uvs[gaussian_idx * 2 + 0];
            _uvs[i * 2 + 1] = uvs[gaussian_idx * 2 + 1];
            _opacity[i] = opacity[gaussian_idx];

            #pragma unroll
            for (int sh_pair_idx = 0; sh_pair_idx < N_SH_PAIRS; sh_pair_idx++) {
                #pragma unroll
                for (int channel = 0; channel < 3; channel++) {
                    // rgb dimensions = (splat_idx, channel_idx, sh_coeff_idx)
                    _rgb[(i * 3 + channel) * N_SH_PAIRS + sh_pair_idx] =
                        rgb[(gaussian_idx * 3 + channel) * N_SH_PAIRS + sh_pair_idx];
                }
            }

            #pragma unroll
            for (int j = 0; j < 3; j++) {
                _conic[i * 3 + j] = conic[gaussian_idx * 3 + j];
            }
        }
        __syncthreads(); // wait for copying to complete before attempting to
                         // use data
        if (valid_pixel) {
            int chunk_start = chunk_idx * CHUNK_SIZE;
            int chunk_end = min((chunk_idx + 1) * CHUNK_SIZE, num_splats_this_tile);
            int num_splats_this_chunk = chunk_end - chunk_start;
            for (int i = 0; i < num_splats_this_chunk; i++) {
                if (alpha_accum > 0.999) {
                    break;
                }
                const T u_mean = _uvs[i * 2 + 0];
                const T v_mean = _uvs[i * 2 + 1];

                const T u_diff = __int2float_rn(u_splat) - u_mean;
                const T v_diff = __int2float_rn(v_splat) - v_mean;

                // 2d covariance matrix
                const T a = _conic[i * 3 + 0];
                const T b = _conic[i * 3 + 1] / 2.0;
                const T c = _conic[i * 3 + 2];
                T det = a * c - b * b;

                T alpha = 0.0;
                // skip any covariance matrices that are not positive definite
                if (det > 0.0) {
                    if (det < 1e-14) {
                        det += 1e-14;
                    }
                    // compute mahalanobis distance
                    const T mh_sq =
                        (c * u_diff * u_diff - (b + b) * u_diff * v_diff + a * v_diff * v_diff) /
                        det;
                    if (mh_sq > 0.0) {
                        // probablity at this pixel normalized to have
                        // probability at the center of the gaussian to be 1.0
                        T norm_prob = 0.0;
                        if (use_fast_exp) {
                            norm_prob = __expf(-0.5 * mh_sq);
                        } else {
                            norm_prob = exp(-0.5 * mh_sq);
                        }
                        alpha = _opacity[i] * norm_prob;
                    }
                }
                alpha_weight = 1.0 - alpha_accum;
                const T weight = alpha * (1.0 - alpha_accum);

                // compute rgb
                __nv_bfloat162 tmp_rgb[3];

                // equal to pair index zero
                #pragma unroll
                for (int channel = 0; channel < 3; channel++) {
                    tmp_rgb[channel] =
                        __hmul2(_rgb[(i * 3 + channel) * N_SH_PAIRS], sh_at_view_dir[0]);
                }

                // add additional sh pairs
                #pragma unroll
                for (int channel = 0; channel < 3; channel++) {
                    #pragma unroll
                    for (int sh_pair_idx = 1; sh_pair_idx < N_SH_PAIRS; sh_pair_idx++) {
                        // hfma2(a, b, c) = a * b + c
                        tmp_rgb[channel] = __hfma2(
                            _rgb[(i * 3 + channel) * N_SH_PAIRS + sh_pair_idx],
                            sh_at_view_dir[sh_pair_idx],
                            tmp_rgb[channel]
                        );
                    }
                }

                // convert rgb to float32
                float computed_rgb[3];
                #pragma unroll
                for (int channel = 0; channel < 3; channel++) {
                    // add the two parts of the accumulated bfloat162 together
                    computed_rgb[channel] =
                        __bfloat162float(__hadd(tmp_rgb[channel].x, tmp_rgb[channel].y));
                }

                #pragma unroll
                for (int channel = 0; channel < 3; channel++) {
                    _image[(threadIdx.y * 16 + threadIdx.x) * 3 + channel] +=
                        computed_rgb[channel] * weight;
                }

                alpha_accum += weight;
                num_splats++;
            } // end splat loop
        }     // valid pixel check
    }         // end chunk loop

    // copy back to global memory
    __syncthreads(); // wait for splatting to complete
    if (valid_pixel) {
        num_splats_per_pixel[v_splat * image_width + u_splat] = num_splats;
        final_weight_per_pixel[v_splat * image_width + u_splat] = alpha_weight;

        #pragma unroll
        for (int channel = 0; channel < 3; channel++) {
            image[(v_splat * image_width + u_splat) * 3 + channel] =
                _image[(threadIdx.y * 16 + threadIdx.x) * 3 + channel];
        }
    }
}

void render_tiles_cuda(
    torch::Tensor uvs,
    torch::Tensor opacity,
    torch::Tensor rgb,
    torch::Tensor conic,
    torch::Tensor view_dir_by_pixel,
    torch::Tensor splat_start_end_idx_by_tile_idx,
    torch::Tensor gaussian_idx_by_splat_idx,
    torch::Tensor num_splats_per_pixel,
    torch::Tensor final_weight_per_pixel,
    torch::Tensor rendered_image
) {
    CHECK_VALID_INPUT(uvs);
    CHECK_VALID_INPUT(opacity);
    CHECK_VALID_INPUT(rgb);
    CHECK_VALID_INPUT(conic);
    CHECK_VALID_INPUT(view_dir_by_pixel);
    CHECK_VALID_INPUT(splat_start_end_idx_by_tile_idx);
    CHECK_VALID_INPUT(gaussian_idx_by_splat_idx);
    CHECK_VALID_INPUT(num_splats_per_pixel);
    CHECK_VALID_INPUT(final_weight_per_pixel);
    CHECK_VALID_INPUT(rendered_image);

    int N = uvs.size(0);
    TORCH_CHECK(uvs.size(1) == 2, "uvs must be Nx2 (u, v)");
    TORCH_CHECK(opacity.size(0) == N, "Opacity must have the same number of elements as uvs");
    TORCH_CHECK(opacity.size(1) == 1, "Opacity must be Nx1");
    TORCH_CHECK(rgb.size(0) == N, "RGB must have the same number of elements as uvs");
    TORCH_CHECK(rgb.size(1) == 3, "RGB must be Nx3");
    TORCH_CHECK(conic.size(0) == N, "Conic must have the same number of elements as uvs");
    TORCH_CHECK(conic.size(1) == 3, "Conic must be Nx3");
    TORCH_CHECK(rendered_image.size(2) == 3, "Image must be HxWx3");

    int image_height = rendered_image.size(0);
    int image_width = rendered_image.size(1);
    TORCH_CHECK(
        view_dir_by_pixel.size(0) == image_height,
        "view_dir_by_pixel must have the same height as the image"
    );
    TORCH_CHECK(
        view_dir_by_pixel.size(1) == image_width,
        "view_dir_by_pixel must have the same width as the image"
    );
    TORCH_CHECK(view_dir_by_pixel.size(2) == 3, "view_dir_by_pixel must have 3 channels");

    int num_tiles_x = (image_width + 16 - 1) / 16;
    int num_tiles_y = (image_height + 16 - 1) / 16;

    dim3 block_size(16, 16, 1);
    dim3 grid_size(num_tiles_x, num_tiles_y, 1);

    int num_sh_coeff;
    if (rgb.dim() == 3) {
        num_sh_coeff = rgb.size(2);
    } else {
        num_sh_coeff = 1;
    }

    at::BFloat16* rgb_at = rgb.data_ptr<at::BFloat16>();
    __nv_bfloat16* rgb_bf16 = reinterpret_cast<__nv_bfloat16*>(rgb_at);

    // create bfloat162 output to store gradient
    const int num_sh_pairs = (num_sh_coeff + 1) / 2;
    const int num_b162 = N * 3 * num_sh_pairs;

    // allocate bfloat162 rgb
    __nv_bfloat162* rgb_bf162;
    cudaMalloc(&rgb_bf162, num_b162 * sizeof(__nv_bfloat162));

    if (uvs.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(opacity);
        CHECK_BFLOAT16_TENSOR(rgb);
        CHECK_FLOAT_TENSOR(conic);
        CHECK_FLOAT_TENSOR(view_dir_by_pixel);
        CHECK_INT_TENSOR(splat_start_end_idx_by_tile_idx);
        CHECK_INT_TENSOR(gaussian_idx_by_splat_idx);
        CHECK_INT_TENSOR(num_splats_per_pixel);
        CHECK_FLOAT_TENSOR(final_weight_per_pixel);
        CHECK_FLOAT_TENSOR(rendered_image);

        if (num_sh_coeff == 1) {
            // convert rgb
            const int max_threads_per_block = 1024 / 3;
            const int num_blocks = (num_b162 + max_threads_per_block - 1) / max_threads_per_block;
            dim3 convert_gridsize(num_blocks, 1, 1);
            dim3 convert_blocksize(max_threads_per_block, 3, 1);

            convert_rgb_bfloat162<<<convert_gridsize, convert_blocksize>>>(
                rgb_bf16, N, num_sh_coeff, num_sh_pairs, rgb_bf162
            );
            cudaDeviceSynchronize();

            render_tiles_kernel<float, true, 960, 1, 1><<<grid_size, block_size>>>(
                uvs.data_ptr<float>(),
                opacity.data_ptr<float>(),
                rgb_bf162,
                conic.data_ptr<float>(),
                view_dir_by_pixel.data_ptr<float>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                image_width,
                image_height,
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<float>(),
                rendered_image.data_ptr<float>()
            );
        } else if (num_sh_coeff == 4) {
            const int max_threads_per_block = 1024 / 12;
            const int num_blocks = (num_b162 + max_threads_per_block - 1) / max_threads_per_block;
            dim3 convert_gridsize(num_blocks, 1, 1);
            dim3 convert_blocksize(max_threads_per_block, 3, 4);

            convert_rgb_bfloat162<<<convert_gridsize, convert_blocksize>>>(
                rgb_bf16, N, num_sh_coeff, num_sh_pairs, rgb_bf162
            );
            cudaDeviceSynchronize();

            render_tiles_kernel<float, true, 960, 4, 2><<<grid_size, block_size>>>(
                uvs.data_ptr<float>(),
                opacity.data_ptr<float>(),
                rgb_bf162,
                conic.data_ptr<float>(),
                view_dir_by_pixel.data_ptr<float>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                image_width,
                image_height,
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<float>(),
                rendered_image.data_ptr<float>()
            );
        } else if (num_sh_coeff == 9) {
            const int max_threads_per_block = 1024 / 27;
            const int num_blocks = (num_b162 + max_threads_per_block - 1) / max_threads_per_block;
            dim3 convert_gridsize(num_blocks, 1, 1);
            dim3 convert_blocksize(max_threads_per_block, 3, 9);

            convert_rgb_bfloat162<<<convert_gridsize, convert_blocksize>>>(
                rgb_bf16, N, num_sh_coeff, num_sh_pairs, rgb_bf162
            );
            cudaDeviceSynchronize();

            render_tiles_kernel<float, true, 512, 9, 5><<<grid_size, block_size>>>(
                uvs.data_ptr<float>(),
                opacity.data_ptr<float>(),
                rgb_bf162,
                conic.data_ptr<float>(),
                view_dir_by_pixel.data_ptr<float>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                image_width,
                image_height,
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<float>(),
                rendered_image.data_ptr<float>()
            );
        } else if (num_sh_coeff == 16) {
            const int max_threads_per_block = 1024 / 48;
            const int num_blocks = (num_b162 + max_threads_per_block - 1) / max_threads_per_block;
            dim3 convert_gridsize(num_blocks, 1, 1);
            dim3 convert_blocksize(max_threads_per_block, 3, 16);

            convert_rgb_bfloat162<<<convert_gridsize, convert_blocksize>>>(
                rgb_bf16, N, num_sh_coeff, num_sh_pairs, rgb_bf162
            );
            cudaDeviceSynchronize();

            render_tiles_kernel<float, true, 384, 16, 8><<<grid_size, block_size>>>(
                uvs.data_ptr<float>(),
                opacity.data_ptr<float>(),
                rgb_bf162,
                conic.data_ptr<float>(),
                view_dir_by_pixel.data_ptr<float>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                image_width,
                image_height,
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<float>(),
                rendered_image.data_ptr<float>()
            );
        } else {
            AT_ERROR("Unsupported number of SH coefficients: ", num_sh_coeff);
        }
    } else {
        AT_ERROR("Inputs must be float32 or float64");
    }
    cudaDeviceSynchronize();
    cudaFree(rgb_bf162);
}
