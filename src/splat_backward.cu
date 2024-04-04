#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "bfloat162_helpers.cuh"
#include "checks.cuh"
#include "spherical_harmonics.cuh"

namespace cg = cooperative_groups;

template <unsigned int CHUNK_SIZE, unsigned int N_SH, unsigned int N_SH_PAIRS>
__global__ void render_tiles_backward_kernel(
    const float* __restrict__ uvs,
    const float* __restrict__ opacity,
    const __nv_bfloat162* __restrict__ rgb,
    const float* __restrict__ conic,
    const float* __restrict__ view_dir_by_pixel,
    const int* __restrict__ splat_start_end_idx_by_tile_idx,
    const int* __restrict__ gaussian_idx_by_splat_idx,
    const int* __restrict__ num_splats_per_pixel,
    const float* __restrict__ final_weight_per_pixel,
    const float* __restrict__ grad_image,
    const int image_width,
    const int image_height,
    bool use_fast_exp,
    __nv_bfloat162* __restrict__ grad_rgb, // N_gaussians x 3 x (N_SH + 1)/2
    float* __restrict__ grad_opacity,      // N_gaussians x 1
    float* __restrict__ grad_uv,           // N_gaussians x 2
    float* __restrict__ grad_conic         // N_gaussians x 3
) {
    auto block = cg::this_thread_block();
    cg::thread_block_tile<32> warp_cg = cg::tiled_partition<32>(block);

    // grid = tiles, blocks = pixels within each tile
    const int u_splat = blockIdx.x * blockDim.x + threadIdx.x;
    const int v_splat = blockIdx.y * blockDim.y + threadIdx.y;

    // tile and splat info
    const int tile_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const int splat_idx_start = splat_start_end_idx_by_tile_idx[tile_idx];
    const int splat_idx_end = splat_start_end_idx_by_tile_idx[tile_idx + 1];
    int num_splats_this_tile = splat_idx_end - splat_idx_start;

    int num_splats_this_pixel;
    float weight;
    // make local copy for faster access
    float grad_image_local[3];
    float color_accum[3] = {0.0, 0.0, 0.0};

    __nv_bfloat16 view_dir[3];
    __nv_bfloat162 sh_at_view_dir[N_SH_PAIRS];

    // keep threads around even if pixel is not valid for copying data
    bool valid_pixel = u_splat < image_width && v_splat < image_height;

    if (valid_pixel) {
        num_splats_this_pixel = num_splats_per_pixel[v_splat * image_width + u_splat];
        weight = final_weight_per_pixel[u_splat + v_splat * image_width];

        #pragma unroll
        for (int channel = 0; channel < 3; channel++) {
            grad_image_local[channel] = grad_image[(v_splat * image_width + u_splat) * 3 + channel];
            view_dir[channel] =
                __float2bfloat16(view_dir_by_pixel[(v_splat * image_width + u_splat) * 3 + channel]
                );
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
    __shared__ float _uvs[CHUNK_SIZE * 2];
    __shared__ float _opacity[CHUNK_SIZE];
    __shared__ __nv_bfloat162 _rgb[CHUNK_SIZE * 3 * N_SH_PAIRS];
    __shared__ float _conic[CHUNK_SIZE * 3];

    const int num_chunks = (num_splats_this_tile + CHUNK_SIZE - 1) / CHUNK_SIZE;
    const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    const int block_size = blockDim.x * blockDim.y;
    // copy chunks last to first
    for (int chunk_idx = num_chunks - 1; chunk_idx >= 0; chunk_idx--) {
        // copy gaussians in-order
        __syncthreads(); // make sure previous iteration is complete before
                         // modifying inputs
        for (int i = thread_id; i < CHUNK_SIZE; i += block_size) {
            const int tile_splat_idx = chunk_idx * CHUNK_SIZE + i;
            if (tile_splat_idx >= num_splats_this_tile) {
                break;
            }
            const int global_splat_idx = splat_idx_start + tile_splat_idx;

            // copy gaussians in the order they are splatted
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

        // compute gradients for this chunk
        int chunk_start = chunk_idx * CHUNK_SIZE;
        int chunk_end = min((chunk_idx + 1) * CHUNK_SIZE, num_splats_this_tile);
        for (int i = chunk_end - chunk_start - 1; i >= 0; i--) {
            const int tile_splat_idx = chunk_idx * CHUNK_SIZE + i;
            __nv_bfloat162 grad_sh[3 * N_SH_PAIRS] = {__float2bfloat162_rn(0.0f)};

            float grad_opa = 0;
            float grad_u = 0;
            float grad_v = 0;

            float grad_conic_splat[3] = {0.0, 0.0, 0.0};

            // don't compute grad if pixel is out of bounds or this splat is
            // after saturation during forward pass
            if (valid_pixel && tile_splat_idx < num_splats_this_pixel) {
                const float u_mean = _uvs[i * 2 + 0];
                const float v_mean = _uvs[i * 2 + 1];

                const float u_diff = __int2float_rn(u_splat) - u_mean;
                const float v_diff = __int2float_rn(v_splat) - v_mean;

                // 2d covariance matrix b == c so we don't need to duplicate
                const float a = _conic[i * 3 + 0];
                const float b = _conic[i * 3 + 1] / 2.0;
                const float c = _conic[i * 3 + 2];
                float det = a * c - b * b;

                float norm_prob = 0.0;
                float reciprocal_det = 1.0 / det;
                if (det > 0.0) {
                    if (det < 1e-14) {
                        det += 1e-14;
                        reciprocal_det = 1.0 / det;
                    }
                    // compute mahalanobis distance
                    const float mh_sq =
                        (c * u_diff * u_diff - (b + b) * u_diff * v_diff + a * v_diff * v_diff) *
                        reciprocal_det;
                    if (mh_sq > 0.0) {
                        if (use_fast_exp) {
                            norm_prob = __expf(-0.5 * mh_sq);
                        } else {
                            norm_prob = exp(-0.5 * mh_sq);
                        }
                    }
                }

                float alpha = _opacity[i] * norm_prob;
                if (abs(alpha - 1.0) < 1e-14) {
                    alpha -= 1e-14;
                }
                const float reciprocal_one_minus_alpha = 1.0 / (1.0 - alpha);

                // update weight if this is not the first iteration
                if (i < num_splats_this_pixel - 1) {
                    weight = weight * reciprocal_one_minus_alpha;
                }

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

                // compute grad wrt spherical harmonic coeff
                #pragma unroll
                for (int channel = 0; channel < 3; channel++) {
                    __nv_bfloat162 grad_channel =
                        __float2bfloat162_rn(alpha * weight * grad_image_local[channel]);
                    #pragma unroll
                    for (int sh_pair_idx = 0; sh_pair_idx < N_SH_PAIRS; sh_pair_idx++) {
                        grad_sh[channel * N_SH_PAIRS + sh_pair_idx] =
                            __hmul2(sh_at_view_dir[sh_pair_idx], grad_channel);
                    }
                }

                float grad_alpha = 0.0;
                #pragma unroll
                for (int channel = 0; channel < 3; channel++) {
                    grad_alpha += (computed_rgb[channel] * weight -
                                   color_accum[channel] * reciprocal_one_minus_alpha) *
                                  grad_image_local[channel];
                }
                grad_opa = norm_prob * grad_alpha;

                // compute gradient for probability
                float grad_prob = _opacity[i] * grad_alpha;
                float grad_mh_sq = -0.5 * norm_prob * grad_prob;

                // compute gradient for projected mean
                grad_u = -(-b * v_diff - b * v_diff + 2 * c * u_diff) * reciprocal_det * grad_mh_sq;
                grad_v = -(2 * a * v_diff - b * u_diff - b * u_diff) * reciprocal_det * grad_mh_sq;

                const float common_frac = (a * v_diff * v_diff - b * u_diff * v_diff -
                                           b * u_diff * v_diff + c * u_diff * u_diff) *
                                          reciprocal_det * reciprocal_det;
                grad_conic_splat[0] =
                    (-c * common_frac + v_diff * v_diff * reciprocal_det) * grad_mh_sq;
                grad_conic_splat[1] =
                    (b * common_frac - u_diff * v_diff * reciprocal_det) * grad_mh_sq;
                grad_conic_splat[2] =
                    (-a * common_frac + u_diff * u_diff * reciprocal_det) * grad_mh_sq;

                // update color_accum for next splat
                for (int channel = 0; channel < 3; channel++) {
                    color_accum[channel] += computed_rgb[channel] * alpha * weight;
                }
            }

            // reduce gradients across warp_cg
            // large speedup here by reducing the number of atomicAdd calls
            warp_cg.sync();
            grad_opa = cg::reduce(warp_cg, grad_opa, cg::plus<float>());
            grad_u = cg::reduce(warp_cg, grad_u, cg::plus<float>());
            grad_v = cg::reduce(warp_cg, grad_v, cg::plus<float>());

            #pragma unroll
            for (int channel = 0; channel < 3; channel++) {
                #pragma unroll
                for (int sh_pair_idx = 0; sh_pair_idx < N_SH_PAIRS; sh_pair_idx++) {
                    grad_sh[(channel * N_SH_PAIRS) + sh_pair_idx].x = cg::reduce(
                        warp_cg,
                        grad_sh[(channel * N_SH_PAIRS) + sh_pair_idx].x,
                        cg::plus<__nv_bfloat16>()
                    );
                    grad_sh[(channel * N_SH_PAIRS) + sh_pair_idx].y = cg::reduce(
                        warp_cg,
                        grad_sh[(channel * N_SH_PAIRS) + sh_pair_idx].y,
                        cg::plus<__nv_bfloat16>()
                    );
                }
            }

            #pragma unroll
            for (int j = 0; j < 3; j++) {
                grad_conic_splat[j] = cg::reduce(warp_cg, grad_conic_splat[j], cg::plus<float>());
            }

            // write gradients to global memory
            if (warp_cg.thread_rank() == 0) {
                const int global_splat_idx = splat_idx_start + tile_splat_idx;
                const int gaussian_idx = gaussian_idx_by_splat_idx[global_splat_idx];

                #pragma unroll
                for (int channel = 0; channel < 3; channel++) {
                    #pragma unroll
                    for (int sh_pair_idx = 0; sh_pair_idx < N_SH_PAIRS; sh_pair_idx++) {
                        // indexing: (gaussian_idx offset) + (channel offset) + (sh_offset)
                        atomicAdd(
                            grad_rgb + (gaussian_idx * 3 * N_SH_PAIRS) + (channel * N_SH_PAIRS) +
                                sh_pair_idx,
                            grad_sh[(channel * N_SH_PAIRS) + sh_pair_idx]
                        );
                    }
                }
                atomicAdd(grad_opacity + gaussian_idx, grad_opa);
                atomicAdd(grad_uv + gaussian_idx * 2 + 0, grad_u);
                atomicAdd(grad_uv + gaussian_idx * 2 + 1, grad_v);

                for (int j = 0; j < 3; j++) {
                    atomicAdd(grad_conic + gaussian_idx * 3 + j, grad_conic_splat[j]);
                }
            }
        } // compute chunk grad
    }     // loop over chunks
}

void render_tiles_backward_cuda(
    torch::Tensor uvs,
    torch::Tensor opacity,
    torch::Tensor rgb,
    torch::Tensor conic,
    torch::Tensor view_dir_by_pixel,
    torch::Tensor splat_start_end_idx_by_tile_idx,
    torch::Tensor gaussian_idx_by_splat_idx,
    torch::Tensor num_splats_per_pixel,
    torch::Tensor final_weight_per_pixel,
    torch::Tensor grad_image,
    torch::Tensor grad_rgb,
    torch::Tensor grad_opacity,
    torch::Tensor grad_uv,
    torch::Tensor grad_conic
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
    CHECK_VALID_INPUT(grad_image);
    CHECK_VALID_INPUT(grad_rgb);
    CHECK_VALID_INPUT(grad_opacity);
    CHECK_VALID_INPUT(grad_uv);
    CHECK_VALID_INPUT(grad_conic);

    int N = uvs.size(0);
    TORCH_CHECK(uvs.size(1) == 2, "uvs must have 2 channels");
    TORCH_CHECK(opacity.size(0) == N, "opacity must have the same size as uvs");
    TORCH_CHECK(rgb.size(0) == N, "rgb must have the same size as uvs");
    TORCH_CHECK(rgb.size(1) == 3, "rgb must have 3 channels");
    TORCH_CHECK(conic.size(0) == N, "conic must have the same size as uvs");
    TORCH_CHECK(conic.size(1) == 3, "conic must have 3 channels");

    int image_height = num_splats_per_pixel.size(0);
    int image_width = num_splats_per_pixel.size(1);
    TORCH_CHECK(
        view_dir_by_pixel.size(0) == image_height,
        "view_dir_by_pixel must have the same size as the image"
    );
    TORCH_CHECK(
        view_dir_by_pixel.size(1) == image_width,
        "view_dir_by_pixel must have the same size as the image"
    );
    TORCH_CHECK(view_dir_by_pixel.size(2) == 3, "view_dir_by_pixel must have 3 channels");

    int num_tiles_x = (image_width + 16 - 1) / 16;
    int num_tiles_y = (image_height + 16 - 1) / 16;

    TORCH_CHECK(
        splat_start_end_idx_by_tile_idx.size(0) == num_tiles_x * num_tiles_y + 1,
        "splat_start_end_idx_by_tile_idx "
    );
    TORCH_CHECK(
        num_splats_per_pixel.size(0) == image_height,
        "num_splats_per_pixel must have the same size as the image"
    );
    TORCH_CHECK(
        num_splats_per_pixel.size(1) == image_width,
        "num_splats_per_pixel must have the same size as the image"
    );
    TORCH_CHECK(
        final_weight_per_pixel.size(0) == image_height,
        "final_weight_per_pixel must have the same size as the image"
    );
    TORCH_CHECK(
        final_weight_per_pixel.size(1) == image_width,
        "final_weight_per_pixel must have the same size as the image"
    );

    TORCH_CHECK(
        grad_image.size(0) == image_height, "grad_image must have the same size as the image"
    );
    TORCH_CHECK(
        grad_image.size(1) == image_width, "grad_image must have the same size as the image"
    );
    TORCH_CHECK(grad_image.size(2) == 3, "grad_image must have 3 channels");

    dim3 block_size(16, 16, 1);
    dim3 grid_size(num_tiles_x, num_tiles_y, 1);

    int num_sh_coeff;
    if (rgb.dim() == 3) {
        num_sh_coeff = rgb.size(2);
    } else {
        num_sh_coeff = 1;
    }

    // cast rgb to bfloat16 type from at::BFloat16
    at::BFloat16* rgb_at = rgb.data_ptr<at::BFloat16>();
    __nv_bfloat16* rgb_bf16 = reinterpret_cast<__nv_bfloat16*>(rgb_at);

    const int num_sh_pairs = (num_sh_coeff + 1) / 2;
    const int num_b162 = N * 3 * num_sh_pairs;

    // allocate bfloat162 rgb
    __nv_bfloat162* rgb_bf162;
    cudaMalloc(&rgb_bf162, num_b162 * sizeof(__nv_bfloat162));

    // create bfloat162 output to store gradient
    __nv_bfloat162* grad_rgb_bf162;
    cudaMalloc(&grad_rgb_bf162, num_b162 * sizeof(__nv_bfloat162));
    cudaMemset(grad_rgb_bf162, 0, num_b162 * sizeof(__nv_bfloat162));

    if (uvs.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(opacity);
        CHECK_BFLOAT16_TENSOR(rgb);
        CHECK_FLOAT_TENSOR(conic);
        CHECK_FLOAT_TENSOR(view_dir_by_pixel);
        CHECK_INT_TENSOR(splat_start_end_idx_by_tile_idx);
        CHECK_INT_TENSOR(gaussian_idx_by_splat_idx);
        CHECK_INT_TENSOR(num_splats_per_pixel);
        CHECK_FLOAT_TENSOR(final_weight_per_pixel);
        CHECK_FLOAT_TENSOR(grad_image);
        CHECK_BFLOAT16_TENSOR(grad_rgb);
        CHECK_FLOAT_TENSOR(grad_opacity);
        CHECK_FLOAT_TENSOR(grad_uv);
        CHECK_FLOAT_TENSOR(grad_conic);

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

            render_tiles_backward_kernel<960, 1, 1><<<grid_size, block_size>>>(
                uvs.data_ptr<float>(),
                opacity.data_ptr<float>(),
                rgb_bf162,
                conic.data_ptr<float>(),
                view_dir_by_pixel.data_ptr<float>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<float>(),
                grad_image.data_ptr<float>(),
                image_width,
                image_height,
                true,
                grad_rgb_bf162,
                grad_opacity.data_ptr<float>(),
                grad_uv.data_ptr<float>(),
                grad_conic.data_ptr<float>()
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

            render_tiles_backward_kernel<960, 4, 2><<<grid_size, block_size>>>(
                uvs.data_ptr<float>(),
                opacity.data_ptr<float>(),
                rgb_bf162,
                conic.data_ptr<float>(),
                view_dir_by_pixel.data_ptr<float>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<float>(),
                grad_image.data_ptr<float>(),
                image_width,
                image_height,
                true,
                grad_rgb_bf162,
                grad_opacity.data_ptr<float>(),
                grad_uv.data_ptr<float>(),
                grad_conic.data_ptr<float>()
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

            render_tiles_backward_kernel<512, 9, 5><<<grid_size, block_size>>>(
                uvs.data_ptr<float>(),
                opacity.data_ptr<float>(),
                rgb_bf162,
                conic.data_ptr<float>(),
                view_dir_by_pixel.data_ptr<float>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<float>(),
                grad_image.data_ptr<float>(),
                image_width,
                image_height,
                true,
                grad_rgb_bf162,
                grad_opacity.data_ptr<float>(),
                grad_uv.data_ptr<float>(),
                grad_conic.data_ptr<float>()
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

            render_tiles_backward_kernel<384, 16, 8><<<grid_size, block_size>>>(
                uvs.data_ptr<float>(),
                opacity.data_ptr<float>(),
                rgb_bf162,
                conic.data_ptr<float>(),
                view_dir_by_pixel.data_ptr<float>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<float>(),
                grad_image.data_ptr<float>(),
                image_width,
                image_height,
                true,
                grad_rgb_bf162,
                grad_opacity.data_ptr<float>(),
                grad_uv.data_ptr<float>(),
                grad_conic.data_ptr<float>()
            );
        } else {
            AT_ERROR("Unsupported number of SH coefficients", num_sh_coeff);
        }
    } else {
        AT_ERROR("Inputs must be float32 or float64");
    }
    cudaDeviceSynchronize();
    cudaFree(rgb_bf162);

    // convert grad_rgb to at::BFloat16
    at::BFloat16* grad_rgb_ptr = grad_rgb.data_ptr<at::BFloat16>();
    __nv_bfloat16* grad_rgb_ptr_bf16 = reinterpret_cast<__nv_bfloat16*>(grad_rgb_ptr);

    if (num_sh_coeff == 1) {
        const int max_threads_per_block = 1024 / 3;
        const int num_blocks = (num_b162 + max_threads_per_block - 1) / max_threads_per_block;
        dim3 convert_gridsize(num_blocks, 1, 1);
        dim3 convert_blocksize(max_threads_per_block, 3, 1);

        convert_rgb_grad_to_bfloat16<<<convert_gridsize, convert_blocksize>>>(
            grad_rgb_bf162, N, num_sh_coeff, num_sh_pairs, grad_rgb_ptr_bf16
        );

    } else if (num_sh_coeff == 4) {
        const int max_threads_per_block = 1024 / 12;
        const int num_blocks = (num_b162 + max_threads_per_block - 1) / max_threads_per_block;
        dim3 convert_gridsize(num_blocks, 1, 1);
        dim3 convert_blocksize(max_threads_per_block, 3, 4);

        convert_rgb_grad_to_bfloat16<<<convert_gridsize, convert_blocksize>>>(
            grad_rgb_bf162, N, num_sh_coeff, num_sh_pairs, grad_rgb_ptr_bf16
        );

    } else if (num_sh_coeff == 9) {
        const int max_threads_per_block = 1024 / 27;
        const int num_blocks = (num_b162 + max_threads_per_block - 1) / max_threads_per_block;
        dim3 convert_gridsize(num_blocks, 1, 1);
        dim3 convert_blocksize(max_threads_per_block, 3, 9);

        convert_rgb_grad_to_bfloat16<<<convert_gridsize, convert_blocksize>>>(
            grad_rgb_bf162, N, num_sh_coeff, num_sh_pairs, grad_rgb_ptr_bf16
        );

    } else if (num_sh_coeff == 16) {
        const int max_threads_per_block = 1024 / 48;
        const int num_blocks = (num_b162 + max_threads_per_block - 1) / max_threads_per_block;
        dim3 convert_gridsize(num_blocks, 1, 1);
        dim3 convert_blocksize(max_threads_per_block, 3, 16);

        convert_rgb_grad_to_bfloat16<<<convert_gridsize, convert_blocksize>>>(
            grad_rgb_bf162, N, num_sh_coeff, num_sh_pairs, grad_rgb_ptr_bf16
        );
    } else {
        AT_ERROR("Unsupported number of SH coefficients", num_sh_coeff);
    }
    cudaDeviceSynchronize();
    cudaFree(grad_rgb_bf162);
}
