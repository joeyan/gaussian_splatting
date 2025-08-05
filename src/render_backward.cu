#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "checks.cuh"
#include "spherical_harmonics.cuh"

namespace cg = cooperative_groups;

template <typename T, unsigned int CHUNK_SIZE, unsigned int N_SH>
__global__ void render_tiles_backward_kernel(
    const T* __restrict__ uvs,
    const T* __restrict__ opacity,
    const T* __restrict__ rgb,
    const T* __restrict__ conic,
    const T* __restrict__ view_dir_by_pixel,
    const int* __restrict__ splat_start_end_idx_by_tile_idx,
    const int* __restrict__ gaussian_idx_by_splat_idx,
    const T* __restrict__ background_rgb,
    const int* __restrict__ num_splats_per_pixel,
    const T* __restrict__ final_weight_per_pixel,
    const T* __restrict__ grad_image,
    const int image_width,
    const int image_height,
    const bool use_fast_exp,
    T* __restrict__ grad_rgb,     // N_gaussians x 3
    T* __restrict__ grad_opacity, // N_gaussians x 1
    T* __restrict__ grad_uv,      // N_gaussians x 2
    T* __restrict__ grad_conic    // N_gaussians x 3
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

    bool background_initialized = false;
    int num_splats_this_pixel;
    T weight;
    // make local copy for faster access
    T grad_image_local[3];
    T color_accum[3] = {0.0, 0.0, 0.0};
    T view_dir[3];
    T sh_at_view_dir[N_SH];

    // keep threads around even if pixel is not valid for copying data
    bool valid_pixel = u_splat < image_width && v_splat < image_height;

    if (valid_pixel) {
        num_splats_this_pixel = num_splats_per_pixel[v_splat * image_width + u_splat];

        #pragma unroll
        for (int channel = 0; channel < 3; channel++) {
            grad_image_local[channel] = grad_image[(v_splat * image_width + u_splat) * 3 + channel];
            view_dir[channel] = view_dir_by_pixel[(v_splat * image_width + u_splat) * 3 + channel];
        }
        compute_sh_coeffs_for_view_dir<T, N_SH>(view_dir, sh_at_view_dir);
        weight = final_weight_per_pixel[u_splat + v_splat * image_width];
    }

    // shared memory copies of inputs
    __shared__ int _gaussian_idx_by_splat_idx[CHUNK_SIZE];
    __shared__ T _uvs[CHUNK_SIZE * 2];
    __shared__ T _opacity[CHUNK_SIZE];
    __shared__ T _rgb[CHUNK_SIZE * 3 * N_SH];
    __shared__ T _conic[CHUNK_SIZE * 3];

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
            _gaussian_idx_by_splat_idx[i] = gaussian_idx;
            _uvs[i * 2 + 0] = uvs[gaussian_idx * 2 + 0];
            _uvs[i * 2 + 1] = uvs[gaussian_idx * 2 + 1];
            _opacity[i] = opacity[gaussian_idx];

            #pragma unroll
            for (int channel = 0; channel < 3; channel++) {
                #pragma unroll
                for (int sh_idx = 0; sh_idx < N_SH; sh_idx++) {
                    // rgb dimensions = (splat_idx, channel_idx, sh_coeff_idx)
                    _rgb[(i * 3 * N_SH) + (channel * N_SH) + sh_idx] =
                        rgb[(gaussian_idx * 3 * N_SH) + (channel * N_SH) + sh_idx];
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
            T grad_sh[3 * N_SH] = {0.0};
            T grad_opa = 0;
            T grad_u = 0;
            T grad_v = 0;

            T grad_conic_splat[3] = {0.0, 0.0, 0.0};

            // don't compute grad if pixel is out of bounds or this splat is
            // after saturation during forward pass
            if (valid_pixel && tile_splat_idx < num_splats_this_pixel) {
                const T u_mean = _uvs[i * 2 + 0];
                const T v_mean = _uvs[i * 2 + 1];

                const T u_diff = T(u_splat) - u_mean;
                const T v_diff = T(v_splat) - v_mean;

                // 2d covariance matrix b == c so we don't need to duplicate
                // add 0.25 to a and c to make sure matrix is positive definite
                T a;
                T c;
                const T b = _conic[i * 3 + 1] * 0.5;
                if (use_fast_exp) {
                    a = _conic[i * 3 + 0] + 0.25;
                    c = _conic[i * 3 + 2] + 0.25;
                } else {
                    a = _conic[i * 3 + 0];
                    c = _conic[i * 3 + 2];
                }
                const T det = a * c - b * b;

                T norm_prob = 0.0;
                T reciprocal_det = 1.0 / det;
                // compute mahalanobis distance
                const T mh_sq =
                    (c * u_diff * u_diff - (b + b) * u_diff * v_diff + a * v_diff * v_diff) *
                    reciprocal_det;
                if (mh_sq > 0.0) {
                    if (use_fast_exp) {
                        norm_prob = __expf(-0.5 * mh_sq);
                    } else {
                        norm_prob = exp(-0.5 * mh_sq);
                    }
                }

                // compute alpha, prevent divide by zero
                T alpha = min(0.9999, _opacity[i] * norm_prob);

                if (alpha >= 0.00392156862 || !use_fast_exp) {
                    // if this is the first iteration and background blending was used, update
                    // color_accum with the background contribution
                    if (!background_initialized) {
                        const T background_weight = 1.0 - (alpha * weight + 1.0 - weight);
                        if (background_weight > 0.001) {
                            #pragma unroll
                            for (int channel = 0; channel < 3; channel++) {
                                color_accum[channel] += background_rgb[channel] * background_weight;
                            }
                        }
                        background_initialized = true;
                    }
                    const T reciprocal_one_minus_alpha = 1.0 / (1.0 - alpha);

                    // update weight if this is not the first iteration
                    if (i < num_splats_this_pixel - 1) {
                        weight = weight * reciprocal_one_minus_alpha;
                    }

                    T grad_rgb_local[3];
                    #pragma unroll
                    for (int channel = 0; channel < 3; channel++) {
                        grad_rgb_local[channel] = alpha * weight * grad_image_local[channel];
                    }

                    // compute rgb from sh and clamp to valid rgb range
                    T computed_rgb[3];
                    bool was_clamped[3];
                    sh_to_rgb<T, N_SH>(_rgb + i * 3 * N_SH, sh_at_view_dir, computed_rgb);
                    for (int channel = 0; channel < 3; channel++) {
                        was_clamped[channel] = (computed_rgb[channel] < T(0));
                        computed_rgb[channel] = fmax(computed_rgb[channel], T(0));
                    }

                    // compute grad wrt spherical harmonic coeff
                    compute_sh_grad<T, N_SH>(grad_rgb_local, sh_at_view_dir, grad_sh, was_clamped);

                    T grad_alpha = 0.0;
                    #pragma unroll
                    for (int channel = 0; channel < 3; channel++) {
                        grad_alpha += (computed_rgb[channel] * weight -
                                       color_accum[channel] * reciprocal_one_minus_alpha) *
                                      grad_image_local[channel];
                    }
                    grad_opa = norm_prob * grad_alpha;

                    // compute gradient for probability
                    T grad_prob = _opacity[i] * grad_alpha;
                    T grad_mh_sq = -0.5 * norm_prob * grad_prob;

                    // compute gradient for projected mean
                    grad_u =
                        -(-b * v_diff - b * v_diff + 2 * c * u_diff) * reciprocal_det * grad_mh_sq;
                    grad_v =
                        -(2 * a * v_diff - b * u_diff - b * u_diff) * reciprocal_det * grad_mh_sq;

                    const T common_frac = (a * v_diff * v_diff - b * u_diff * v_diff -
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
            }

            // reduce gradients across warp_cg
            // large speedup here by reducing the number of atomicAdd calls
            warp_cg.sync();
            grad_opa = cg::reduce(warp_cg, grad_opa, cg::plus<T>());
            grad_u = cg::reduce(warp_cg, grad_u, cg::plus<T>());
            grad_v = cg::reduce(warp_cg, grad_v, cg::plus<T>());

            #pragma unroll
            for (int channel = 0; channel < 3; channel++) {
                #pragma unroll
                for (int sh_idx = 0; sh_idx < N_SH; sh_idx++) {
                    grad_sh[(channel * N_SH) + sh_idx] =
                        cg::reduce(warp_cg, grad_sh[(channel * N_SH) + sh_idx], cg::plus<T>());
                }
            }

            #pragma unroll
            for (int j = 0; j < 3; j++) {
                grad_conic_splat[j] = cg::reduce(warp_cg, grad_conic_splat[j], cg::plus<T>());
            }

            // write gradients to global memory
            if (warp_cg.thread_rank() == 0) {
                const int gaussian_idx = _gaussian_idx_by_splat_idx[i];

                #pragma unroll
                for (int channel = 0; channel < 3; channel++) {
                    #pragma unroll
                    for (int sh_idx = 0; sh_idx < N_SH; sh_idx++) {
                        // indexing: (gaussian_idx offset) + (channel offset) +
                        // (sh_offset)
                        atomicAdd(
                            grad_rgb + (gaussian_idx * 3 * N_SH) + (channel * N_SH) + sh_idx,
                            grad_sh[(channel * N_SH) + sh_idx]
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
    torch::Tensor background_rgb,
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
    CHECK_VALID_INPUT(background_rgb);
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
    TORCH_CHECK(background_rgb.dim() == 1, "Background RGB must be 1D");
    TORCH_CHECK(background_rgb.size(0) == 3, "Background RGB must have 3 elements");

    int image_height = num_splats_per_pixel.size(0);
    int image_width = num_splats_per_pixel.size(1);
    int num_sh_coeff;
    if (rgb.dim() == 3) {
        num_sh_coeff = rgb.size(2);
    } else {
        num_sh_coeff = 1;
    }

    if (num_sh_coeff > 1) {
        TORCH_CHECK(
            view_dir_by_pixel.size(0) == image_height,
            "view_dir_by_pixel must have the same size as the image"
        );
        TORCH_CHECK(
            view_dir_by_pixel.size(1) == image_width,
            "view_dir_by_pixel must have the same size as the image"
        );
        TORCH_CHECK(view_dir_by_pixel.size(2) == 3, "view_dir_by_pixel must have 3 channels");
    }

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

    if (uvs.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(opacity);
        CHECK_FLOAT_TENSOR(rgb);
        CHECK_FLOAT_TENSOR(conic);
        CHECK_FLOAT_TENSOR(view_dir_by_pixel);
        CHECK_INT_TENSOR(splat_start_end_idx_by_tile_idx);
        CHECK_INT_TENSOR(gaussian_idx_by_splat_idx);
        CHECK_FLOAT_TENSOR(background_rgb);
        CHECK_INT_TENSOR(num_splats_per_pixel);
        CHECK_FLOAT_TENSOR(final_weight_per_pixel);
        CHECK_FLOAT_TENSOR(grad_image);
        CHECK_FLOAT_TENSOR(grad_rgb);
        CHECK_FLOAT_TENSOR(grad_opacity);
        CHECK_FLOAT_TENSOR(grad_uv);
        CHECK_FLOAT_TENSOR(grad_conic);
        if (num_sh_coeff == 1) {
            render_tiles_backward_kernel<float, 960, 1><<<grid_size, block_size>>>(
                uvs.data_ptr<float>(),
                opacity.data_ptr<float>(),
                rgb.data_ptr<float>(),
                conic.data_ptr<float>(),
                view_dir_by_pixel.data_ptr<float>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                background_rgb.data_ptr<float>(),
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<float>(),
                grad_image.data_ptr<float>(),
                image_width,
                image_height,
                true,
                grad_rgb.data_ptr<float>(),
                grad_opacity.data_ptr<float>(),
                grad_uv.data_ptr<float>(),
                grad_conic.data_ptr<float>()
            );
        } else if (num_sh_coeff == 4) {
            render_tiles_backward_kernel<float, 576, 4><<<grid_size, block_size>>>(
                uvs.data_ptr<float>(),
                opacity.data_ptr<float>(),
                rgb.data_ptr<float>(),
                conic.data_ptr<float>(),
                view_dir_by_pixel.data_ptr<float>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                background_rgb.data_ptr<float>(),
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<float>(),
                grad_image.data_ptr<float>(),
                image_width,
                image_height,
                true,
                grad_rgb.data_ptr<float>(),
                grad_opacity.data_ptr<float>(),
                grad_uv.data_ptr<float>(),
                grad_conic.data_ptr<float>()
            );

        } else if (num_sh_coeff == 9) {
            render_tiles_backward_kernel<float, 320, 9><<<grid_size, block_size>>>(
                uvs.data_ptr<float>(),
                opacity.data_ptr<float>(),
                rgb.data_ptr<float>(),
                conic.data_ptr<float>(),
                view_dir_by_pixel.data_ptr<float>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                background_rgb.data_ptr<float>(),
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<float>(),
                grad_image.data_ptr<float>(),
                image_width,
                image_height,
                true,
                grad_rgb.data_ptr<float>(),
                grad_opacity.data_ptr<float>(),
                grad_uv.data_ptr<float>(),
                grad_conic.data_ptr<float>()
            );
        } else if (num_sh_coeff == 16) {
            render_tiles_backward_kernel<float, 160, 16><<<grid_size, block_size>>>(
                uvs.data_ptr<float>(),
                opacity.data_ptr<float>(),
                rgb.data_ptr<float>(),
                conic.data_ptr<float>(),
                view_dir_by_pixel.data_ptr<float>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                background_rgb.data_ptr<float>(),
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<float>(),
                grad_image.data_ptr<float>(),
                image_width,
                image_height,
                true,
                grad_rgb.data_ptr<float>(),
                grad_opacity.data_ptr<float>(),
                grad_uv.data_ptr<float>(),
                grad_conic.data_ptr<float>()
            );
        } else {
            AT_ERROR("Unsupported number of SH coefficients", num_sh_coeff);
        }
    } else if (uvs.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(opacity);
        CHECK_DOUBLE_TENSOR(rgb);
        CHECK_DOUBLE_TENSOR(conic);
        CHECK_DOUBLE_TENSOR(view_dir_by_pixel);
        CHECK_INT_TENSOR(splat_start_end_idx_by_tile_idx);
        CHECK_INT_TENSOR(gaussian_idx_by_splat_idx);
        CHECK_DOUBLE_TENSOR(background_rgb);
        CHECK_INT_TENSOR(num_splats_per_pixel);
        CHECK_DOUBLE_TENSOR(final_weight_per_pixel);
        CHECK_DOUBLE_TENSOR(grad_image);
        CHECK_DOUBLE_TENSOR(grad_rgb);
        CHECK_DOUBLE_TENSOR(grad_opacity);
        CHECK_DOUBLE_TENSOR(grad_uv);
        CHECK_DOUBLE_TENSOR(grad_conic);
        if (num_sh_coeff == 1) {
            render_tiles_backward_kernel<double, 320, 1><<<grid_size, block_size>>>(
                uvs.data_ptr<double>(),
                opacity.data_ptr<double>(),
                rgb.data_ptr<double>(),
                conic.data_ptr<double>(),
                view_dir_by_pixel.data_ptr<double>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                background_rgb.data_ptr<double>(),
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<double>(),
                grad_image.data_ptr<double>(),
                image_width,
                image_height,
                false,
                grad_rgb.data_ptr<double>(),
                grad_opacity.data_ptr<double>(),
                grad_uv.data_ptr<double>(),
                grad_conic.data_ptr<double>()
            );
        } else if (num_sh_coeff == 4) {
            render_tiles_backward_kernel<double, 160, 4><<<grid_size, block_size>>>(
                uvs.data_ptr<double>(),
                opacity.data_ptr<double>(),
                rgb.data_ptr<double>(),
                conic.data_ptr<double>(),
                view_dir_by_pixel.data_ptr<double>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                background_rgb.data_ptr<double>(),
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<double>(),
                grad_image.data_ptr<double>(),
                image_width,
                image_height,
                false,
                grad_rgb.data_ptr<double>(),
                grad_opacity.data_ptr<double>(),
                grad_uv.data_ptr<double>(),
                grad_conic.data_ptr<double>()
            );
        } else if (num_sh_coeff == 9) {
            render_tiles_backward_kernel<double, 128, 9><<<grid_size, block_size>>>(
                uvs.data_ptr<double>(),
                opacity.data_ptr<double>(),
                rgb.data_ptr<double>(),
                conic.data_ptr<double>(),
                view_dir_by_pixel.data_ptr<double>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                background_rgb.data_ptr<double>(),
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<double>(),
                grad_image.data_ptr<double>(),
                image_width,
                image_height,
                false,
                grad_rgb.data_ptr<double>(),
                grad_opacity.data_ptr<double>(),
                grad_uv.data_ptr<double>(),
                grad_conic.data_ptr<double>()
            );
        } else if (num_sh_coeff == 16) {
            render_tiles_backward_kernel<double, 64, 16><<<grid_size, block_size>>>(
                uvs.data_ptr<double>(),
                opacity.data_ptr<double>(),
                rgb.data_ptr<double>(),
                conic.data_ptr<double>(),
                view_dir_by_pixel.data_ptr<double>(),
                splat_start_end_idx_by_tile_idx.data_ptr<int>(),
                gaussian_idx_by_splat_idx.data_ptr<int>(),
                background_rgb.data_ptr<double>(),
                num_splats_per_pixel.data_ptr<int>(),
                final_weight_per_pixel.data_ptr<double>(),
                grad_image.data_ptr<double>(),
                image_width,
                image_height,
                false,
                grad_rgb.data_ptr<double>(),
                grad_opacity.data_ptr<double>(),
                grad_uv.data_ptr<double>(),
                grad_conic.data_ptr<double>()
            );
        } else {
            AT_ERROR("Unsupported number of SH coefficients", num_sh_coeff);
        }
    } else {
        AT_ERROR("Inputs must be float32 or float64");
    }
    cudaDeviceSynchronize();
}
