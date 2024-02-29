#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


template<typename T>
__device__ T compute_alpha(
    const int gaussian_idx,
    const int u_splat,
    const int v_splat,
    const T* __restrict__ uvs,
    const T* __restrict__ opacity,
    const T* __restrict__ sigma_image
) {
    const T u_mean = uvs[gaussian_idx * 2 + 0];
    const T v_mean = uvs[gaussian_idx * 2 + 1];

    const T u_diff = T(u_splat) - u_mean;
    const T v_diff = T(v_splat) - v_mean;

    // 2d covariance matrix
    const T a = sigma_image[gaussian_idx * 4 + 0];
    const T b = sigma_image[gaussian_idx * 4 + 1];
    const T c = sigma_image[gaussian_idx * 4 + 2];
    const T d = sigma_image[gaussian_idx * 4 + 3];
    T det = a * d - b * c;
    
    // compute mahalanobis distance
    const T mh_sq = (d * u_diff * u_diff - (b + c) * u_diff * v_diff + a * v_diff * v_diff) / det;

    // probablity at this pixel normalized to have probability at the center of the gaussian to be 1.0
    const T norm_prob = exp(-0.5 * mh_sq);

    // unlikely to produce a visible result with 8 bit images
    T alpha = opacity[gaussian_idx] * norm_prob;
    return alpha;
}


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
        T* __restrict__ grad_rgb, // N_gaussians x 3
        T* __restrict__ grad_opacity, // N_gaussians x 1
        T* __restrict__ grad_uv, // N_gaussians x 2
        T* __restrict__ grad_sigma_image // N_gaussians x 4
) {
    // grid = tiles, blocks = pixels within each tile
    const int u_splat = blockIdx.x * blockDim.x + threadIdx.x;
    const int v_splat = blockIdx.y * blockDim.y + threadIdx.y;
    if (u_splat >= image_width || v_splat >= image_height) {
        return;
    }

    const int tile_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const int splat_idx_start = splat_start_end_idx_by_tile_idx[tile_idx];
    int num_splats = num_splats_per_pixel[u_splat + v_splat * image_width];
    if (num_splats == 0) {
        return;
    }

    T grad_image_r = grad_image[(u_splat + v_splat * image_width) * 3 + 0];
    T grad_image_g = grad_image[(u_splat + v_splat * image_width) * 3 + 1];
    T grad_image_b = grad_image[(u_splat + v_splat * image_width) * 3 + 2];

    T color_accum[3] = {0.0, 0.0, 0.0};
    T weight = final_weight_per_pixel[u_splat + v_splat * image_width];
    for (int i = num_splats - 1; i >= 0; i--) {
        const int splat_idx = splat_idx_start + i;
        const int gaussian_idx = gaussian_idx_by_splat_idx[splat_idx];

        // compute alpha
        const T u_mean = uvs[gaussian_idx * 2 + 0];
        const T v_mean = uvs[gaussian_idx * 2 + 1];

        const T u_diff = T(u_splat) - u_mean;
        const T v_diff = T(v_splat) - v_mean;

        // 2d covariance matrix
        const T a = sigma_image[gaussian_idx * 4 + 0];
        const T b = sigma_image[gaussian_idx * 4 + 1];
        const T c = sigma_image[gaussian_idx * 4 + 2];
        const T d = sigma_image[gaussian_idx * 4 + 3];
        T det = a * d - b * c;

        // compute mahalanobis distance
        const T mh_sq = (d * u_diff * u_diff - (b + c) * u_diff * v_diff + a * v_diff * v_diff) / det;

        // probablity at this pixel normalized to have probability at the center of the gaussian to be 1.0
        const T norm_prob = exp(-0.5 * mh_sq);

        T alpha = opacity[gaussian_idx] * norm_prob;
        
        // update rgb gradient. Since each gaussian is splat to multiple pixels, we need to use atomicAdd
        atomicAdd(grad_rgb + gaussian_idx * 3 + 0, alpha * weight * grad_image_r);
        atomicAdd(grad_rgb + gaussian_idx * 3 + 1, alpha * weight * grad_image_g);
        atomicAdd(grad_rgb + gaussian_idx * 3 + 2, alpha * weight * grad_image_b);
        
        
        // prevent divide by zero
        if (alpha == 1.0) {
            alpha -= 1e-14;
        }
        T grad_alpha_r = (rgb[gaussian_idx * 3 + 0] * weight - color_accum[0]/(1.0 - alpha)) * grad_image_r;
        T grad_alpha_g = (rgb[gaussian_idx * 3 + 1] * weight - color_accum[1]/(1.0 - alpha)) * grad_image_g;
        T grad_alpha_b = (rgb[gaussian_idx * 3 + 2] * weight - color_accum[2]/(1.0 - alpha)) * grad_image_b;
        T grad_alpha = grad_alpha_r + grad_alpha_g + grad_alpha_b;

        // update opacity gradient
        atomicAdd(grad_opacity + gaussian_idx, norm_prob * grad_alpha);


        // compute gradient for probability
        T grad_prob = opacity[gaussian_idx] * grad_alpha;
        T grad_mh_sq = -0.5 * norm_prob * grad_prob;

        // compute gradient for projected mean
        T grad_u = -(-b * v_diff - c * v_diff + 2 * d * u_diff) / det * grad_mh_sq;
        T grad_v = -(2 * a * v_diff - b * u_diff - c * u_diff) / det * grad_mh_sq;
        atomicAdd(grad_uv + gaussian_idx * 2 + 0, grad_u);
        atomicAdd(grad_uv + gaussian_idx * 2 + 1, grad_v);

        T grad_a = (-d * (a * v_diff * v_diff - b * u_diff * v_diff - c * u_diff * v_diff + d * u_diff * u_diff) / (det * det)  + (v_diff * v_diff) / det) * grad_mh_sq;
        T grad_b = (c * (a * v_diff * v_diff - b * u_diff * v_diff - c * u_diff * v_diff + d * u_diff * u_diff) / (det * det)  - (u_diff * v_diff) / det) * grad_mh_sq;
        T grad_c = (b * (a * v_diff * v_diff - b * u_diff * v_diff - c * u_diff * v_diff + d * u_diff * u_diff) / (det * det)  - (u_diff * v_diff) / det) * grad_mh_sq;
        T grad_d = (-a * (a * v_diff * v_diff - b * u_diff * v_diff - c * u_diff * v_diff + d * u_diff * u_diff) / (det * det)  + (u_diff * u_diff) / det) * grad_mh_sq;
        atomicAdd(grad_sigma_image + gaussian_idx * 4 + 0, grad_a);
        atomicAdd(grad_sigma_image + gaussian_idx * 4 + 1, grad_b);
        atomicAdd(grad_sigma_image + gaussian_idx * 4 + 2, grad_c);
        atomicAdd(grad_sigma_image + gaussian_idx * 4 + 3, grad_d);

        // update color_accum for next splat
        color_accum[0] += rgb[gaussian_idx * 3 + 0] * alpha * weight;
        color_accum[1] += rgb[gaussian_idx * 3 + 1] * alpha * weight;
        color_accum[2] += rgb[gaussian_idx * 3 + 2] * alpha * weight;

        // update weight for next splat
        if (i > 0) {
            const int splat_idx_next = splat_idx_start + i - 1;
            const T alpha_next = compute_alpha(
                gaussian_idx_by_splat_idx[splat_idx_next],
                u_splat,
                v_splat,
                uvs,
                opacity,
                sigma_image
            );
            weight = weight / (1.0 - alpha_next);
        }
    }
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
    const int image_width,
    const int image_height,
    torch::Tensor grad_rgb,
    torch::Tensor grad_opacity,
    torch::Tensor grad_uv,
    torch::Tensor grad_sigma_image
) {
    int num_tiles_x = (image_width + 16 - 1) / 16;
    int num_tiles_y = (image_height + 16 - 1) / 16;
    
    dim3 block_size(16, 16, 1);
    dim3 grid_size(num_tiles_x, num_tiles_y, 1);

    if (uvs.dtype() == torch::kFloat32) {
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
            grad_rgb.data_ptr<float>(),
            grad_opacity.data_ptr<float>(),
            grad_uv.data_ptr<float>(),
            grad_sigma_image.data_ptr<float>()
        );
    } else if (uvs.dtype() == torch::kFloat64) {
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
            grad_rgb.data_ptr<double>(),
            grad_opacity.data_ptr<double>(),
            grad_uv.data_ptr<double>(),
            grad_sigma_image.data_ptr<double>()
        );
    } else {
        AT_ERROR("Unsupported dtype");
    }
    cudaDeviceSynchronize();
}