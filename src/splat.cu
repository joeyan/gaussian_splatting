#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


template<typename T>
__global__ void render_tiles_kernel(
        const T* __restrict__ uvs,
        const T* __restrict__ opacity,
        const T* __restrict__ rgb,
        const T* __restrict__ sigma_image,
        const int* __restrict__ splat_start_end_idx_by_tile_idx,
        const int* __restrict__ gaussian_idx_by_splat_idx,
        const int image_width,
        const int image_height,
        int* num_splats_per_pixel,
        T* final_weight_per_pixel,
        T* image
) {
    // grid = tiles, blocks = pixels within each tile
    const int u_splat = blockIdx.x * blockDim.x + threadIdx.x;
    const int v_splat = blockIdx.y * blockDim.y + threadIdx.y;
    if (u_splat >= image_width || v_splat >= image_height) {
        return;
    }

    const int tile_idx = blockIdx.x + blockIdx.y * gridDim.x;

    const int splat_idx_start = splat_start_end_idx_by_tile_idx[tile_idx];
    const int splat_idx_end = splat_start_end_idx_by_tile_idx[tile_idx + 1];

    T alpha_accum = 0.0;
    T alpha_weight = 0.0;

    int num_splats = 0;
    for (int splat_idx = splat_idx_start; splat_idx < splat_idx_end; splat_idx++) {
        if (alpha_accum > 0.999) {
            break;
        }
        const int gaussian_idx = gaussian_idx_by_splat_idx[splat_idx];

        const T u_mean = uvs[gaussian_idx * 2 + 0];
        const T v_mean = uvs[gaussian_idx * 2 + 1];

        const T u_diff = T(u_splat) - u_mean;
        const T v_diff = T(v_splat) - v_mean;

        // 2d covariance matrix
        const T a = sigma_image[gaussian_idx * 4 + 0];
        const T b = sigma_image[gaussian_idx * 4 + 1];
        const T c = sigma_image[gaussian_idx * 4 + 2];
        const T d = sigma_image[gaussian_idx * 4 + 3];

        // compute mahalanobis distance
        const T mh_sq = (d * u_diff * u_diff - (b + c) * u_diff * v_diff + a * v_diff * v_diff) / (a * d - b * c);
        
        // probablity at this pixel normalized to have probability at the center of the gaussian to be 1.0
        const T norm_prob = exp(-0.5 * mh_sq);

        // alpha blending
        const T alpha = opacity[gaussian_idx] * norm_prob;
        alpha_weight = 1.0 - alpha_accum;
        const T weight = alpha * (1.0 - alpha_accum);

        image[(v_splat * image_width + u_splat) * 3 + 0] += rgb[gaussian_idx * 3 + 0] * weight;
        image[(v_splat * image_width + u_splat) * 3 + 1] += rgb[gaussian_idx * 3 + 1] * weight;
        image[(v_splat * image_width + u_splat) * 3 + 2] += rgb[gaussian_idx * 3 + 2] * weight;

        alpha_accum += weight;
        num_splats++;
    }
    num_splats_per_pixel[v_splat * image_width + u_splat] = num_splats;
    final_weight_per_pixel[v_splat * image_width + u_splat] = alpha_weight;
}

void render_tiles_cuda(
        torch::Tensor uvs,
        torch::Tensor opacity,
        torch::Tensor rgb,
        torch::Tensor sigma_image,
        torch::Tensor splat_start_end_idx_by_tile_idx,
        torch::Tensor gaussian_idx_by_splat_idx,
        torch::Tensor num_splats_per_pixel,
        torch::Tensor final_weight_per_pixel,
        torch::Tensor rendered_image) {
    TORCH_CHECK(uvs.is_cuda(), "uvs must be CUDA tensor");
    TORCH_CHECK(opacity.is_cuda(), "opacity must be CUDA tensor");
    TORCH_CHECK(rgb.is_cuda(), "rgb must be CUDA tensor");
    TORCH_CHECK(sigma_image.is_cuda(), "sigma_image must be CUDA tensor");
    TORCH_CHECK(splat_start_end_idx_by_tile_idx.is_cuda(), "splat_start_end_idx_by_tile_idx must be CUDA tensor");
    TORCH_CHECK(gaussian_idx_by_splat_idx.is_cuda(), "gaussian_idx_by_splat_idx must be CUDA tensor");
    TORCH_CHECK(num_splats_per_pixel.is_cuda(), "num_splats_per_pixel must be CUDA tensor");
    TORCH_CHECK(final_weight_per_pixel.is_cuda(), "final_weight_per_pixel must be CUDA tensor");
    TORCH_CHECK(rendered_image.is_cuda(), "rendered_image must be CUDA tensor");

    int N = uvs.size(0);
    TORCH_CHECK(uvs.size(1) == 2, "uvs must be Nx2 (u, v)");
    TORCH_CHECK(opacity.size(0) == N, "Opacity must have the same number of elements as uvs");
    TORCH_CHECK(opacity.size(1) == 1, "Opacity must be Nx1");
    TORCH_CHECK(rgb.size(0) == N, "RGB must have the same number of elements as uvs");
    TORCH_CHECK(rgb.size(1) == 3, "RGB must be Nx3");
    TORCH_CHECK(sigma_image.size(0) == N, "Sigma image must have the same number of elements as uvs");
    TORCH_CHECK(sigma_image.size(1) == 2, "Sigma image must be Nx2x2");
    TORCH_CHECK(sigma_image.size(1) == 2, "Sigma image must be Nx2x2");
    TORCH_CHECK(rendered_image.size(2) == 3, "Image must be HxWx3");

    int image_height = rendered_image.size(0);
    int image_width = rendered_image.size(1);
    int num_tiles_x = (image_width + 16 - 1) / 16;
    int num_tiles_y = (image_height + 16 - 1) / 16;

    dim3 block_size(16, 16, 1);
    dim3 grid_size(num_tiles_x, num_tiles_y, 1);

    if (uvs.dtype() == torch::kFloat32) {
        render_tiles_kernel<float><<<grid_size, block_size>>>(
            uvs.data_ptr<float>(),
            opacity.data_ptr<float>(),
            rgb.data_ptr<float>(),
            sigma_image.data_ptr<float>(),
            splat_start_end_idx_by_tile_idx.data_ptr<int>(),
            gaussian_idx_by_splat_idx.data_ptr<int>(),
            image_width,
            image_height,
            num_splats_per_pixel.data_ptr<int>(),
            final_weight_per_pixel.data_ptr<float>(),
            rendered_image.data_ptr<float>()
        );
    } else if (uvs.dtype() == torch::kFloat64) {
        render_tiles_kernel<double><<<grid_size, block_size>>>(
            uvs.data_ptr<double>(),
            opacity.data_ptr<double>(),
            rgb.data_ptr<double>(),
            sigma_image.data_ptr<double>(),
            splat_start_end_idx_by_tile_idx.data_ptr<int>(),
            gaussian_idx_by_splat_idx.data_ptr<int>(),
            image_width,
            image_height,
            num_splats_per_pixel.data_ptr<int>(),
            final_weight_per_pixel.data_ptr<double>(),
            rendered_image.data_ptr<double>()
        );
    } else {
        TORCH_CHECK(false, "Unsupported dtype");
    }
    cudaDeviceSynchronize();
}
