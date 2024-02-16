#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void render_tiles_kernel(
        const float* uvs,
        const float* opacity,
        const float* rgb,
        const float* sigma_image,
        const int* gaussian_start_end_indices,
        const int* gaussian_indices_by_tile,
        const int image_width,
        const int image_height,
        float* image
) {
    // grid = tiles, blocks = pixels within each tile
    const int u_splat = blockIdx.x * blockDim.x + threadIdx.x;
    const int v_splat = blockIdx.y * blockDim.y + threadIdx.y;

    if (u_splat >= image_width || v_splat >= image_height) {
        return;
    }

    const int tile_idx = blockIdx.x + blockIdx.y * gridDim.x;

    const int start = gaussian_start_end_indices[tile_idx];
    const int end = gaussian_start_end_indices[tile_idx + 1];

    float alpha_accum = 0.0f;
    for (int i = start; i < end; i++) {
        if (alpha_accum > 0.999f) {
            break;
        }
        const int gaussian_idx = gaussian_indices_by_tile[i];

        const float u_mean = uvs[gaussian_idx * 2 + 0];
        const float v_mean = uvs[gaussian_idx * 2 + 1];

        const float u_diff = __int2float_rn(u_splat) - u_mean;
        const float v_diff = __int2float_rn(v_splat) - v_mean;

        // 2d covariance matrix
        const float a = sigma_image[gaussian_idx * 4 + 0];
        const float b = sigma_image[gaussian_idx * 4 + 1];
        const float c = sigma_image[gaussian_idx * 4 + 2];
        const float d = sigma_image[gaussian_idx * 4 + 3];

        // compute mahalanobis distance
        const float mh = (d * u_diff * u_diff - (b + c) * u_diff * v_diff + a * v_diff * v_diff) / (a * d - b * c);
        
        // probablity at this pixel normalized to have probability at the center of the gaussian to be 1.0
        const float norm_prob = __expf(-0.5f * mh);

        // unlikely to produce a visible result with 8 bit images
        if (norm_prob < 1e-14f) {
            continue;
        }

        // alpha blending
        const float alpha = opacity[gaussian_idx] * norm_prob;
        const float weight = alpha * (1.0f - alpha_accum);

        image[(v_splat * image_width + u_splat) * 3 + 0] += rgb[gaussian_idx * 3 + 0] * weight;
        image[(v_splat * image_width + u_splat) * 3 + 1] += rgb[gaussian_idx * 3 + 1] * weight;
        image[(v_splat * image_width + u_splat) * 3 + 2] += rgb[gaussian_idx * 3 + 2] * weight;

        alpha_accum += weight;
    }
}

void render_tiles_cuda(
        torch::Tensor uvs,
        torch::Tensor opacity,
        torch::Tensor rgb,
        torch::Tensor sigma_image,
        torch::Tensor gaussian_start_end_indices,
        torch::Tensor gaussian_indices_by_tile,
        torch::Tensor rendered_image) {
    TORCH_CHECK(uvs.is_cuda(), "uvs must be CUDA tensor");
    TORCH_CHECK(opacity.is_cuda(), "opacity must be CUDA tensor");
    TORCH_CHECK(rgb.is_cuda(), "rgb must be CUDA tensor");
    TORCH_CHECK(sigma_image.is_cuda(), "sigma_image must be CUDA tensor");
    TORCH_CHECK(gaussian_start_end_indices.is_cuda(), "gaussian_start_end_indices must be CUDA tensor");
    TORCH_CHECK(gaussian_indices_by_tile.is_cuda(), "gaussian_indices_by_tile must be CUDA tensor");
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

    render_tiles_kernel<<<grid_size, block_size>>>(
        uvs.data_ptr<float>(),
        opacity.data_ptr<float>(),
        rgb.data_ptr<float>(),
        sigma_image.data_ptr<float>(),
        gaussian_start_end_indices.data_ptr<int>(),
        gaussian_indices_by_tile.data_ptr<int>(),
        image_width,
        image_height,
        rendered_image.data_ptr<float>()
    );
    cudaDeviceSynchronize();

}
