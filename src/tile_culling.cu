#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void compute_tiles_kernel (
    float* uvs,
    float* sigma_image,
    int n_tiles_x,
    int n_tiles_y,
    float mh_dist,
    int N,
    int max_gaussians_per_tile,
    int* gaussian_indices_per_tile,
    int* num_gaussians_per_tile
) {
    int gaussian_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gaussian_idx >= N) {
        return;
    }

    float a = sigma_image[gaussian_idx * 4];
    float b = sigma_image[gaussian_idx * 4 + 1];
    float c = sigma_image[gaussian_idx * 4 + 2];
    float d = sigma_image[gaussian_idx * 4 + 3];

    // compute major axis radius of ellipse
    float left = (a + d) / 2;
    float right = sqrtf((a - d) * (a - d) / 4 + b * c);
    float lambda1 = left + right;
    float lambda2 = left - right;

    float r_major = mh_dist * sqrtf(lambda1);
    float r_minor = mh_dist * sqrtf(lambda2);

    float box_left = uvs[gaussian_idx * 2] - r_major;
    float box_right = uvs[gaussian_idx * 2] + r_major;
    float box_top = uvs[gaussian_idx * 2 + 1] - r_major;
    float box_bottom = uvs[gaussian_idx * 2 + 1] + r_major;

    // iterate through tiles
    for (int tile_x = 0; tile_x < n_tiles_x; tile_x++) {
        for (int tile_y = 0; tile_y < n_tiles_y; tile_y++) {
            int tile_idx = tile_y * n_tiles_x + tile_x;
            // if the tile is already full, skip
            if (num_gaussians_per_tile[tile_idx] >= max_gaussians_per_tile) {
                continue;
            }

            float tile_left = (float)tile_x * 16.0f;
            float tile_right = (float)(tile_x + 1)  * 16.0f;
            float tile_top = (float)tile_y  * 16.0f;
            float tile_bottom = (float)(tile_y + 1)  * 16.0f;

            float min_right = fminf(tile_right, box_right);
            float max_left = fmaxf(tile_left, box_left);

            float min_bottom = fminf(tile_bottom, box_bottom);
            float max_top = fmaxf(tile_top, box_top);

            // from split axis theorem, need overlap on all axes
            if (min_right > max_left && min_bottom > max_top) {
                // get the next available index in gaussian_indices_per_tile[tile_idx]
                int index = atomicAdd(num_gaussians_per_tile + tile_idx, 1);
                if (index < max_gaussians_per_tile) {
                    // write gaussian index
                    gaussian_indices_per_tile[tile_idx * max_gaussians_per_tile + index] = gaussian_idx;
                }
            }
        }
    }
}


void compute_tiles_cuda (
    torch::Tensor uvs,
    torch::Tensor sigma_image,
    int n_tiles_x,
    int n_tiles_y,
    float mh_dist,
    torch::Tensor gaussian_indices_per_tile,
    torch::Tensor num_gaussians_per_tile
) {
    TORCH_CHECK(uvs.is_cuda(), "uvs must be a CUDA tensor");
    TORCH_CHECK(sigma_image.is_cuda(), "sigma_image must be a CUDA tensor");
    TORCH_CHECK(gaussian_indices_per_tile.is_cuda(), "gaussian_indices_per_tile must be a CUDA tensor");
    TORCH_CHECK(num_gaussians_per_tile.is_cuda(), "num_gaussians_per_tile must be a CUDA tensor");

    TORCH_CHECK(uvs.is_contiguous(), "uvs must be contiguous");
    TORCH_CHECK(sigma_image.is_contiguous(), "sigma_image must be contiguous");
    TORCH_CHECK(gaussian_indices_per_tile.is_contiguous(), "gaussian_indices_per_tile must be contiguous");
    TORCH_CHECK(num_gaussians_per_tile.is_contiguous(), "num_gaussians_per_tile must be contiguous");

    const int N = uvs.size(0);
    TORCH_CHECK(uvs.size(1) == 2, "uvs must have shape Nx2");
    TORCH_CHECK(sigma_image.size(0) == N, "sigma_image must have shape Nx2x2");
    TORCH_CHECK(sigma_image.size(1) == 2, "sigma_image must have shape Nx2x2");
    TORCH_CHECK(sigma_image.size(2) == 2, "sigma_image must have shape Nx2x2");
    TORCH_CHECK(gaussian_indices_per_tile.size(0) == n_tiles_x * n_tiles_y, "gaussian_indices_per_tile must have shape n_tiles x max_gaussians_per_tile");
    TORCH_CHECK(num_gaussians_per_tile.size(0) == n_tiles_x * n_tiles_y, "num_gaussians_per_tile must have shape n_tiles x 1");
    TORCH_CHECK(num_gaussians_per_tile.size(1) == 1, "num_gaussians_per_tile must have shape n_tiles x 1");

    int max_gaussians_per_tile = gaussian_indices_per_tile.size(1);
    const int max_threads_per_block = 1024;
    const int num_blocks = (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    compute_tiles_kernel<<<gridsize, blocksize>>>(
        uvs.data_ptr<float>(),
        sigma_image.data_ptr<float>(),
        n_tiles_x,
        n_tiles_y,
        mh_dist,
        N,
        max_gaussians_per_tile,
        gaussian_indices_per_tile.data_ptr<int>(),
        num_gaussians_per_tile.data_ptr<int>()
    );
    cudaDeviceSynchronize();
}

__global__ void compute_tile_to_gaussian_vector_kernel(
    int* gaussian_indices_per_tile, 
    int* num_gaussians_per_tile,
    int* tile_to_gaussian_vector_offsets,
    int n_tiles,
    int max_gaussians_per_tile,
    int* tile_to_gaussian_vector
) {
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_idx >= n_tiles) {
        return;
    }
    int gaussian_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (gaussian_idx >= num_gaussians_per_tile[tile_idx]) {
        return;
    }

    int offset = tile_to_gaussian_vector_offsets[tile_idx] + gaussian_idx;
    tile_to_gaussian_vector[offset] = gaussian_indices_per_tile[tile_idx * max_gaussians_per_tile + gaussian_idx];
}


void compute_tile_to_gaussian_vector(
    torch::Tensor gaussian_indices_per_tile,
    torch::Tensor num_gaussians_per_tile,
    torch::Tensor tile_to_gaussian_vector_offsets,
    torch::Tensor tile_to_gaussian_vector
) {
    TORCH_CHECK(gaussian_indices_per_tile.is_cuda(), "gaussian_indices_per_tile must be a CUDA tensor");
    TORCH_CHECK(num_gaussians_per_tile.is_cuda(), "num_gaussians_per_tile must be a CUDA tensor");
    TORCH_CHECK(tile_to_gaussian_vector.is_cuda(), "tile_to_gaussian_vector must be a CUDA tensor");
    TORCH_CHECK(tile_to_gaussian_vector_offsets.is_cuda(), "tile_to_gaussian_vector_offsets must be a CUDA tensor");

    TORCH_CHECK(gaussian_indices_per_tile.is_contiguous(), "gaussian_indices_per_tile must be contiguous");
    TORCH_CHECK(num_gaussians_per_tile.is_contiguous(), "num_gaussians_per_tile must be contiguous");
    TORCH_CHECK(tile_to_gaussian_vector.is_contiguous(), "tile_to_gaussian_vector must be contiguous");
    TORCH_CHECK(tile_to_gaussian_vector_offsets.is_contiguous(), "tile_to_gaussian_vector_offsets must be contiguous");

    // lay out grid and block in 2d array in shape of gaussian_indices_per_tile
    dim3 blocksize(32, 32, 1);

    // x dimension is the number of tiles
    const int n_tiles = num_gaussians_per_tile.size(0);
    const int n_tile_blocks = (n_tiles + 32 - 1) / 32;

    // y dimension is the gaussian in each tile 
    const int max_gaussians_per_tile = gaussian_indices_per_tile.size(1);
    const int n_gaussian_blocks = (max_gaussians_per_tile + 32 - 1) / 32;

    dim3 gridsize(n_tile_blocks, n_gaussian_blocks, 1);

    compute_tile_to_gaussian_vector_kernel<<<gridsize, blocksize>>>(
        gaussian_indices_per_tile.data_ptr<int>(),
        num_gaussians_per_tile.data_ptr<int>(),
        tile_to_gaussian_vector_offsets.data_ptr<int>(),
        n_tiles,
        max_gaussians_per_tile,
        tile_to_gaussian_vector.data_ptr<int>()
    );
}
