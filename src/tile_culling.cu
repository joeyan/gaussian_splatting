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

    // Use square AABB for small gaussians, use OBB for large gaussians
    if (r_major < 32.0f) {
        float box_left = uvs[gaussian_idx * 2] - r_major;
        float box_right = uvs[gaussian_idx * 2] + r_major;
        float box_top = uvs[gaussian_idx * 2 + 1] - r_major;
        float box_bottom = uvs[gaussian_idx * 2 + 1] + r_major;

        // don't need to search the entire image, only need to look at all tiles within max radius of the projected center of the gaussian
        int radius_tiles = ceilf(r_major / 16.0f) + 1;

        int projected_tile_x = floorf(uvs[gaussian_idx * 2] / 16.0f);
        int start_tile_x = fmaxf(0, projected_tile_x - radius_tiles);
        int end_tile_x = fminf(n_tiles_x, projected_tile_x + radius_tiles);

        int projected_tile_y = floorf(uvs[gaussian_idx * 2 + 1] / 16.0f);
        int start_tile_y = fmaxf(0, projected_tile_y - radius_tiles);
        int end_tile_y = fminf(n_tiles_y, projected_tile_y + radius_tiles);

        // iterate through tiles
        for (int tile_x = start_tile_x; tile_x < end_tile_x; tile_x++) {
            for (int tile_y = start_tile_y; tile_y < end_tile_y; tile_y++) {
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
    } else {
        float r_minor = mh_dist * sqrtf(lambda2);

        // compute theta
        float theta;
        if (fabsf(b) < 1e-16) {
            if (a >= d) {
                theta = 0.0f;
            } else {
                theta = M_PI / 2;
            }
        } else {
            theta = atan2f(lambda1 - a, b);
        }
        const float cos_theta = cosf(theta);
        const float sin_theta = sinf(theta);

        // compute obb
        // top_left aabb [-r_major, -r_minor]
        float obb_tl_x = -1 * r_major * cos_theta + r_minor * sin_theta + uvs[gaussian_idx * 2];
        float obb_tl_y = -1 * r_major * sin_theta - r_minor * cos_theta + uvs[gaussian_idx * 2 + 1];

        // top_right aabb [r_major, -r_minor]
        float obb_tr_x = r_major * cos_theta + r_minor * sin_theta + uvs[gaussian_idx * 2];
        float obb_tr_y = r_major * sin_theta - r_minor * cos_theta + uvs[gaussian_idx * 2 + 1];

        // bottom_left aabb [-r_major, r_minor]
        float obb_bl_x = -1 * r_major * cos_theta - r_minor * sin_theta + uvs[gaussian_idx * 2];
        float obb_bl_y = -1 * r_major * sin_theta + r_minor * cos_theta + uvs[gaussian_idx * 2 + 1];

        // bottom_right aabb [r_major, r_minor]
        float obb_br_x = r_major * cos_theta - r_minor * sin_theta + uvs[gaussian_idx * 2];
        float obb_br_y = r_major * sin_theta + r_minor * cos_theta + uvs[gaussian_idx * 2 + 1];

        // don't need to search the entire image, only need to look at all tiles within max radius of the projected center of the gaussian
        int radius_tiles = ceilf(r_major / 16.0f) + 1;

        int projected_tile_x = floorf(uvs[gaussian_idx * 2] / 16.0f);
        int start_tile_x = fmaxf(0, projected_tile_x - radius_tiles);
        int end_tile_x = fminf(n_tiles_x, projected_tile_x + radius_tiles);

        int projected_tile_y = floorf(uvs[gaussian_idx * 2 + 1] / 16.0f);
        int start_tile_y = fmaxf(0, projected_tile_y - radius_tiles);
        int end_tile_y = fminf(n_tiles_y, projected_tile_y + radius_tiles);

        // iterate through tiles
        for (int tile_x = start_tile_x; tile_x < end_tile_x; tile_x++) {
            for (int tile_y = start_tile_y; tile_y < end_tile_y; tile_y++) {
                int tile_idx = tile_y * n_tiles_x + tile_x;
                // if the tile is already full, skip
                if (num_gaussians_per_tile[tile_idx] >= max_gaussians_per_tile) {
                    continue;
                }

                float tile_left = (float)tile_x * 16.0f;
                float tile_right = (float)(tile_x + 1)  * 16.0f;
                float tile_top = (float)tile_y  * 16.0f;
                float tile_bottom = (float)(tile_y + 1)  * 16.0f;

                // from split axis theorem, need overlap on all axes
                // axis0 - X axis
                float obb_min_x = fminf(fminf(obb_tl_x, obb_tr_x), fminf(obb_bl_x, obb_br_x));
                float obb_max_x = fmaxf(fmaxf(obb_tl_x, obb_tr_x), fmaxf(obb_bl_x, obb_br_x));
                if (obb_min_x > tile_right || obb_max_x < tile_left) {
                    continue;
                }
                // axis1 - Y axis
                float obb_min_y = fminf(fminf(obb_tl_y, obb_tr_y), fminf(obb_bl_y, obb_br_y));
                float obb_max_y = fmaxf(fmaxf(obb_tl_y, obb_tr_y), fmaxf(obb_bl_y, obb_br_y));
                if (obb_min_y > tile_bottom || obb_max_y < tile_top) {
                    continue;
                }
                // axis 2 - obb major axis
                float obb_major_axis_x = obb_tr_x - obb_tl_x;
                float obb_major_axis_y = obb_tr_y - obb_tl_y;
                float tl_ax2 = obb_major_axis_x * tile_left + obb_major_axis_y * tile_top; // tl
                float tr_ax2 = obb_major_axis_x * tile_right + obb_major_axis_y * tile_top; // tr
                float bl_ax2 = obb_major_axis_x * tile_left + obb_major_axis_y * tile_bottom; // bl
                float br_ax2 = obb_major_axis_x * tile_right + obb_major_axis_y * tile_bottom; // br

                float min_tile = fminf(fminf(tl_ax2, tr_ax2), fminf(bl_ax2, br_ax2));
                float max_tile = fmaxf(fmaxf(tl_ax2, tr_ax2), fmaxf(bl_ax2, br_ax2));

                // top and bottom corners of obb project to same points on ax2
                float obb_r_ax2 = obb_major_axis_x * obb_tr_x + obb_major_axis_y * obb_tr_y; // obb top right
                float obb_l_ax2 = obb_major_axis_x * obb_tl_x + obb_major_axis_y * obb_tl_y; // obb top left
                float min_obb = fminf(obb_r_ax2, obb_l_ax2);
                float max_obb = fmaxf(obb_r_ax2, obb_l_ax2);

                if (min_tile > max_obb || max_tile < min_obb) {
                    continue;
                }
                // axis 3 - obb minor axis
                float obb_minor_axis_x = obb_tr_x - obb_br_x;
                float obb_minor_axis_y = obb_tr_y - obb_br_y;
                tl_ax2 = obb_minor_axis_x * tile_left + obb_minor_axis_y * tile_top; // tl
                tr_ax2 = obb_minor_axis_x * tile_right + obb_minor_axis_y * tile_top; // tr
                bl_ax2 = obb_minor_axis_x * tile_left + obb_minor_axis_y * tile_bottom; // bl
                br_ax2 = obb_minor_axis_x * tile_right + obb_minor_axis_y * tile_bottom; // br

                min_tile = fminf(fminf(tl_ax2, tr_ax2), fminf(bl_ax2, br_ax2));
                max_tile = fmaxf(fmaxf(tl_ax2, tr_ax2), fmaxf(bl_ax2, br_ax2));

                // top and bottom corners of obb project to same points on ax2
                float obb_t_ax2 = obb_minor_axis_x * obb_tr_x + obb_minor_axis_y * obb_tr_y; // obb top right
                float obb_b_ax2 = obb_minor_axis_x * obb_br_x + obb_minor_axis_y * obb_br_y; // obb bottom right
                min_obb = fminf(obb_t_ax2, obb_b_ax2);
                max_obb = fmaxf(obb_t_ax2, obb_b_ax2);
                if (min_tile > max_obb || max_tile < min_obb) {
                    continue;
                }
                // passed all checks - there is overlap
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

__global__ void compute_splat_to_gaussian_id_vector_kernel(
    int* gaussian_indices_per_tile, 
    int* num_gaussians_per_tile,
    int* splat_to_gaussian_id_vector_offsets,
    int n_tiles,
    int max_gaussians_per_tile,
    int* splat_to_gaussian_id_vector,
    int* tile_idx_by_splat_idx
) {
    int tile_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_idx >= n_tiles) {
        return;
    }
    int gaussian_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (gaussian_idx >= num_gaussians_per_tile[tile_idx]) {
        return;
    }
    int splat_idx = splat_to_gaussian_id_vector_offsets[tile_idx] + gaussian_idx;
    splat_to_gaussian_id_vector[splat_idx] = gaussian_indices_per_tile[tile_idx * max_gaussians_per_tile + gaussian_idx];
    tile_idx_by_splat_idx[splat_idx] = tile_idx;
}


void compute_splat_to_gaussian_id_vector_cuda(
    torch::Tensor gaussian_indices_per_tile,
    torch::Tensor num_gaussians_per_tile,
    torch::Tensor splat_to_gaussian_id_vector_offsets,
    torch::Tensor splat_to_gaussian_id_vector,
    torch::Tensor tile_idx_by_splat_idx
) {
    TORCH_CHECK(gaussian_indices_per_tile.is_cuda(), "gaussian_indices_per_tile must be a CUDA tensor");
    TORCH_CHECK(num_gaussians_per_tile.is_cuda(), "num_gaussians_per_tile must be a CUDA tensor");
    TORCH_CHECK(splat_to_gaussian_id_vector.is_cuda(), "splat_to_gaussian_id_vector must be a CUDA tensor");
    TORCH_CHECK(splat_to_gaussian_id_vector_offsets.is_cuda(), "splat_to_gaussian_id_vector_offsets must be a CUDA tensor");
    TORCH_CHECK(tile_idx_by_splat_idx.is_cuda(), "tile_idx_by_splat_idx must be a CUDA tensor");


    TORCH_CHECK(gaussian_indices_per_tile.is_contiguous(), "gaussian_indices_per_tile must be contiguous");
    TORCH_CHECK(num_gaussians_per_tile.is_contiguous(), "num_gaussians_per_tile must be contiguous");
    TORCH_CHECK(splat_to_gaussian_id_vector.is_contiguous(), "splat_to_gaussian_id_vector must be contiguous");
    TORCH_CHECK(splat_to_gaussian_id_vector_offsets.is_contiguous(), "splat_to_gaussian_id_vector_offsets must be contiguous");
    TORCH_CHECK(tile_idx_by_splat_idx.is_contiguous(), "tile_idx_by_splat_idx must be contiguous");

    // lay out grid and block in 2d array in shape of gaussian_indices_per_tile
    dim3 blocksize(32, 32, 1);

    // x dimension is the number of tiles
    const int n_tiles = num_gaussians_per_tile.size(0);
    const int n_tile_blocks = (n_tiles + 32 - 1) / 32;

    // y dimension is the gaussian in each tile 
    const int max_gaussians_per_tile = gaussian_indices_per_tile.size(1);
    const int n_gaussian_blocks = (max_gaussians_per_tile + 32 - 1) / 32;

    dim3 gridsize(n_tile_blocks, n_gaussian_blocks, 1);

    compute_splat_to_gaussian_id_vector_kernel<<<gridsize, blocksize>>>(
        gaussian_indices_per_tile.data_ptr<int>(),
        num_gaussians_per_tile.data_ptr<int>(),
        splat_to_gaussian_id_vector_offsets.data_ptr<int>(),
        n_tiles,
        max_gaussians_per_tile,
        splat_to_gaussian_id_vector.data_ptr<int>(),
        tile_idx_by_splat_idx.data_ptr<int>()
    );
    cudaDeviceSynchronize();
}
