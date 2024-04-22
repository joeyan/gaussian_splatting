#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "checks.cuh"

__global__ void compute_tiles_kernel(
    const float* __restrict__ uvs,
    const float* __restrict__ conic,
    const int n_tiles_x,
    const int n_tiles_y,
    const float mh_dist,
    const int N,
    const int max_tiles_per_gaussian,
    int* __restrict__ tile_indices_per_gaussian,
    int* __restrict__ num_tiles_per_gaussian,
    int* __restrict__ num_gaussians_per_tile
) {
    int gaussian_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gaussian_idx >= N) {
        return;
    }

    const float a = conic[gaussian_idx * 3] + 0.25f;
    const float b = conic[gaussian_idx * 3 + 1] / 2.0f;
    const float c = conic[gaussian_idx * 3 + 2] + 0.25f;

    // compute major axis radius of ellipse
    const float left = (a + c) / 2;
    const float right = sqrtf((a - c) * (a - c) / 4.0f + b * b);
    const float lambda1 = left + right;
    const float lambda2 = left - right;

    const float r_major = mh_dist * sqrtf(lambda1);

    const float r_minor = mh_dist * sqrtf(lambda2);

    // compute theta
    float theta;
    if (fabsf(b) < 1e-16) {
        if (a >= c) {
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

    // don't need to search the entire image, only need to look at all tiles
    // within max radius of the projected center of the gaussian
    int radius_tiles = ceilf(r_major / 16.0f) + 1;

    if (radius_tiles > 16) {
        return;
    }

    int projected_tile_x = floorf(uvs[gaussian_idx * 2] / 16.0f);
    int start_tile_x = fmaxf(0, projected_tile_x - radius_tiles);
    int end_tile_x = fminf(n_tiles_x, projected_tile_x + radius_tiles);

    int projected_tile_y = floorf(uvs[gaussian_idx * 2 + 1] / 16.0f);
    int start_tile_y = fmaxf(0, projected_tile_y - radius_tiles);
    int end_tile_y = fminf(n_tiles_y, projected_tile_y + radius_tiles);

    int n_tiles = 0;
    // iterate through tiles
    for (int tile_x = start_tile_x; tile_x < end_tile_x; tile_x++) {
        for (int tile_y = start_tile_y; tile_y < end_tile_y; tile_y++) {
            int tile_idx = tile_y * n_tiles_x + tile_x;

            float tile_left = (float)tile_x * 16.0f;
            float tile_right = (float)(tile_x + 1) * 16.0f;
            float tile_top = (float)tile_y * 16.0f;
            float tile_bottom = (float)(tile_y + 1) * 16.0f;

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
            float tl_ax2 = obb_major_axis_x * tile_left + obb_major_axis_y * tile_top;     // tl
            float tr_ax2 = obb_major_axis_x * tile_right + obb_major_axis_y * tile_top;    // tr
            float bl_ax2 = obb_major_axis_x * tile_left + obb_major_axis_y * tile_bottom;  // bl
            float br_ax2 = obb_major_axis_x * tile_right + obb_major_axis_y * tile_bottom; // br

            float min_tile = fminf(fminf(tl_ax2, tr_ax2), fminf(bl_ax2, br_ax2));
            float max_tile = fmaxf(fmaxf(tl_ax2, tr_ax2), fmaxf(bl_ax2, br_ax2));

            // top and bottom corners of obb project to same points on ax2
            float obb_r_ax2 =
                obb_major_axis_x * obb_tr_x + obb_major_axis_y * obb_tr_y; // obb top right
            float obb_l_ax2 =
                obb_major_axis_x * obb_tl_x + obb_major_axis_y * obb_tl_y; // obb top left
            float min_obb = fminf(obb_r_ax2, obb_l_ax2);
            float max_obb = fmaxf(obb_r_ax2, obb_l_ax2);

            if (min_tile > max_obb || max_tile < min_obb) {
                continue;
            }
            // axis 3 - obb minor axis
            float obb_minor_axis_x = obb_tr_x - obb_br_x;
            float obb_minor_axis_y = obb_tr_y - obb_br_y;
            tl_ax2 = obb_minor_axis_x * tile_left + obb_minor_axis_y * tile_top;     // tl
            tr_ax2 = obb_minor_axis_x * tile_right + obb_minor_axis_y * tile_top;    // tr
            bl_ax2 = obb_minor_axis_x * tile_left + obb_minor_axis_y * tile_bottom;  // bl
            br_ax2 = obb_minor_axis_x * tile_right + obb_minor_axis_y * tile_bottom; // br

            min_tile = fminf(fminf(tl_ax2, tr_ax2), fminf(bl_ax2, br_ax2));
            max_tile = fmaxf(fmaxf(tl_ax2, tr_ax2), fmaxf(bl_ax2, br_ax2));

            // top and bottom corners of obb project to same points on ax2
            float obb_t_ax2 =
                obb_minor_axis_x * obb_tr_x + obb_minor_axis_y * obb_tr_y; // obb top right
            float obb_b_ax2 =
                obb_minor_axis_x * obb_br_x + obb_minor_axis_y * obb_br_y; // obb bottom right
            min_obb = fminf(obb_t_ax2, obb_b_ax2);
            max_obb = fmaxf(obb_t_ax2, obb_b_ax2);
            if (min_tile > max_obb || max_tile < min_obb) {
                continue;
            }
            // passed all checks - there is overlap

            // increment num_gaussians per tile
            if (n_tiles < max_tiles_per_gaussian) {
                // write tile index
                atomicAdd(num_gaussians_per_tile + tile_idx, 1);
                tile_indices_per_gaussian[gaussian_idx * max_tiles_per_gaussian + n_tiles] =
                    tile_idx;
                n_tiles++;
            } else {
                break;
            }
        }
        if (n_tiles >= max_tiles_per_gaussian) {
            break;
        }
    }
    if (n_tiles < max_tiles_per_gaussian) {
        num_tiles_per_gaussian[gaussian_idx] = n_tiles;
    } else {
        // don't render gaussians that cover more than the number of tiles
        num_tiles_per_gaussian[gaussian_idx] = n_tiles;
        printf(
            "Gaussian %d greater than max_tiles_per_gaussian, has radius %d\n",
            gaussian_idx,
            radius_tiles
        );
    }
}

__global__ void generate_gaussian_and_sort_list_kernel(
    const float* __restrict__ xyz_camera_frame,
    const int* __restrict__ tile_indices_per_gaussian,
    const int* __restrict__ splat_start_end_idx_by_gaussian_idx,
    const int n_gaussians,
    const int max_tiles_per_gaussian,
    const float tile_idx_key_multiplier,
    int* __restrict__ gaussian_idx_by_splat_idx,
    float* __restrict__ sort_keys
) {
    // get gaussian index
    const int gaussian_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gaussian_idx >= n_gaussians) {
        return;
    }

    const float z = xyz_camera_frame[gaussian_idx * 3 + 2];

    // output start / stop indices
    const int output_start_idx = splat_start_end_idx_by_gaussian_idx[gaussian_idx];
    const int output_end_idx = splat_start_end_idx_by_gaussian_idx[gaussian_idx + 1];

    for (int i = 0; i < output_end_idx - output_start_idx; i++) {
        const int tile_idx = tile_indices_per_gaussian[gaussian_idx * max_tiles_per_gaussian + i];
        gaussian_idx_by_splat_idx[output_start_idx + i] = gaussian_idx;
        // sort key is depth + (max_depth + 1.0) * tile_idx
        sort_keys[output_start_idx + i] = z + tile_idx_key_multiplier * __int2float_rn(tile_idx);
    }
}

std::tuple<torch::Tensor, torch::Tensor> get_sorted_gaussian_list(
    const int max_tiles_per_gaussian,
    torch::Tensor uvs,
    torch::Tensor xyz_camera_frame,
    torch::Tensor conic,
    const int n_tiles_x,
    const int n_tiles_y,
    const float mh_dist
) {
    CHECK_VALID_INPUT(uvs);
    CHECK_VALID_INPUT(xyz_camera_frame);
    CHECK_VALID_INPUT(conic);

    CHECK_FLOAT_TENSOR(uvs);
    CHECK_FLOAT_TENSOR(xyz_camera_frame);
    CHECK_FLOAT_TENSOR(conic);

    const int N = uvs.size(0);

    const int max_threads_per_block = 1024;
    const int num_blocks = (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    // create tensors for output
    torch::Tensor tile_indices_per_gaussian =
        torch::zeros({N, max_tiles_per_gaussian}, torch::dtype(torch::kInt32).device(uvs.device()));
    torch::Tensor num_tiles_per_gaussian =
        torch::zeros({N}, torch::dtype(torch::kInt32).device(uvs.device()));
    torch::Tensor num_gaussians_per_tile =
        torch::zeros({n_tiles_x * n_tiles_y}, torch::dtype(torch::kInt32).device(uvs.device()));

    compute_tiles_kernel<<<gridsize, blocksize>>>(
        uvs.data_ptr<float>(),
        conic.data_ptr<float>(),
        n_tiles_x,
        n_tiles_y,
        mh_dist,
        N,
        max_tiles_per_gaussian,
        tile_indices_per_gaussian.data_ptr<int>(),
        num_tiles_per_gaussian.data_ptr<int>(),
        num_gaussians_per_tile.data_ptr<int>()
    );
    cudaDeviceSynchronize();

    // create intermediate values
    torch::Tensor cumsum = num_tiles_per_gaussian.cumsum(0);
    torch::Tensor splat_start_end_idx_by_gaussian_idx = torch::cat(
        {torch::zeros({1}, torch::dtype(torch::kInt32).device(uvs.device())),
         cumsum.to(torch::kInt32)},
        0
    );
    CHECK_INT_TENSOR(splat_start_end_idx_by_gaussian_idx);

    const int num_splats = splat_start_end_idx_by_gaussian_idx[N].item<int>();

    // max_depth + 1.0
    const float tile_idx_key_multiplier = xyz_camera_frame.select(1, 2).max().item<float>() + 1.0f;

    // outputs
    torch::Tensor gaussian_idx_by_splat_idx =
        torch::zeros({num_splats}, torch::dtype(torch::kInt32).device(uvs.device()));
    torch::Tensor sort_keys =
        torch::zeros({num_splats}, torch::dtype(torch::kFloat32).device(uvs.device()));

    CHECK_FLOAT_TENSOR(xyz_camera_frame);
    CHECK_INT_TENSOR(tile_indices_per_gaussian);
    CHECK_INT_TENSOR(splat_start_end_idx_by_gaussian_idx);
    CHECK_INT_TENSOR(gaussian_idx_by_splat_idx);
    CHECK_FLOAT_TENSOR(sort_keys);

    generate_gaussian_and_sort_list_kernel<<<gridsize, blocksize>>>(
        xyz_camera_frame.data_ptr<float>(),
        tile_indices_per_gaussian.data_ptr<int>(),
        splat_start_end_idx_by_gaussian_idx.data_ptr<int>(),
        N,
        max_tiles_per_gaussian,
        tile_idx_key_multiplier,
        gaussian_idx_by_splat_idx.data_ptr<int>(),
        sort_keys.data_ptr<float>()
    );
    cudaDeviceSynchronize();

    auto result = torch::sort(sort_keys);
    torch::Tensor sorted_indices = std::get<1>(result);
    torch::Tensor sorted_gaussians = gaussian_idx_by_splat_idx.index_select(0, sorted_indices);

    // compute indices for each tile
    torch::Tensor num_gaussians_per_tile_cumsum = num_gaussians_per_tile.cumsum(0);
    torch::Tensor splat_start_end_idx_by_tile_idx = torch::cat(
        {torch::zeros({1}, torch::dtype(torch::kInt32).device(uvs.device())),
         num_gaussians_per_tile_cumsum.to(torch::kInt32)},
        0
    );

    return std::make_tuple(sorted_gaussians, splat_start_end_idx_by_tile_idx);
}
