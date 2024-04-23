#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "checks.cuh"

// returns true if there is overlap between the two bboxes, false otherwise
__device__ __forceinline__ bool split_axis_test(
    const float* __restrict__ obb,        // [tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y]
    const float* __restrict__ tile_bounds // [left, right, top, bottom]
) {
    // from split axis theorem, need overlap on all axes
    // axis0 - X axis
    const float obb_min_x = fminf(fminf(obb[0], obb[2]), fminf(obb[4], obb[6]));
    const float obb_max_x = fmaxf(fmaxf(obb[0], obb[2]), fmaxf(obb[4], obb[6]));
    if (obb_min_x > tile_bounds[1] || obb_max_x < tile_bounds[0]) {
        return false;
    }
    // axis1 - Y axis
    const float obb_min_y = fminf(fminf(obb[1], obb[3]), fminf(obb[5], obb[7]));
    const float obb_max_y = fmaxf(fmaxf(obb[1], obb[3]), fmaxf(obb[5], obb[7]));
    if (obb_min_y > tile_bounds[3] || obb_max_y < tile_bounds[2]) {
        return false;
    }
    // axis 2 - obb major axis
    const float obb_major_axis_x = obb[2] - obb[0];
    const float obb_major_axis_y = obb[3] - obb[1];
    float tl_ax2 = obb_major_axis_x * tile_bounds[0] + obb_major_axis_y * tile_bounds[2]; // tl
    float tr_ax2 = obb_major_axis_x * tile_bounds[1] + obb_major_axis_y * tile_bounds[2]; // tr
    float bl_ax2 = obb_major_axis_x * tile_bounds[0] + obb_major_axis_y * tile_bounds[3]; // bl
    float br_ax2 = obb_major_axis_x * tile_bounds[1] + obb_major_axis_y * tile_bounds[3]; // br

    float min_tile = fminf(fminf(tl_ax2, tr_ax2), fminf(bl_ax2, br_ax2));
    float max_tile = fmaxf(fmaxf(tl_ax2, tr_ax2), fmaxf(bl_ax2, br_ax2));

    // top and bottom corners of obb project to same points on ax2
    const float obb_r_ax2 = obb_major_axis_x * obb[2] + obb_major_axis_y * obb[3]; // obb top right
    const float obb_l_ax2 = obb_major_axis_x * obb[0] + obb_major_axis_y * obb[1]; // obb top left
    float min_obb = fminf(obb_r_ax2, obb_l_ax2);
    float max_obb = fmaxf(obb_r_ax2, obb_l_ax2);

    if (min_tile > max_obb || max_tile < min_obb) {
        return false;
    }
    // axis 3 - obb minor axis
    const float obb_minor_axis_x = obb[2] - obb[6];
    const float obb_minor_axis_y = obb[3] - obb[7];
    tl_ax2 = obb_minor_axis_x * tile_bounds[0] + obb_minor_axis_y * tile_bounds[2]; // tl
    tr_ax2 = obb_minor_axis_x * tile_bounds[1] + obb_minor_axis_y * tile_bounds[2]; // tr
    bl_ax2 = obb_minor_axis_x * tile_bounds[0] + obb_minor_axis_y * tile_bounds[3]; // bl
    br_ax2 = obb_minor_axis_x * tile_bounds[1] + obb_minor_axis_y * tile_bounds[3]; // br

    min_tile = fminf(fminf(tl_ax2, tr_ax2), fminf(bl_ax2, br_ax2));
    max_tile = fmaxf(fmaxf(tl_ax2, tr_ax2), fmaxf(bl_ax2, br_ax2));

    // top and bottom corners of obb project to same points on ax2
    const float obb_t_ax2 = obb_minor_axis_x * obb[2] + obb_minor_axis_y * obb[3]; // obb top right
    const float obb_b_ax2 =
        obb_minor_axis_x * obb[6] + obb_minor_axis_y * obb[7]; // obb bottom right
    min_obb = fminf(obb_t_ax2, obb_b_ax2);
    max_obb = fmaxf(obb_t_ax2, obb_b_ax2);
    if (min_tile > max_obb || max_tile < min_obb) {
        return false;
    }
    return true;
}

// returns tile search radius and computes oriented bounding box
__device__ __forceinline__ int compute_obb(
    const float u,
    const float v,
    const float a,
    const float b,
    const float c,
    const float mh_dist,
    float* obb // [tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y]
) {
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
    obb[0] = -1 * r_major * cos_theta + r_minor * sin_theta + u;
    obb[1] = -1 * r_major * sin_theta - r_minor * cos_theta + v;

    // top_right aabb [r_major, -r_minor]
    obb[2] = r_major * cos_theta + r_minor * sin_theta + u;
    obb[3] = r_major * sin_theta - r_minor * cos_theta + v;

    // bottom_left aabb [-r_major, r_minor]
    obb[4] = -1 * r_major * cos_theta - r_minor * sin_theta + u;
    obb[5] = -1 * r_major * sin_theta + r_minor * cos_theta + v;

    // bottom_right aabb [r_major, r_minor]
    obb[6] = r_major * cos_theta - r_minor * sin_theta + u;
    obb[7] = r_major * sin_theta + r_minor * cos_theta + v;

    // don't need to search the entire image, only need to look at all tiles
    // within max radius of the projected center of the gaussian
    const int radius_tiles = ceilf(r_major / 16.0f) + 1;
    return radius_tiles;
}

__global__ void compute_num_splats_kernel(
    const float* __restrict__ uvs,
    const float* __restrict__ conic,
    const int n_tiles_x,
    const int n_tiles_y,
    const float mh_dist,
    const int N,
    int* __restrict__ num_tiles_per_gaussian,
    int* __restrict__ num_gaussians_per_tile
) {
    int gaussian_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gaussian_idx >= N) {
        return;
    }

    const float u = uvs[gaussian_idx * 2];
    const float v = uvs[gaussian_idx * 2 + 1];

    const float a = conic[gaussian_idx * 3] + 0.25f;
    const float b = conic[gaussian_idx * 3 + 1] / 2.0f;
    const float c = conic[gaussian_idx * 3 + 2] + 0.25f;

    float obb[8];
    const int radius_tiles = compute_obb(u, v, a, b, c, mh_dist, obb);

    const int projected_tile_x = floorf(u / 16.0f);
    const int start_tile_x = fmaxf(0, projected_tile_x - radius_tiles);
    const int end_tile_x = fminf(n_tiles_x, projected_tile_x + radius_tiles);

    const int projected_tile_y = floorf(v / 16.0f);
    const int start_tile_y = fmaxf(0, projected_tile_y - radius_tiles);
    const int end_tile_y = fminf(n_tiles_y, projected_tile_y + radius_tiles);

    int n_tiles = 0;
    // iterate through tiles
    for (int tile_x = start_tile_x; tile_x < end_tile_x; tile_x++) {
        for (int tile_y = start_tile_y; tile_y < end_tile_y; tile_y++) {
            const int tile_idx = tile_y * n_tiles_x + tile_x;

            float tile_bounds[4]; // [left, right, top, bottom]
            tile_bounds[0] = __int2float_rn(tile_x) * 16.0f;
            tile_bounds[1] = __int2float_rn(tile_x + 1) * 16.0f;
            tile_bounds[2] = __int2float_rn(tile_y) * 16.0f;
            tile_bounds[3] = __int2float_rn(tile_y + 1) * 16.0f;

            if (split_axis_test(obb, tile_bounds)) {
                // update tile counts
                atomicAdd(num_gaussians_per_tile + tile_idx, 1);
                n_tiles++;
            }
        }
    }
    num_tiles_per_gaussian[gaussian_idx] = n_tiles;
}

__global__ void compute_tiles_kernel(
    const float* __restrict__ uvs,
    const float* __restrict__ xyz_camera_frame,
    const float* __restrict__ conic,
    const int* __restrict__ splat_start_end_idx_by_gaussian_idx,
    const int n_tiles_x,
    const int n_tiles_y,
    const float mh_dist,
    const int N,
    const double tile_idx_key_multiplier,
    int* __restrict__ gaussian_idx_by_splat_idx,
    double* __restrict__ sort_keys
) {
    int gaussian_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gaussian_idx >= N) {
        return;
    }

    // get per gaussian values
    const float u = uvs[gaussian_idx * 2];
    const float v = uvs[gaussian_idx * 2 + 1];
    const double z = (double)(xyz_camera_frame[gaussian_idx * 3 + 2]);

    const float a = conic[gaussian_idx * 3] + 0.25f;
    const float b = conic[gaussian_idx * 3 + 1] / 2.0f;
    const float c = conic[gaussian_idx * 3 + 2] + 0.25f;

    const int output_start_idx = splat_start_end_idx_by_gaussian_idx[gaussian_idx];
    const int output_end_idx = splat_start_end_idx_by_gaussian_idx[gaussian_idx + 1];

    float obb[8];
    const int radius_tiles = compute_obb(u, v, a, b, c, mh_dist, obb);

    const int projected_tile_x = floorf(u / 16.0f);
    const int start_tile_x = fmaxf(0, projected_tile_x - radius_tiles);
    const int end_tile_x = fminf(n_tiles_x, projected_tile_x + radius_tiles);

    const int projected_tile_y = floorf(v / 16.0f);
    const int start_tile_y = fmaxf(0, projected_tile_y - radius_tiles);
    const int end_tile_y = fminf(n_tiles_y, projected_tile_y + radius_tiles);

    int n_tiles = 0;
    // iterate through tiles
    for (int tile_x = start_tile_x; tile_x < end_tile_x; tile_x++) {
        for (int tile_y = start_tile_y; tile_y < end_tile_y; tile_y++) {
            const int tile_idx = tile_y * n_tiles_x + tile_x;

            float tile_bounds[4]; // [left, right, top, bottom]
            tile_bounds[0] = __int2float_rn(tile_x) * 16.0f;
            tile_bounds[1] = __int2float_rn(tile_x + 1) * 16.0f;
            tile_bounds[2] = __int2float_rn(tile_y) * 16.0f;
            tile_bounds[3] = __int2float_rn(tile_y + 1) * 16.0f;

            if (split_axis_test(obb, tile_bounds) &&
                (output_start_idx + n_tiles) < output_end_idx) {
                // update gaussian index by splat index
                gaussian_idx_by_splat_idx[output_start_idx + n_tiles] = gaussian_idx;
                sort_keys[output_start_idx + n_tiles] =
                    z + tile_idx_key_multiplier * __int2double_rn(tile_idx);
                n_tiles++;
            }
        }
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

    // compute number of splats per gaussian/tile
    torch::Tensor num_tiles_per_gaussian =
        torch::zeros({N}, torch::dtype(torch::kInt32).device(uvs.device()));

    torch::Tensor num_gaussians_per_tile =
        torch::zeros({n_tiles_x * n_tiles_y}, torch::dtype(torch::kInt32).device(uvs.device()));

    compute_num_splats_kernel<<<gridsize, blocksize>>>(
        uvs.data_ptr<float>(),
        conic.data_ptr<float>(),
        n_tiles_x,
        n_tiles_y,
        mh_dist,
        N,
        num_tiles_per_gaussian.data_ptr<int>(),
        num_gaussians_per_tile.data_ptr<int>()
    );
    cudaDeviceSynchronize();

    // create vector of gaussian indices for each splat and sort keys
    torch::Tensor cumsum = num_tiles_per_gaussian.cumsum(0);
    torch::Tensor splat_start_end_idx_by_gaussian_idx = torch::cat(
        {torch::zeros({1}, torch::dtype(torch::kInt32).device(uvs.device())),
         cumsum.to(torch::kInt32)},
        0
    );

    // create output gaussian idx vector and sort key vector
    const int num_splats = splat_start_end_idx_by_gaussian_idx[N].item<int>();
    torch::Tensor gaussian_idx_by_splat_idx =
        torch::zeros({num_splats}, torch::dtype(torch::kInt32).device(uvs.device()));
    torch::Tensor sort_keys =
        torch::zeros({num_splats}, torch::dtype(torch::kFloat64).device(uvs.device()));

    CHECK_FLOAT_TENSOR(xyz_camera_frame);
    CHECK_INT_TENSOR(splat_start_end_idx_by_gaussian_idx);
    CHECK_INT_TENSOR(gaussian_idx_by_splat_idx);
    CHECK_DOUBLE_TENSOR(sort_keys);

    // max_depth + 1.0
    const double tile_idx_key_multiplier =
        (double)(xyz_camera_frame.select(1, 2).max().item<float>() + 1.0f);

    // compute gaussian index and key for each gaussian-tile intersection
    compute_tiles_kernel<<<gridsize, blocksize>>>(
        uvs.data_ptr<float>(),
        xyz_camera_frame.data_ptr<float>(),
        conic.data_ptr<float>(),
        splat_start_end_idx_by_gaussian_idx.data_ptr<int>(),
        n_tiles_x,
        n_tiles_y,
        mh_dist,
        N,
        tile_idx_key_multiplier,
        gaussian_idx_by_splat_idx.data_ptr<int>(),
        sort_keys.data_ptr<double>()
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
