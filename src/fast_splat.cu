#include <torch/extension.h>

#include "checks.cuh"
#include "spherical_harmonics.cuh"

template <typename T, unsigned int N_SH>
__global__ void precompute_rgb_from_sh_kernel(
    const T* __restrict__ xyz,
    const T* __restrict__ sh_coeff,
    const T* __restrict__ camera_T_world,
    const unsigned int N,
    T* __restrict__ rgb
) {
    const int gaussian_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gaussian_idx >= N) {
        return;
    }

    if (N_SH == 1) {
        #pragma unroll
        for (int channel = 0; channel < 3; channel++) {
            rgb[gaussian_idx * 3 + channel] = sh_coeff[gaussian_idx * 3 + channel];
        }
    } else {
        // compute normalized view direction
        const T camera_center[3] = {
            camera_T_world[0 * 4 + 3], camera_T_world[1 * 4 + 3], camera_T_world[2 * 4 + 3]};
        T view_dir[3] = {
            xyz[gaussian_idx * 3 + 0] - camera_center[0],
            xyz[gaussian_idx * 3 + 1] - camera_center[1],
            xyz[gaussian_idx * 3 + 2] - camera_center[2]};
        const T r_view_dir_norm = rsqrt(
            view_dir[0] * view_dir[0] + view_dir[1] * view_dir[1] + view_dir[2] * view_dir[2]
        );
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            view_dir[i] *= r_view_dir_norm;
        }

        T sh_at_view_dir[N_SH];
        compute_sh_coeffs_for_view_dir<T, N_SH>(view_dir, sh_at_view_dir);

        #pragma unroll
        for (int channel = 0; channel < 3; channel++) {
            T temp_rgb = 0.0;
            #pragma unroll
            for (int sh_idx = 0; sh_idx < N_SH; sh_idx++) {
                temp_rgb += sh_at_view_dir[sh_idx] *
                            sh_coeff[gaussian_idx * N_SH * 3 + N_SH * channel + sh_idx];
            }
            // divide by SH_0 to maintain compatibility with downstream rasterizer
            temp_rgb *= r_SH_0;
            // set value on output
            rgb[gaussian_idx * 3 + channel] = temp_rgb;
        }
    }
}

template <typename T, unsigned int N_SH>
__global__ void precompute_rgb_from_sh_backward_kernel(
    const T* __restrict__ xyz,
    const T* __restrict__ camera_T_world,
    const T* __restrict__ grad_rgb,
    const unsigned int N,
    T* __restrict__ grad_sh
) {
    const int gaussian_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gaussian_idx >= N) {
        return;
    }
    if (N_SH == 1) {
        #pragma unroll
        for (int channel = 0; channel < 3; channel++) {
            grad_sh[gaussian_idx * 3 + channel] = grad_rgb[gaussian_idx * 3 + channel];
        }
    } else {
        // compute normalized view direction
        const T camera_center[3] = {
            camera_T_world[0 * 4 + 3], camera_T_world[1 * 4 + 3], camera_T_world[2 * 4 + 3]};
        T view_dir[3] = {
            xyz[gaussian_idx * 3 + 0] - camera_center[0],
            xyz[gaussian_idx * 3 + 1] - camera_center[1],
            xyz[gaussian_idx * 3 + 2] - camera_center[2]};
        const T r_view_dir_norm = rsqrt(
            view_dir[0] * view_dir[0] + view_dir[1] * view_dir[1] + view_dir[2] * view_dir[2]
        );
        #pragma unroll
        for (int i = 0; i < 3; ++i) {
            view_dir[i] *= r_view_dir_norm;
        }

        T sh_at_view_dir[N_SH];
        compute_sh_coeffs_for_view_dir<T, N_SH>(view_dir, sh_at_view_dir);

        // make local copy and undo scaling by SH_0
        T grad_rgb_local[3] = {
            grad_rgb[gaussian_idx * 3 + 0] * r_SH_0,
            grad_rgb[gaussian_idx * 3 + 1] * r_SH_0,
            grad_rgb[gaussian_idx * 3 + 2] * r_SH_0};

        #pragma unroll
        for (int channel = 0; channel < 3; channel++) {
            #pragma unroll
            for (int sh_idx = 0; sh_idx < N_SH; sh_idx++) {
                grad_sh[gaussian_idx * N_SH * 3 + N_SH * channel + sh_idx] =
                    grad_rgb_local[channel] * sh_at_view_dir[sh_idx];
            }
        }
    }
}

void precompute_rgb_from_sh_cuda(
    const torch::Tensor xyz,
    const torch::Tensor sh_coeff,
    const torch::Tensor camera_T_world,
    torch::Tensor rgb
) {
    CHECK_VALID_INPUT(xyz);
    CHECK_VALID_INPUT(sh_coeff);
    CHECK_VALID_INPUT(camera_T_world);
    CHECK_VALID_INPUT(rgb);

    const int N = xyz.size(0);
    int num_sh_coeff;
    if (sh_coeff.dim() == 3) {
        num_sh_coeff = sh_coeff.size(2);
    } else {
        num_sh_coeff = 1;
    }

    const int max_threads_per_block = 1024;
    const int num_blocks = (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (xyz.dtype() == torch::kFloat32) {
        if (num_sh_coeff == 1) {
            precompute_rgb_from_sh_kernel<float, 1><<<gridsize, blocksize>>>(
                xyz.data_ptr<float>(),
                sh_coeff.data_ptr<float>(),
                camera_T_world.data_ptr<float>(),
                N,
                rgb.data_ptr<float>()
            );
        } else if (num_sh_coeff == 4) {
            precompute_rgb_from_sh_kernel<float, 4><<<gridsize, blocksize>>>(
                xyz.data_ptr<float>(),
                sh_coeff.data_ptr<float>(),
                camera_T_world.data_ptr<float>(),
                N,
                rgb.data_ptr<float>()
            );
        } else if (num_sh_coeff == 9) {
            precompute_rgb_from_sh_kernel<float, 9><<<gridsize, blocksize>>>(
                xyz.data_ptr<float>(),
                sh_coeff.data_ptr<float>(),
                camera_T_world.data_ptr<float>(),
                N,
                rgb.data_ptr<float>()
            );
        } else if (num_sh_coeff == 16) {
            precompute_rgb_from_sh_kernel<float, 16><<<gridsize, blocksize>>>(
                xyz.data_ptr<float>(),
                sh_coeff.data_ptr<float>(),
                camera_T_world.data_ptr<float>(),
                N,
                rgb.data_ptr<float>()
            );
        } else {
            AT_ERROR("Unsupported number of SH coefficients: ", num_sh_coeff);
        }
    } else if (xyz.dtype() == torch::kFloat64) {
        if (num_sh_coeff == 1) {
            precompute_rgb_from_sh_kernel<double, 1><<<gridsize, blocksize>>>(
                xyz.data_ptr<double>(),
                sh_coeff.data_ptr<double>(),
                camera_T_world.data_ptr<double>(),
                N,
                rgb.data_ptr<double>()
            );
        } else if (num_sh_coeff == 4) {
            precompute_rgb_from_sh_kernel<double, 4><<<gridsize, blocksize>>>(
                xyz.data_ptr<double>(),
                sh_coeff.data_ptr<double>(),
                camera_T_world.data_ptr<double>(),
                N,
                rgb.data_ptr<double>()
            );
        } else if (num_sh_coeff == 9) {
            precompute_rgb_from_sh_kernel<double, 9><<<gridsize, blocksize>>>(
                xyz.data_ptr<double>(),
                sh_coeff.data_ptr<double>(),
                camera_T_world.data_ptr<double>(),
                N,
                rgb.data_ptr<double>()
            );
        } else if (num_sh_coeff == 16) {
            precompute_rgb_from_sh_kernel<double, 16><<<gridsize, blocksize>>>(
                xyz.data_ptr<double>(),
                sh_coeff.data_ptr<double>(),
                camera_T_world.data_ptr<double>(),
                N,
                rgb.data_ptr<double>()
            );
        } else {
            AT_ERROR("Unsupported number of SH coefficients: ", num_sh_coeff);
        }
    } else {
        AT_ERROR("Unsupported data type: ", xyz.dtype());
    }
    cudaDeviceSynchronize();
}

void precompute_rgb_from_sh_backward_cuda(
    const torch::Tensor xyz,
    const torch::Tensor camera_T_world,
    const torch::Tensor grad_rgb,
    torch::Tensor grad_sh
) {
    CHECK_VALID_INPUT(xyz);
    CHECK_VALID_INPUT(camera_T_world);
    CHECK_VALID_INPUT(grad_rgb);
    CHECK_VALID_INPUT(grad_sh);

    const int N = xyz.size(0);
    int num_sh_coeff;
    if (grad_sh.dim() == 3) {
        num_sh_coeff = grad_sh.size(2);
    } else {
        num_sh_coeff = 1;
    }

    const int max_threads_per_block = 1024;
    const int num_blocks = (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (xyz.dtype() == torch::kFloat32) {
        if (num_sh_coeff == 1) {
            precompute_rgb_from_sh_backward_kernel<float, 1><<<gridsize, blocksize>>>(
                xyz.data_ptr<float>(),
                camera_T_world.data_ptr<float>(),
                grad_rgb.data_ptr<float>(),
                N,
                grad_sh.data_ptr<float>()
            );
        } else if (num_sh_coeff == 4) {
            precompute_rgb_from_sh_backward_kernel<float, 4><<<gridsize, blocksize>>>(
                xyz.data_ptr<float>(),
                camera_T_world.data_ptr<float>(),
                grad_rgb.data_ptr<float>(),
                N,
                grad_sh.data_ptr<float>()
            );
        } else if (num_sh_coeff == 9) {
            precompute_rgb_from_sh_backward_kernel<float, 9><<<gridsize, blocksize>>>(
                xyz.data_ptr<float>(),
                camera_T_world.data_ptr<float>(),
                grad_rgb.data_ptr<float>(),
                N,
                grad_sh.data_ptr<float>()
            );
        } else if (num_sh_coeff == 16) {
            precompute_rgb_from_sh_backward_kernel<float, 16><<<gridsize, blocksize>>>(
                xyz.data_ptr<float>(),
                camera_T_world.data_ptr<float>(),
                grad_rgb.data_ptr<float>(),
                N,
                grad_sh.data_ptr<float>()
            );
        } else {
            AT_ERROR("Unsupported number of SH coefficients: ", num_sh_coeff);
        }
    } else if (xyz.dtype() == torch::kFloat64) {
        if (num_sh_coeff == 1) {
            precompute_rgb_from_sh_backward_kernel<double, 1><<<gridsize, blocksize>>>(
                xyz.data_ptr<double>(),
                camera_T_world.data_ptr<double>(),
                grad_rgb.data_ptr<double>(),
                N,
                grad_sh.data_ptr<double>()
            );
        } else if (num_sh_coeff == 4) {
            precompute_rgb_from_sh_backward_kernel<double, 4><<<gridsize, blocksize>>>(
                xyz.data_ptr<double>(),
                camera_T_world.data_ptr<double>(),
                grad_rgb.data_ptr<double>(),
                N,
                grad_sh.data_ptr<double>()
            );
        } else if (num_sh_coeff == 9) {
            precompute_rgb_from_sh_backward_kernel<double, 9><<<gridsize, blocksize>>>(
                xyz.data_ptr<double>(),
                camera_T_world.data_ptr<double>(),
                grad_rgb.data_ptr<double>(),
                N,
                grad_sh.data_ptr<double>()
            );
        } else if (num_sh_coeff == 16) {
            precompute_rgb_from_sh_backward_kernel<double, 16><<<gridsize, blocksize>>>(
                xyz.data_ptr<double>(),
                camera_T_world.data_ptr<double>(),
                grad_rgb.data_ptr<double>(),
                N,
                grad_sh.data_ptr<double>()
            );
        } else {
            AT_ERROR("Unsupported number of SH coefficients: ", num_sh_coeff);
        }
    } else {
        AT_ERROR("Unsupported data type: ", xyz.dtype());
    }
    cudaDeviceSynchronize();
}
