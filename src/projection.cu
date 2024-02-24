#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "matrix.cuh"


template <typename T>
__global__ void camera_projection_kernel(
    const T* __restrict__ xyz,
    const T* __restrict__ K,
    const int N,
    T* uv
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    // u = fx * X / Z + cx
    uv[i * 2 + 0] = K[0] * xyz[i * 3 + 0] / xyz[i * 3 + 2] + K[2];
    // v = fy * Y / Z + cy
    uv[i * 2 + 1] = K[4] * xyz[i * 3 + 1] / xyz[i * 3 + 2] + K[5];
}

void camera_projection_cuda(
    torch::Tensor xyz,
    torch::Tensor K,
    torch::Tensor uv
) {
    TORCH_CHECK(xyz.is_cuda(), "xyz must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(uv.is_cuda(), "uv must be a CUDA tensor");

    TORCH_CHECK(xyz.is_contiguous(), "xyz must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(uv.is_contiguous(), "uv must be contiguous");

    const int N = xyz.size(0);
    TORCH_CHECK(xyz.size(1) == 3, "xyz must have shape Nx3");
    TORCH_CHECK(K.size(0) == 3, "K must have shape 3x3");
    TORCH_CHECK(K.size(1) == 3, "K must have shape 3x3");
    TORCH_CHECK(uv.size(0) == N, "uv must have shape Nx2");
    TORCH_CHECK(uv.size(1) == 2, "uv must have shape Nx2");

    const int max_threads_per_block = 1024;
    const int num_blocks = (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);


    // templated to allow use of float64 for gradcheck
    if (xyz.dtype() == torch::kFloat32) {
        camera_projection_kernel<float><<<gridsize, blocksize>>>(
            xyz.data_ptr<float>(),
            K.data_ptr<float>(),
            N,
            uv.data_ptr<float>()
        );
    } else if (xyz.dtype() == torch::kFloat64) {
        camera_projection_kernel<double><<<gridsize, blocksize>>>(
            xyz.data_ptr<double>(),
            K.data_ptr<double>(),
            N,
            uv.data_ptr<double>()
        );
    } else {
        TORCH_CHECK(false, "xyz must be float32 or float64");
    }
    cudaDeviceSynchronize();
}


template <typename T>
__global__ void compute_sigma_world_kernel(
    const T* __restrict__ quaternions,
    const T* __restrict__ scales,
    const int N,
    T* sigma_world
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    T qw = quaternions[i * 4 + 0];
    T qx = quaternions[i * 4 + 1];
    T qy = quaternions[i * 4 + 2];
    T qz = quaternions[i * 4 + 3];

    T norm = sqrt(qx * qx + qy * qy + qz * qz + qw * qw);

    // // zero magnitude quaternion is not valid
    qx /= norm;
    qy /= norm;
    qz /= norm;
    qw /= norm;

    T r00 = 1 - 2 * qy * qy - 2 * qz * qz;  
    T r01 = 2 * qx * qy - 2 * qz * qw;
    T r02 = 2 * qx * qz + 2 * qy * qw;
    T r10 = 2 * qx * qy + 2 * qz * qw;
    T r11 = 1 - 2 * qx * qx - 2 * qz * qz;
    T r12 = 2 * qy * qz - 2 * qx * qw;
    T r20 = 2 * qx * qz - 2 * qy * qw;
    T r21 = 2 * qy * qz + 2 * qx * qw;
    T r22 = 1 - 2 * qx * qx - 2 * qy * qy;

    T sx = exp(scales[i * 3 + 0]);
    T sy = exp(scales[i * 3 + 1]);
    T sz = exp(scales[i * 3 + 2]);

    T sx_sq = sx * sx;
    T sy_sq = sy * sy;
    T sz_sq = sz * sz;

    sigma_world[i * 9 + 0] = r00*r00*sx_sq + r01*r01*sy_sq + r02*r02*sz_sq;
    sigma_world[i * 9 + 1] = r00*r10*sx_sq + r01*r11*sy_sq + r02*r12*sz_sq;
    sigma_world[i * 9 + 2] = r00*r20*sx_sq + r01*r21*sy_sq + r02*r22*sz_sq;

    sigma_world[i * 9 + 3] = r00*r10*sx_sq + r01*r11*sy_sq + r02*r12*sz_sq;
    sigma_world[i * 9 + 4] = r10*r10*sx_sq + r11*r11*sy_sq + r12*r12*sz_sq;
    sigma_world[i * 9 + 5] = r10*r20*sx_sq + r11*r21*sy_sq + r12*r22*sz_sq;

    sigma_world[i * 9 + 6] = r00*r20*sx_sq + r01*r21*sy_sq + r02*r22*sz_sq;
    sigma_world[i * 9 + 7] = r10*r20*sx_sq + r11*r21*sy_sq + r12*r22*sz_sq;
    sigma_world[i * 9 + 8] = r20*r20*sx_sq + r21*r21*sy_sq + r22*r22*sz_sq;
}


void compute_sigma_world_cuda(
    torch::Tensor quaternions,
    torch::Tensor scales,
    torch::Tensor sigma_world
) {
    TORCH_CHECK(quaternions.is_cuda(), "quaternions must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(sigma_world.is_cuda(), "sigma_world must be a CUDA tensor");

    TORCH_CHECK(quaternions.is_contiguous(), "quaternions must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(sigma_world.is_contiguous(), "sigma_world must be contiguous");

    const int N = quaternions.size(0);
    TORCH_CHECK(quaternions.size(1) == 4, "quaternions must have shape Nx4");
    TORCH_CHECK(scales.size(0) == N, "scales must have shape Nx1");
    TORCH_CHECK(sigma_world.size(0) == N, "sigma_world must have shape Nx3x3");
    TORCH_CHECK(sigma_world.size(1) == 3, "sigma_world must have shape Nx3x3");
    TORCH_CHECK(sigma_world.size(2) == 3, "sigma_world must have shape Nx3x3");


    // can probably update this to improve perf
    const int max_threads_per_block = 1024;
    const int num_blocks = (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (quaternions.dtype() == torch::kFloat32) {
        compute_sigma_world_kernel<float><<<gridsize, blocksize>>>(
            quaternions.data_ptr<float>(),
            scales.data_ptr<float>(),
            N,
            sigma_world.data_ptr<float>()
        );
    } else if (quaternions.dtype() == torch::kFloat64) {
        compute_sigma_world_kernel<double><<<gridsize, blocksize>>>(
            quaternions.data_ptr<double>(),
            scales.data_ptr<double>(),
            N,
            sigma_world.data_ptr<double>()
        );
    } else {
        TORCH_CHECK(false, "quaternions must be float32 or float64");
    }
    cudaDeviceSynchronize();
}


template <typename T>
__global__ void compute_projection_jacobian_kernel(
    const T* __restrict__ xyz,
    const T* __restrict__ K,
    const int N,
    T* J
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    T x = xyz[i * 3 + 0];
    T y = xyz[i * 3 + 1];
    T z = xyz[i * 3 + 2];

    J[i * 6 + 0] = K[0] / z;
    J[i * 6 + 1] = 0;
    J[i * 6 + 2] = -K[0] * x / (z * z);
    J[i * 6 + 3] = 0;
    J[i * 6 + 4] = K[4] / z;
    J[i * 6 + 5] = -K[4] * y / (z * z);
}


void compute_projection_jacobian_cuda(
    torch::Tensor xyz,
    torch::Tensor K,
    torch::Tensor J
) {
    TORCH_CHECK(xyz.is_cuda(), "xyz must be a CUDA tensor");
    TORCH_CHECK(K.is_cuda(), "K must be a CUDA tensor");
    TORCH_CHECK(J.is_cuda(), "J must be a CUDA tensor");

    TORCH_CHECK(xyz.is_contiguous(), "xyz must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(J.is_contiguous(), "J must be contiguous");

    const int N = xyz.size(0);
    TORCH_CHECK(xyz.size(1) == 3, "xyz must have shape Nx3");
    TORCH_CHECK(K.size(0) == 3, "K must have shape 3x3");
    TORCH_CHECK(K.size(1) == 3, "K must have shape 3x3");
    TORCH_CHECK(J.size(0) == N, "J must have shape Nx2x3");
    TORCH_CHECK(J.size(1) == 2, "J must have shape Nx2x3");
    TORCH_CHECK(J.size(2) == 3, "J must have shape Nx2x3");

    const int max_threads_per_block = 1024;
    const int num_blocks = (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (xyz.dtype() == torch::kFloat32) {
        compute_projection_jacobian_kernel<float><<<gridsize, blocksize>>>(
            xyz.data_ptr<float>(),
            K.data_ptr<float>(),
            N,
            J.data_ptr<float>()
        );
    } else if (xyz.dtype() == torch::kFloat64) {
        compute_projection_jacobian_kernel<double><<<gridsize, blocksize>>>(
            xyz.data_ptr<double>(),
            K.data_ptr<double>(),
            N,
            J.data_ptr<double>()
        );
    } else {
        TORCH_CHECK(false, "xyz must be float32 or float64");
    }
}


template <typename T>
__global__ void compute_sigma_image_kernel(
    const T* __restrict__ sigma_world,
    const T* __restrict__ J,
    const T* __restrict__ world_T_image,
    const int N,
    T* sigma_image
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    // get rotation matrix
    T W[9];
    W[0] = world_T_image[0];
    W[1] = world_T_image[1];
    W[2] = world_T_image[2];
    W[3] = world_T_image[4];
    W[4] = world_T_image[5];
    W[5] = world_T_image[6];
    W[6] = world_T_image[8];
    W[7] = world_T_image[9];
    W[8] = world_T_image[10];

    // compute JW = J * W)
    T JW[6];
    matrix_multiply<T>(J + i * 6, W, JW, 2, 3, 3);

    // compute JWSigma = JW * sigma_world
    T JWSigma[6];
    matrix_multiply<T>(JW, sigma_world + i * 9, JWSigma, 2, 3, 3);

    T JW_t[6];
    transpose<T>(JW, JW_t, 2, 3);

    // compute sigma_image = JWSigma @ JW_t
    matrix_multiply<T>(JWSigma, JW_t, sigma_image + i * 4, 2, 3, 2);
}


void compute_sigma_image_cuda(
    torch::Tensor sigma_world,
    torch::Tensor J,
    torch::Tensor world_T_image,
    torch::Tensor sigma_image
) {
    TORCH_CHECK(sigma_world.is_cuda(), "sigma_world must be a CUDA tensor");
    TORCH_CHECK(J.is_cuda(), "J must be a CUDA tensor");
    TORCH_CHECK(world_T_image.is_cuda(), "world_T_image must be a CUDA tensor");
    TORCH_CHECK(sigma_image.is_cuda(), "sigma_image must be a CUDA tensor");

    TORCH_CHECK(sigma_world.is_contiguous(), "sigma_world must be contiguous");
    TORCH_CHECK(J.is_contiguous(), "J must be contiguous");
    TORCH_CHECK(world_T_image.is_contiguous(), "world_T_image must be contiguous");
    TORCH_CHECK(sigma_image.is_contiguous(), "sigma_image must be contiguous");

    const int N = sigma_world.size(0);
    TORCH_CHECK(sigma_world.size(1) == 3, "sigma_world must have shape Nx3x3");
    TORCH_CHECK(sigma_world.size(2) == 3, "sigma_world must have shape Nx3x3");
    TORCH_CHECK(J.size(0) == N, "J must have shape Nx2x3");
    TORCH_CHECK(J.size(1) == 2, "J must have shape Nx2x3");
    TORCH_CHECK(J.size(2) == 3, "J must have shape Nx2x3");
    TORCH_CHECK(world_T_image.size(0) == 4, "world_T_image must have shape 4x4");
    TORCH_CHECK(world_T_image.size(1) == 4, "world_T_image must have shape 4x4");
    TORCH_CHECK(sigma_image.size(0) == N, "sigma_image must have shape Nx2x2");
    TORCH_CHECK(sigma_image.size(1) == 2, "sigma_image must have shape Nx2x2");
    TORCH_CHECK(sigma_image.size(2) == 2, "sigma_image must have shape Nx2x2");

    const int max_threads_per_block = 1024;
    const int num_blocks = (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (sigma_world.dtype() == torch::kFloat32) {
        compute_sigma_image_kernel<float><<<gridsize, blocksize>>>(
            sigma_world.data_ptr<float>(),
            J.data_ptr<float>(),
            world_T_image.data_ptr<float>(),
            N,
            sigma_image.data_ptr<float>()
        );
    } else if (sigma_world.dtype() == torch::kFloat64) {
        compute_sigma_image_kernel<double><<<gridsize, blocksize>>>(
            sigma_world.data_ptr<double>(),
            J.data_ptr<double>(),
            world_T_image.data_ptr<double>(),
            N,
            sigma_image.data_ptr<double>()
        );
    } else {
        TORCH_CHECK(false, "sigma_world must be float32 or float64");
    }
}
