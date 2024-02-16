#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void camera_projection_kernel(
    const float* xyz,
    const float* K,
    const int N,
    float* uv
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

    camera_projection_kernel<<<gridsize, blocksize>>>(
        xyz.data_ptr<float>(),
        K.data_ptr<float>(),
        N,
        uv.data_ptr<float>()
    );
    cudaDeviceSynchronize();
}
