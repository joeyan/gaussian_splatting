#include "bfloat162_helpers.cuh"

__global__ void convert_rgb_bfloat162(
    const __nv_bfloat16* __restrict__ rgb_bf16,
    const int N,
    const int N_SH,
    const int N_SH_PAIRS,
    __nv_bfloat162* __restrict__ rgb_bf162
) {
    int splat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = threadIdx.y;
    int sh_pair_idx = threadIdx.z;

    // out of bounds
    if (splat_idx >= N || channel_idx >= 3 || sh_pair_idx * 2 >= N_SH) {
        return;
    }
    int rgb_in_idx = (splat_idx * 3 * N_SH) + (channel_idx * N_SH) + (sh_pair_idx * 2);
    int rgb_pair_idx = (splat_idx * 3 * N_SH_PAIRS) + (channel_idx * N_SH_PAIRS) + sh_pair_idx;

    // first value should always be valid
    rgb_bf162[rgb_pair_idx].x = rgb_bf16[rgb_in_idx];

    // second value may be invalid if N_SH is odd -> set to zero
    if ((sh_pair_idx * 2 + 1) < N_SH) {
        rgb_bf162[rgb_pair_idx].y = rgb_bf16[rgb_in_idx + 1];
    } else {
        rgb_bf162[rgb_pair_idx].y = __float2bfloat16(0.0f);
    }
}

__global__ void convert_rgb_grad_to_bfloat16(
    const __nv_bfloat162* __restrict__ rgb_grad_bf162,
    const int N,
    const int N_SH,
    const int N_SH_PAIRS,
    __nv_bfloat16* __restrict__ rgb_grad
) {
    int splat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int channel_idx = threadIdx.y;
    int sh_pair_idx = threadIdx.z;

    // out of bounds
    if (splat_idx >= N || channel_idx >= 3 || sh_pair_idx * 2 >= N_SH) {
        return;
    }
    int grad_out_idx = (splat_idx * 3 * N_SH) + (channel_idx * N_SH) + (sh_pair_idx * 2);
    int grad_pair_idx = (splat_idx * 3 * N_SH_PAIRS) + (channel_idx * N_SH_PAIRS) + sh_pair_idx;

    // first value should always be valid
    rgb_grad[grad_out_idx] = rgb_grad_bf162[grad_pair_idx].x;

    // second value may be invalid if N_SH is odd
    if ((sh_pair_idx * 2 + 1) < N_SH) {
        rgb_grad[grad_out_idx + 1] = rgb_grad_bf162[grad_pair_idx].y;
    }
}