#pragma once
#include <cuda_bf16.h>

__global__ void convert_rgb_bfloat162(
    const __nv_bfloat16* __restrict__ rgb_bf16,
    const int N,
    const int N_SH,
    const int N_SH_PAIRS,
    __nv_bfloat162* __restrict__ rgb_bf162
);

__global__ void convert_rgb_grad_to_bfloat16(
    const __nv_bfloat162* __restrict__ rgb_grad_bf162,
    const int N,
    const int N_SH,
    const int N_SH_PAIRS,
    __nv_bfloat16* __restrict__ rgb_grad
);
