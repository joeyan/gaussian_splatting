#pragma once
#ifndef CHECKS_H
#define CHECKS_H

#define CHECK_IS_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " is not a CUDA tensor")
#define CHECK_IS_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " is not a contiguous tensor")
#define CHECK_VALID_INPUT(x)                                                                       \
    CHECK_IS_CUDA(x);                                                                              \
    CHECK_IS_CONTIGUOUS(x)

#define CHECK_HALF_TENSOR(x) TORCH_CHECK(x.dtype() == torch::kFloat16, #x " is not a half tensor")
#define CHECK_BFLOAT16_TENSOR(x)                                                                   \
    TORCH_CHECK(x.dtype() == torch::kBFloat16, #x " is not a bfloat16 tensor")
#define CHECK_FLOAT_TENSOR(x) TORCH_CHECK(x.dtype() == torch::kFloat32, #x " is not a float tensor")
#define CHECK_DOUBLE_TENSOR(x)                                                                     \
    TORCH_CHECK(x.dtype() == torch::kFloat64, #x " is not a double tensor")
#define CHECK_INT_TENSOR(x) TORCH_CHECK(x.dtype() == torch::kInt32, #x " is not an int tensor")

#endif
