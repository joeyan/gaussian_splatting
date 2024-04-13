#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "checks.cuh"
#include "matrix.cuh"

template <typename T>
__global__ void camera_projection_backwards_kernel(
    const T* __restrict__ xyz,
    const T* __restrict__ K,
    const T* __restrict__ uv_grad_out,
    const int N,
    T* xyz_grad_in
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    if (xyz[i * 3 + 2] <= 0.0) {
        return;
    }
    // du/dx = fx / z
    T du_dx = K[0] / xyz[i * 3 + 2];
    // dv/dy = fy / z
    T dv_dy = K[4] / xyz[i * 3 + 2];
    // du/dz = -x* fx / (z * z)
    T du_dz = -K[0] * xyz[i * 3 + 0] / (xyz[i * 3 + 2] * xyz[i * 3 + 2]);
    // dv/dz = -y * fy / (z * z)
    T dv_dz = -K[4] * xyz[i * 3 + 1] / (xyz[i * 3 + 2] * xyz[i * 3 + 2]);

    xyz_grad_in[i * 3 + 0] = uv_grad_out[i * 2 + 0] * du_dx; // grad_x
    xyz_grad_in[i * 3 + 1] = uv_grad_out[i * 2 + 1] * dv_dy; // grad_y
    xyz_grad_in[i * 3 + 2] =
        uv_grad_out[i * 2 + 0] * du_dz + uv_grad_out[i * 2 + 1] * dv_dz; // grad_z
}

void camera_projection_backward_cuda(
    torch::Tensor xyz,
    torch::Tensor K,
    torch::Tensor uv_grad_out,
    torch::Tensor xyz_grad_in
) {
    CHECK_VALID_INPUT(xyz);
    CHECK_VALID_INPUT(K);
    CHECK_VALID_INPUT(uv_grad_out);
    CHECK_VALID_INPUT(xyz_grad_in);

    const int N = xyz.size(0);
    TORCH_CHECK(xyz.size(1) == 3, "xyz must be of shape Nx3");
    TORCH_CHECK(K.size(0) == 3 && K.size(1) == 3, "K must be of shape 3x3");
    TORCH_CHECK(
        uv_grad_out.size(0) == N && uv_grad_out.size(1) == 2, "uv_grad_out must be of shape Nx2"
    );
    TORCH_CHECK(
        xyz_grad_in.size(0) == N && xyz_grad_in.size(1) == 3, "xyz_grad_in must be of shape Nx3"
    );

    const int max_threads_per_block = 1024;
    const int num_blocks = (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (xyz.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(K);
        CHECK_FLOAT_TENSOR(uv_grad_out);
        CHECK_FLOAT_TENSOR(xyz_grad_in);
        camera_projection_backwards_kernel<float><<<gridsize, blocksize>>>(
            xyz.data_ptr<float>(),
            K.data_ptr<float>(),
            uv_grad_out.data_ptr<float>(),
            N,
            xyz_grad_in.data_ptr<float>()
        );
    } else if (xyz.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(K);
        CHECK_DOUBLE_TENSOR(uv_grad_out);
        CHECK_DOUBLE_TENSOR(xyz_grad_in);
        camera_projection_backwards_kernel<double><<<gridsize, blocksize>>>(
            xyz.data_ptr<double>(),
            K.data_ptr<double>(),
            uv_grad_out.data_ptr<double>(),
            N,
            xyz_grad_in.data_ptr<double>()
        );
    } else {
        AT_ERROR("Inputs must be float32 or float64");
    }
    cudaDeviceSynchronize();
}

template <typename T>
__global__ void compute_projection_jacobian_backward_kernel(
    const T* __restrict__ xyz,
    const T* __restrict__ K,
    const T* __restrict__ jac_grad_out,
    const int N,
    T* xyz_grad_in
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    T fx = K[0];
    T fy = K[4];

    // dJ/dx = dz/du * -fx/z^2
    xyz_grad_in[i * 3 + 0] = jac_grad_out[i * 6 + 2] * -fx / (xyz[i * 3 + 2] * xyz[i * 3 + 2]);
    // dJ/dy = dz/dv * -fy/z^2
    xyz_grad_in[i * 3 + 1] = jac_grad_out[i * 6 + 5] * -fy / (xyz[i * 3 + 2] * xyz[i * 3 + 2]);
    // dJ/dz = dx/du * -fx/z^2 + dy/dv * -fy/z^2 + dz/du * 2x * fx / z^3 + dz/dv
    // * 2y * fy / z^3
    xyz_grad_in[i * 3 + 2] = jac_grad_out[i * 6 + 0] * -fx / (xyz[i * 3 + 2] * xyz[i * 3 + 2]) +
                             jac_grad_out[i * 6 + 4] * -fy / (xyz[i * 3 + 2] * xyz[i * 3 + 2]) +
                             jac_grad_out[i * 6 + 2] * 2 * xyz[i * 3 + 0] * fx /
                                 (xyz[i * 3 + 2] * xyz[i * 3 + 2] * xyz[i * 3 + 2]) +
                             jac_grad_out[i * 6 + 5] * 2 * xyz[i * 3 + 1] * fy /
                                 (xyz[i * 3 + 2] * xyz[i * 3 + 2] * xyz[i * 3 + 2]);
}

void compute_projection_jacobian_backward_cuda(
    torch::Tensor xyz,
    torch::Tensor K,
    torch::Tensor jac_grad_out,
    torch::Tensor xyz_grad_in
) {
    CHECK_VALID_INPUT(xyz);
    CHECK_VALID_INPUT(K);
    CHECK_VALID_INPUT(jac_grad_out);
    CHECK_VALID_INPUT(xyz_grad_in);

    const int N = xyz.size(0);

    const int max_threads_per_block = 1024;
    const int num_blocks = (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (xyz.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(K);
        CHECK_FLOAT_TENSOR(jac_grad_out);
        CHECK_FLOAT_TENSOR(xyz_grad_in);
        compute_projection_jacobian_backward_kernel<float><<<gridsize, blocksize>>>(
            xyz.data_ptr<float>(),
            K.data_ptr<float>(),
            jac_grad_out.data_ptr<float>(),
            N,
            xyz_grad_in.data_ptr<float>()
        );
    } else if (xyz.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(K);
        CHECK_DOUBLE_TENSOR(jac_grad_out);
        CHECK_DOUBLE_TENSOR(xyz_grad_in);
        compute_projection_jacobian_backward_kernel<double><<<gridsize, blocksize>>>(
            xyz.data_ptr<double>(),
            K.data_ptr<double>(),
            jac_grad_out.data_ptr<double>(),
            N,
            xyz_grad_in.data_ptr<double>()
        );
    } else {
        AT_ERROR("Inputs must be float32 or float64");
    }
    cudaDeviceSynchronize();
}

// __launch_bounds__ limits max threads per block
// This is necesary for float64 used in gradcheck
// See more:
// https://github.com/pytorch/pytorch/issues/7680#issuecomment-390729076
// https://discuss.pytorch.org/t/too-many-resources-requested-for-launch-when-use-gradcheck/9761
template <typename T>
__launch_bounds__(1024) __global__ void compute_sigma_world_backward_kernel(
    const T* __restrict__ quaternion,
    const T* __restrict__ scale,
    const T* __restrict__ sigma_world_grad_out,
    const int N,
    T* quaternion_grad_in,
    T* scale_grad_in
) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }
    // compute scale
    T S[9];
    S[0] = exp(scale[i * 3 + 0]);
    S[1] = 0;
    S[2] = 0;
    S[3] = 0;
    S[4] = exp(scale[i * 3 + 1]);
    S[5] = 0;
    S[6] = 0;
    S[7] = 0;
    S[8] = exp(scale[i * 3 + 2]);

    T qw = quaternion[i * 4 + 0];
    T qx = quaternion[i * 4 + 1];
    T qy = quaternion[i * 4 + 2];
    T qz = quaternion[i * 4 + 3];

    T norm_q = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    T qw_norm = qw / norm_q;
    T qx_norm = qx / norm_q;
    T qy_norm = qy / norm_q;
    T qz_norm = qz / norm_q;

    T R[9];
    R[0] = 1.0 - 2.0 * qy_norm * qy_norm - 2.0 * qz_norm * qz_norm;
    R[1] = 2.0 * qx_norm * qy_norm - 2.0 * qz_norm * qw_norm;
    R[2] = 2.0 * qx_norm * qz_norm + 2 * qy_norm * qw_norm;
    R[3] = 2.0 * qx_norm * qy_norm + 2 * qz_norm * qw_norm;
    R[4] = 1.0 - 2.0 * qx_norm * qx_norm - 2.0 * qz_norm * qz_norm;
    R[5] = 2.0 * qy_norm * qz_norm - 2.0 * qx_norm * qw_norm;
    R[6] = 2.0 * qx_norm * qz_norm - 2.0 * qy_norm * qw_norm;
    R[7] = 2.0 * qy_norm * qz_norm + 2.0 * qx_norm * qw_norm;
    R[8] = 1.0 - 2.0 * qx_norm * qx_norm - 2.0 * qy_norm * qy_norm;

    T RS[9];
    matrix_multiply(R, S, RS, 3, 3, 3);

    T gradRS[9];
    matrix_multiply(sigma_world_grad_out + i * 9, RS, gradRS, 3, 3, 3);

    T RS_t[9];
    transpose(RS, RS_t, 3, 3);

    T gradSR[9];
    matrix_multiply(RS_t, sigma_world_grad_out + i * 9, gradSR, 3, 3, 3);

    // first half of gradR
    T gradR[9];
    matrix_multiply(gradRS, S, gradR, 3, 3, 3);

    // second half of gradR
    T SgradSR[9];
    matrix_multiply(S, gradSR, SgradSR, 3, 3, 3);

    // add the transpose of SgradSR
    gradR[0] += SgradSR[0];
    gradR[1] += SgradSR[3];
    gradR[2] += SgradSR[6];
    gradR[3] += SgradSR[1];
    gradR[4] += SgradSR[4];
    gradR[5] += SgradSR[7];
    gradR[6] += SgradSR[2];
    gradR[7] += SgradSR[5];
    gradR[8] += SgradSR[8];

    T R_t[9];
    transpose(R, R_t, 3, 3);

    // first half of grad_S
    T grad_S[9];
    matrix_multiply(R_t, gradRS, grad_S, 3, 3, 3);

    // second half of grad_S
    T gradSRR[9];
    matrix_multiply(gradSR, R, gradSRR, 3, 3, 3);

    // add the transpose of gradSRR
    grad_S[0] += gradSRR[0];
    grad_S[1] += gradSRR[3];
    grad_S[2] += gradSRR[6];
    grad_S[3] += gradSRR[1];
    grad_S[4] += gradSRR[4];
    grad_S[5] += gradSRR[7];
    grad_S[6] += gradSRR[2];
    grad_S[7] += gradSRR[5];
    grad_S[8] += gradSRR[8];

    // write out scale gradients
    scale_grad_in[i * 3 + 0] = grad_S[0] * exp(scale[i * 3 + 0]);
    scale_grad_in[i * 3 + 1] = grad_S[4] * exp(scale[i * 3 + 1]);
    scale_grad_in[i * 3 + 2] = grad_S[8] * exp(scale[i * 3 + 2]);

    // compute gradient for normalized quaternion
    T grad_q_norm[4];
    grad_q_norm[0] = -2.0 * qz_norm * gradR[1] + 2.0 * qy_norm * gradR[2] +
                     2.0 * qz_norm * gradR[3] - 2.0 * qx_norm * gradR[5] -
                     2.0 * qy_norm * gradR[6] + 2.0 * qx_norm * gradR[7];
    grad_q_norm[1] = 2.0 * qy_norm * gradR[1] + 2.0 * qz_norm * gradR[2] +
                     2.0 * qy_norm * gradR[3] - 4.0 * qx_norm * gradR[4] -
                     2.0 * qw_norm * gradR[5] + 2.0 * qz_norm * gradR[6] +
                     2.0 * qw_norm * gradR[7] - 4.0 * qx_norm * gradR[8];
    grad_q_norm[2] = -4.0 * qy_norm * gradR[0] + 2.0 * qx_norm * gradR[1] +
                     2.0 * qw_norm * gradR[2] + 2.0 * qx_norm * gradR[3] +
                     2.0 * qz_norm * gradR[5] - 2.0 * qw_norm * gradR[6] +
                     2.0 * qz_norm * gradR[7] - 4.0 * qy_norm * gradR[8];
    grad_q_norm[3] = -4.0 * qz_norm * gradR[0] - 2.0 * qw_norm * gradR[1] +
                     2.0 * qx_norm * gradR[2] + 2.0 * qw_norm * gradR[3] -
                     4.0 * qz_norm * gradR[4] + 2.0 * qy_norm * gradR[5] +
                     2.0 * qx_norm * gradR[6] + 2.0 * qy_norm * gradR[7];

    // apply gradient for quaternion normalization
    T q_norm_cubed = norm_q * norm_q * norm_q; // (x^2 + y^2 + z^2 + w^2)^(3/2)

    quaternion_grad_in[i * 4 + 0] = (1.0 / norm_q - qw * qw / q_norm_cubed) * grad_q_norm[0] -
                                    qw * qx / q_norm_cubed * grad_q_norm[1] -
                                    qw * qy / q_norm_cubed * grad_q_norm[2] -
                                    qw * qz / q_norm_cubed * grad_q_norm[3];
    quaternion_grad_in[i * 4 + 1] = -qw * qx / q_norm_cubed * grad_q_norm[0] +
                                    (1.0 / norm_q - qx * qx / q_norm_cubed) * grad_q_norm[1] -
                                    qx * qy / q_norm_cubed * grad_q_norm[2] -
                                    qx * qz / q_norm_cubed * grad_q_norm[3];
    quaternion_grad_in[i * 4 + 2] = -qw * qy / q_norm_cubed * grad_q_norm[0] -
                                    qx * qy / q_norm_cubed * grad_q_norm[1] +
                                    (1.0 / norm_q - qy * qy / q_norm_cubed) * grad_q_norm[2] -
                                    qy * qz / q_norm_cubed * grad_q_norm[3];
    quaternion_grad_in[i * 4 + 3] = -qw * qz / q_norm_cubed * grad_q_norm[0] -
                                    qx * qz / q_norm_cubed * grad_q_norm[1] -
                                    qy * qz / q_norm_cubed * grad_q_norm[2] +
                                    (1.0 / norm_q - qz * qz / q_norm_cubed) * grad_q_norm[3];
}

void compute_sigma_world_backward_cuda(
    torch::Tensor quaternion,
    torch::Tensor scale,
    torch::Tensor sigma_world_grad_out,
    torch::Tensor quaternion_grad_in,
    torch::Tensor scale_grad_in
) {
    CHECK_VALID_INPUT(quaternion);
    CHECK_VALID_INPUT(scale);
    CHECK_VALID_INPUT(sigma_world_grad_out);
    CHECK_VALID_INPUT(quaternion_grad_in);
    CHECK_VALID_INPUT(scale_grad_in);

    const int N = quaternion.size(0);
    TORCH_CHECK(quaternion.size(1) == 4, "quaternion must be of shape Nx4");
    TORCH_CHECK(scale.size(0) == N && scale.size(1) == 3, "scale must be of shape Nx3");
    TORCH_CHECK(
        sigma_world_grad_out.size(0) == N && sigma_world_grad_out.size(1) == 3 &&
            sigma_world_grad_out.size(2) == 3,
        "sigma_world_grad_out must be of shape Nx3x3"
    );
    TORCH_CHECK(
        quaternion_grad_in.size(0) == N && quaternion_grad_in.size(1) == 4,
        "quaternion_grad_in must be of shape Nx4"
    );
    TORCH_CHECK(
        scale_grad_in.size(0) == N && scale_grad_in.size(1) == 3,
        "scale_grad_in must be of shape Nx3"
    );

    const int max_threads_per_block = 1024;
    const int num_blocks = (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (quaternion.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(scale);
        CHECK_FLOAT_TENSOR(sigma_world_grad_out);
        CHECK_FLOAT_TENSOR(quaternion_grad_in);
        CHECK_FLOAT_TENSOR(scale_grad_in);
        compute_sigma_world_backward_kernel<float><<<gridsize, blocksize>>>(
            quaternion.data_ptr<float>(),
            scale.data_ptr<float>(),
            sigma_world_grad_out.data_ptr<float>(),
            N,
            quaternion_grad_in.data_ptr<float>(),
            scale_grad_in.data_ptr<float>()
        );
    } else if (quaternion.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(scale);
        CHECK_DOUBLE_TENSOR(sigma_world_grad_out);
        CHECK_DOUBLE_TENSOR(quaternion_grad_in);
        CHECK_DOUBLE_TENSOR(scale_grad_in);
        compute_sigma_world_backward_kernel<double><<<gridsize, blocksize>>>(
            quaternion.data_ptr<double>(),
            scale.data_ptr<double>(),
            sigma_world_grad_out.data_ptr<double>(),
            N,
            quaternion_grad_in.data_ptr<double>(),
            scale_grad_in.data_ptr<double>()
        );
    } else {
        AT_ERROR("Inputs must be float32 or float64");
    }
    cudaDeviceSynchronize();
}

template <typename T>
__launch_bounds__(1024) __global__ void compute_conic_backward_kernel(
    const T* __restrict__ sigma_world,
    const T* __restrict__ J,
    const T* __restrict__ world_T_image,
    const T* __restrict__ conic_grad_out,
    const int N,
    T* sigma_world_grad_in,
    T* J_grad_in
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

    // compute JW = J @ W)
    T JW[6]; // 2x3
    matrix_multiply<T>(J + i * 6, W, JW, 2, 3, 3);

    T JW_t[6]; // 3x2
    transpose<T>(JW, JW_t, 2, 3);

    // convert conic to sigma image
    T sigma_image_grad_out[4];
    sigma_image_grad_out[0] = conic_grad_out[i * 3 + 0];
    sigma_image_grad_out[1] = conic_grad_out[i * 3 + 1];
    sigma_image_grad_out[2] = conic_grad_out[i * 3 + 1];
    sigma_image_grad_out[3] = conic_grad_out[i * 3 + 2];

    T JW_t_grad_sigma_image[6]; // 3x2
    matrix_multiply<T>(JW_t, sigma_image_grad_out, JW_t_grad_sigma_image, 3, 2, 2);

    // Write sigma_world_grad to output
    // compute sigma_world_grad_in (3x3) = JW_t_grad_sigma_image (3x2) @ JW
    // (2x3)
    matrix_multiply<T>(JW_t_grad_sigma_image, JW, sigma_world_grad_in + i * 9, 3, 2, 3);

    T sigma_world_JW_t[6]; // (3x2) = sigma_world (3x3) @ JW_t (3x2)
    matrix_multiply<T>(sigma_world + i * 9, JW_t, sigma_world_JW_t, 3, 3, 2);

    T grad_sigma_image_t[4]; // 2x2
    transpose<T>(sigma_image_grad_out, grad_sigma_image_t, 2, 2);

    T grad_JW_t_left[6]; // (3x2) = sigma_world_JW_t (3x2) @ grad_sigma_image_t
                         // (2x2)
    matrix_multiply<T>(sigma_world_JW_t, grad_sigma_image_t, grad_JW_t_left, 3, 2, 2);

    T sigma_world_t[9]; // 3x3
    transpose<T>(sigma_world + i * 9, sigma_world_t, 3, 3);

    T sigma_world_tJW_t[6]; // (3x2) = sigma_world_t (3x3) @ JW_t (3x2)
    matrix_multiply<T>(sigma_world_t, JW_t, sigma_world_tJW_t, 3, 3, 2);

    T grad_JW_t_right[6]; // (3x2) = sigma_world_tJW_t (3x2) @ grad_sigma_image
                          // (2x2)
    matrix_multiply<T>(sigma_world_tJW_t, sigma_image_grad_out, grad_JW_t_right, 3, 2, 2);

    // add grad_JW_t_left and grad_JW_t_right
    T grad_JW_t[6]; // 3x2
    grad_JW_t[0] = grad_JW_t_left[0] + grad_JW_t_right[0];
    grad_JW_t[1] = grad_JW_t_left[1] + grad_JW_t_right[1];
    grad_JW_t[2] = grad_JW_t_left[2] + grad_JW_t_right[2];
    grad_JW_t[3] = grad_JW_t_left[3] + grad_JW_t_right[3];
    grad_JW_t[4] = grad_JW_t_left[4] + grad_JW_t_right[4];
    grad_JW_t[5] = grad_JW_t_left[5] + grad_JW_t_right[5];

    // viewing transform is not optimized so the gradient is not needed
    // leaving this here in case we need it in the future
    // T grad_W_t[9]; // 3x3 = grad_JW_t (3x2) @ J (2x3)
    // matrix_multiply<T>(grad_JW_t, J + i * 6, grad_W_t, 3, 2, 3);

    T grad_J_t[6]; // 3x2 = W (3x3) @ grad_JW_t (3x2)
    matrix_multiply<T>(W, grad_JW_t, grad_J_t, 3, 3, 2);

    // write out grad_J
    transpose<T>(grad_J_t, J_grad_in + i * 6, 3, 2);
}

void compute_conic_backward_cuda(
    torch::Tensor sigma_world,
    torch::Tensor J,
    torch::Tensor world_T_image,
    torch::Tensor conic_grad_out,
    torch::Tensor sigma_world_grad_in,
    torch::Tensor J_grad_in
) {
    CHECK_VALID_INPUT(sigma_world);
    CHECK_VALID_INPUT(J);
    CHECK_VALID_INPUT(world_T_image);
    CHECK_VALID_INPUT(conic_grad_out);
    CHECK_VALID_INPUT(sigma_world_grad_in);
    CHECK_VALID_INPUT(J_grad_in);

    const int N = sigma_world.size(0);
    TORCH_CHECK(
        sigma_world.size(1) == 3 && sigma_world.size(2) == 3, "sigma_world must be of shape Nx3x3"
    );
    TORCH_CHECK(J.size(0) == N && J.size(1) == 2 && J.size(2) == 3, "J must be of shape Nx2x3");
    TORCH_CHECK(
        world_T_image.size(0) == 4 && world_T_image.size(1) == 4,
        "world_T_image must be of shape 4x4"
    );
    TORCH_CHECK(
        conic_grad_out.size(0) == N && conic_grad_out.size(1) == 3,
        "conic_grad_out must be of shape Nx3"
    );
    TORCH_CHECK(
        sigma_world_grad_in.size(0) == N && sigma_world_grad_in.size(1) == 3 &&
            sigma_world_grad_in.size(2) == 3,
        "sigma_world_grad_in must be of shape Nx3x3"
    );
    TORCH_CHECK(
        J_grad_in.size(0) == N && J_grad_in.size(1) == 2 && J_grad_in.size(2) == 3,
        "J_grad_in must be of shape Nx2x3"
    );

    const int max_threads_per_block = 1024;
    const int num_blocks = (N + max_threads_per_block - 1) / max_threads_per_block;
    dim3 gridsize(num_blocks, 1, 1);
    dim3 blocksize(max_threads_per_block, 1, 1);

    if (sigma_world.dtype() == torch::kFloat32) {
        CHECK_FLOAT_TENSOR(J);
        CHECK_FLOAT_TENSOR(world_T_image);
        CHECK_FLOAT_TENSOR(conic_grad_out);
        CHECK_FLOAT_TENSOR(sigma_world_grad_in);
        CHECK_FLOAT_TENSOR(J_grad_in);
        compute_conic_backward_kernel<float><<<gridsize, blocksize>>>(
            sigma_world.data_ptr<float>(),
            J.data_ptr<float>(),
            world_T_image.data_ptr<float>(),
            conic_grad_out.data_ptr<float>(),
            N,
            sigma_world_grad_in.data_ptr<float>(),
            J_grad_in.data_ptr<float>()
        );
    } else if (sigma_world.dtype() == torch::kFloat64) {
        CHECK_DOUBLE_TENSOR(J);
        CHECK_DOUBLE_TENSOR(world_T_image);
        CHECK_DOUBLE_TENSOR(conic_grad_out);
        CHECK_DOUBLE_TENSOR(sigma_world_grad_in);
        CHECK_DOUBLE_TENSOR(J_grad_in);
        compute_conic_backward_kernel<double><<<gridsize, blocksize>>>(
            sigma_world.data_ptr<double>(),
            J.data_ptr<double>(),
            world_T_image.data_ptr<double>(),
            conic_grad_out.data_ptr<double>(),
            N,
            sigma_world_grad_in.data_ptr<double>(),
            J_grad_in.data_ptr<double>()
        );
    } else {
        AT_ERROR("Inputs must be float32 or float64");
    }
    cudaDeviceSynchronize();
}
