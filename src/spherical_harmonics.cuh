#ifndef SPHERICAL_HARMONICS_CUH
#define SPHERICAL_HARMONICS_CUH

__device__ __constant__ const float SH_0 = 0.28209479177387814;
// repeat same value to make sign management easier during SH calculation
__device__ __constant__ const float SH_1[3] = {
    -0.4886025119029199,
    0.4886025119029199,
    -0.4886025119029199
};
__device__ __constant__ const float SH_2[5] = {
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
};
__device__ __constant__ const float SH_3[7] = {
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.263875515352797,
    0.4570457994644658,
    1.445305721320277,
    0.5900435899266435
};

template <typename T, unsigned int N_SH>
__device__ __inline__ void sh_to_rgb(
    const T* __restrict__ sh_coeff,
    const T* __restrict__ view_dir,
    T* __restrict__ rgb
){
    T x = view_dir[0];
    T y = view_dir[1];
    T z = view_dir[2];

    // Band 0
    #pragma unroll
    for (int i = 0; i < 3; i++){
        rgb[i] = T(SH_0) * sh_coeff[N_SH * i];
    }
    // Band 1
    if (N_SH >= 4){
        #pragma unroll
        for (int i = 0; i < 3; i++){
            rgb[i] += T(SH_1[0]) * x * sh_coeff[N_SH * i + 1]; // x
            rgb[i] += T(SH_1[1]) * y * sh_coeff[N_SH * i + 2]; // y
            rgb[i] += T(SH_1[2]) * z * sh_coeff[N_SH * i + 3]; // z
        }
    }
    // Band 2
    if (N_SH >= 9) {
        #pragma unroll
        for (int i = 0; i < 3; i++){
            rgb[i] += T(SH_2[0]) * x * y * sh_coeff[N_SH * i + 4]; // xy
            rgb[i] += T(SH_2[1]) * y * z * sh_coeff[N_SH * i + 5]; // yz
            rgb[i] += T(SH_2[2]) * (3 * z * z - 1.0) * sh_coeff[N_SH * i + 6]; // 3z^2 - 1
            rgb[i] += T(SH_2[3]) * x * z * sh_coeff[N_SH * i + 7]; // xz
            rgb[i] += T(SH_2[4]) * (x * x - y * y) * sh_coeff[N_SH * i + 8]; // x^2 - y^2
        }
    }
    // Band 3
    if (N_SH >= 16) {
        #pragma unroll
        for (int i = 0; i <3; i++) {
            rgb[i] += T(SH_3[0]) * y * (3 * x * x - y * y) * sh_coeff[N_SH * i + 9]; // y * (3x^2 - y^2)
            rgb[i] += T(SH_3[1]) * x * y * z * sh_coeff[N_SH * i + 10]; // xyz
            rgb[i] += T(SH_3[2]) * y * (5 * z * z - 1.0) * sh_coeff[N_SH * i + 11]; // y(5z^2 - 1)
            rgb[i] += T(SH_3[3]) * z * (5 * z * z - 3.0) * sh_coeff[N_SH * i + 12]; // z(5z^2 - 3)
            rgb[i] += T(SH_3[4]) * x * (5 * z * z - 1.0) * sh_coeff[N_SH * i + 13]; // x(5z^2 - 1)
            rgb[i] += T(SH_3[5]) * z * (x * x - y * y) * sh_coeff[N_SH * i + 14]; // z(x^2 - y^2)
            rgb[i] += T(SH_3[6]) * x * (x * x - 3 * y * y) * sh_coeff[N_SH * i + 15]; // x(x^2 - 3y^2)
        }
    }
}

template <typename T, unsigned int N_SH>
__device__ __inline__ void compute_sh_grad(
    const T* __restrict__ grad_rgb,
    const T* __restrict__ view_dir,
    T* __restrict__ grad_sh
) {
    T x = view_dir[0];
    T y = view_dir[1];
    T z = view_dir[2];

    // Band 0 grad
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        grad_sh[N_SH * i] = T(SH_0) * grad_rgb[i];
    }
    // Band 1 grad
    if (N_SH >= 4){
        #pragma unroll
        for (int i = 0; i < 3; i++){
            grad_sh[N_SH * i + 1] = T(SH_1[0]) * x * grad_rgb[i]; // x
            grad_sh[N_SH * i + 2] = T(SH_1[1]) * y * grad_rgb[i]; // y
            grad_sh[N_SH * i + 3] = T(SH_1[2]) * z * grad_rgb[i]; // z
        }
    } 
    if (N_SH >= 9) {
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            grad_sh[N_SH * i + 4] = T(SH_2[0]) * x * y * grad_rgb[i]; // xy
            grad_sh[N_SH * i + 5] = T(SH_2[1]) * y * z * grad_rgb[i]; // yz
            grad_sh[N_SH * i + 6] = T(SH_2[2]) * (3 * z * z - 1.0) * grad_rgb[i]; // 3z^2 - 1
            grad_sh[N_SH * i + 7] = T(SH_2[3]) * x * z * grad_rgb[i]; // xz
            grad_sh[N_SH * i + 8] = T(SH_2[4]) * (x * x - y * y) * grad_rgb[i]; // x^2 - y^2
        }
    }
    if (N_SH >= 16) {
        #pragma unroll
        for (int i = 0; i <3; i++) {
            grad_sh[N_SH * i + 9]= T(SH_3[0]) * y * (3 * x * x - y * y) * grad_rgb[i]; // y * (3x^2 - y^2)
            grad_sh[N_SH * i + 10]= T(SH_3[1]) * x * y * z * grad_rgb[i]; // xyz
            grad_sh[N_SH * i + 11]= T(SH_3[2]) * y * (5 * z * z - 1.0) * grad_rgb[i]; // y(5z^2 - 1)
            grad_sh[N_SH * i + 12]= T(SH_3[3]) * z * (5 * z * z - 3.0) * grad_rgb[i]; // z(5z^2 - 3)
            grad_sh[N_SH * i + 13]= T(SH_3[4]) * x * (5 * z * z - 1.0) * grad_rgb[i]; // x(5z^2 - 1)
            grad_sh[N_SH * i + 14]= T(SH_3[5]) * z * (x * x - y * y) * grad_rgb[i]; // z(x^2 - y^2)
            grad_sh[N_SH * i + 15]= T(SH_3[6]) * x * (x * x - 3 * y * y) * grad_rgb[i]; // x(x^2 - 3y^2)
        }
    }
}
#endif // SPHERICAL_HARMONICS_CUH
