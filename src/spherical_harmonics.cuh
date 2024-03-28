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

template <typename T, unsigned int N_SH>
__device__ __inline__ void sh_to_rgb(
    const T* __restrict__ sh_coeff,
    const T* __restrict__ view_dir,
    T* __restrict__ rgb
){
    // Band 0
    # pragma unroll
    for (int i = 0; i < 3; i++){
        rgb[i] = T(SH_0) * sh_coeff[N_SH * i];
    }
    // Band 1
    if (N_SH >= 4){
        # pragma unroll
        for (int i = 0; i < 3; i++){
            rgb[i] += T(SH_1[0]) * view_dir[0] * sh_coeff[N_SH * i + 1]; // x
            rgb[i] += T(SH_1[1]) * view_dir[1] * sh_coeff[N_SH * i + 2]; // y
            rgb[i] += T(SH_1[2]) * view_dir[2] * sh_coeff[N_SH * i + 3]; // z
        }
    }
    if (N_SH >= 9) {
    // Band 2
        # pragma unroll
        for (int i = 0; i < 3; i++){
            rgb[i] += T(SH_2[0]) * view_dir[0] * view_dir[1] * sh_coeff[N_SH * i + 4]; // xy
            rgb[i] += T(SH_2[1]) * view_dir[1] * view_dir[2] * sh_coeff[N_SH * i + 5]; // yz
            rgb[i] += T(SH_2[2]) * (3 * view_dir[2] * view_dir[2] - 1.0) * sh_coeff[N_SH * i + 6]; // 3z^2 - 1
            rgb[i] += T(SH_2[3]) * view_dir[0] * view_dir[2] * sh_coeff[N_SH * i + 7]; // xz
            rgb[i] += T(SH_2[4]) * (view_dir[0] * view_dir[0] - view_dir[1] * view_dir[1]) * sh_coeff[N_SH * i + 8]; // x^2 - y^2
        }
    }
}

template <typename T, unsigned int N_SH>
__device__ __inline__ void compute_sh_grad(
    const T* __restrict__ grad_rgb,
    const T* __restrict__ view_dir,
    T* __restrict__ grad_sh
) {
    // Band 0 grad
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        grad_sh[N_SH * i] = T(SH_0) * grad_rgb[i];
    }
    // Band 1 grad
    if (N_SH >= 4){
        #pragma unroll
        for (int i = 0; i < 3; i++){
            grad_sh[N_SH * i + 1] = T(SH_1[0]) * view_dir[0] * grad_rgb[i]; // x
            grad_sh[N_SH * i + 2] = T(SH_1[1]) * view_dir[1] * grad_rgb[i]; // y
            grad_sh[N_SH * i + 3] = T(SH_1[2]) * view_dir[2] * grad_rgb[i]; // z
        }
    } 
    if (N_SH >= 9) {
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            grad_sh[N_SH * i + 4] = T(SH_2[0]) * view_dir[0] * view_dir[1] * grad_rgb[i]; // xy
            grad_sh[N_SH * i + 5] = T(SH_2[1]) * view_dir[1] * view_dir[2] * grad_rgb[i]; // yz
            grad_sh[N_SH * i + 6] = T(SH_2[2]) * (3 * view_dir[2] * view_dir[2] - 1.0) * grad_rgb[i]; // 3z^2 - 1
            grad_sh[N_SH * i + 7] = T(SH_2[3]) * view_dir[0] * view_dir[2] * grad_rgb[i]; // xz
            grad_sh[N_SH * i + 8] = T(SH_2[4]) * (view_dir[0] * view_dir[0] - view_dir[1] * view_dir[1]) * grad_rgb[i]; // x^2 - y^2
        }
    }
}
#endif // SPHERICAL_HARMONICS_CUH
