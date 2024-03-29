#ifndef SPHERICAL_HARMONICS_CUH
#define SPHERICAL_HARMONICS_CUH

__device__ __constant__ const float SH_0 = 0.28209479177387814;
// repeat same value to make sign management easier during SH calculation
__device__ __constant__ const float SH_1[3] = {-0.4886025119029199,
                                               0.4886025119029199,
                                               -0.4886025119029199};
__device__ __constant__ const float SH_2[5] = {1.0925484305920792,
                                               -1.0925484305920792,
                                               0.31539156525252005,
                                               -1.0925484305920792,
                                               0.5462742152960396};
__device__ __constant__ const float SH_3[7] = {-0.5900435899266435,
                                               2.890611442640554,
                                               -0.4570457994644658,
                                               0.263875515352797,
                                               -0.4570457994644658,
                                               1.445305721320277,
                                               -0.5900435899266435};

template <typename T, unsigned int N_SH>
__device__ __inline__ void compute_sh_coeffs_for_view_dir(
    const T* __restrict__ view_dir,
    T* __restrict__ sh_at_view_dir) {
  // Band 0
  sh_at_view_dir[0] = T(SH_0);

  if (N_SH < 4) return;

  const T x = view_dir[0];
  const T y = view_dir[1];
  const T z = view_dir[2];

  // Band 1
  sh_at_view_dir[1] = T(SH_1[0]) * x;
  sh_at_view_dir[2] = T(SH_1[1]) * y;
  sh_at_view_dir[3] = T(SH_1[2]) * z;

  if (N_SH < 9) return;

  const T xy = x * y;
  const T yz = y * z;
  const T xz = x * z;
  const T xx = x * x;
  const T yy = y * y;
  const T zz = z * z;

  // Band 2
  sh_at_view_dir[4] = T(SH_2[0]) * xy;              // xy
  sh_at_view_dir[5] = T(SH_2[1]) * yz;              // yz
  sh_at_view_dir[6] = T(SH_2[2]) * (3 * zz - 1.0);  // 3z^2 - 1
  sh_at_view_dir[7] = T(SH_2[3]) * xz;              // xz
  sh_at_view_dir[8] = T(SH_2[4]) * (xx - yy);       // x^2 - y^2

  if (N_SH < 16) return;

  // Band 3
  sh_at_view_dir[9] = T(SH_3[0]) * y * (3 * xx - yy);    // y * (3x^2 - y^2)
  sh_at_view_dir[10] = T(SH_3[1]) * xy * z;              // xyz
  sh_at_view_dir[11] = T(SH_3[2]) * y * (5 * zz - 1.0);  // y(5z^2 - 1)
  sh_at_view_dir[12] = T(SH_3[3]) * z * (5 * zz - 3.0);  // z(5z^2 - 3)
  sh_at_view_dir[13] = T(SH_3[4]) * x * (5 * zz - 1.0);  // x(5z^2 - 1)
  sh_at_view_dir[14] = T(SH_3[5]) * z * (xx - yy);       // z(x^2 - y^2)
  sh_at_view_dir[15] = T(SH_3[6]) * x * (xx - 3 * yy);   // x(x^2 - 3y^2)
}

template <typename T, unsigned int N_SH>
__device__ __inline__ void sh_to_rgb(const T* __restrict__ sh_coeff,
                                     const T* __restrict__ sh_at_view_dir,
                                     T* __restrict__ rgb) {
// set rgb to zero order value
#pragma unroll
  for (int channel = 0; channel < 3; channel++) {
    rgb[channel] = sh_at_view_dir[0] * sh_coeff[N_SH * channel];
  }

  // add higher order values if needed
  if (N_SH < 4) return;
#pragma unroll
  for (int sh = 1; sh < N_SH; sh++) {
#pragma unroll
    for (int channel = 0; channel < 3; channel++) {
      rgb[channel] += sh_at_view_dir[sh] * sh_coeff[N_SH * channel + sh];
    }
  }
}

template <typename T, unsigned int N_SH>
__device__ __inline__ void compute_sh_grad(const T* __restrict__ grad_rgb,
                                           const T* __restrict__ sh_at_view_dir,
                                           T* __restrict__ grad_sh) {
#pragma unroll
  for (int sh = 0; sh < N_SH; sh++) {
#pragma unroll
    for (int channel = 0; channel < 3; channel++) {
      grad_sh[N_SH * channel + sh] = sh_at_view_dir[sh] * grad_rgb[channel];
    }
  }
}
#endif  // SPHERICAL_HARMONICS_CUH
