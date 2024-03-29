#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "checks.cuh"
#include "spherical_harmonics.cuh"

template <typename T, unsigned int CHUNK_SIZE, unsigned int N_SH>
__global__ void render_tiles_kernel(
    const T *__restrict__ uvs, const T *__restrict__ opacity,
    const T *__restrict__ rgb, const T *__restrict__ sigma_image,
    const T *__restrict__ view_dir_by_pixel,
    const int *__restrict__ splat_start_end_idx_by_tile_idx,
    const int *__restrict__ gaussian_idx_by_splat_idx, const int image_width,
    const int image_height, bool use_fast_exp, int *num_splats_per_pixel,
    T *final_weight_per_pixel, T *image) {
  // grid = tiles, blocks = pixels within each tile
  const int u_splat = blockIdx.x * blockDim.x + threadIdx.x;
  const int v_splat = blockIdx.y * blockDim.y + threadIdx.y;
  const int tile_idx = blockIdx.x + blockIdx.y * gridDim.x;

  // keep threads around even if pixel is not valid for copying data
  bool valid_pixel = u_splat < image_width && v_splat < image_height;

  const int splat_idx_start = splat_start_end_idx_by_tile_idx[tile_idx];
  const int splat_idx_end = splat_start_end_idx_by_tile_idx[tile_idx + 1];
  int num_splats_this_tile = splat_idx_end - splat_idx_start;

  const int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
  const int block_size = blockDim.x * blockDim.y;

  T alpha_accum = 0.0;
  T alpha_weight = 0.0;
  int num_splats = 0;

  T view_dir[3];
  T sh_at_view_dir[N_SH];
  if (valid_pixel) {
#pragma unroll
    for (int axis = 0; axis < 3; axis++) {
      view_dir[axis] =
          view_dir_by_pixel[(v_splat * image_width + u_splat) * 3 + axis];
    }

    compute_sh_coeffs_for_view_dir<T, N_SH>(view_dir, sh_at_view_dir);
  }

  // shared memory copies of inputs
  __shared__ T _uvs[CHUNK_SIZE * 2];
  __shared__ T _opacity[CHUNK_SIZE];
  __shared__ T _rgb[CHUNK_SIZE * 3 * N_SH];
  __shared__ T _sigma_image[CHUNK_SIZE * 4];

  const int shared_image_size = 16 * 16 * 3;
  __shared__ T _image[shared_image_size];

#pragma unroll
  for (int i = thread_id; i < shared_image_size; i += block_size) {
    _image[i] = 0.0;
  }

  const int num_chunks = (num_splats_this_tile + CHUNK_SIZE - 1) / CHUNK_SIZE;
  // copy chunks
  for (int chunk_idx = 0; chunk_idx < num_chunks; chunk_idx++) {
    __syncthreads(); // make sure previous iteration is complete before
                     // modifying inputs
    for (int i = thread_id; i < CHUNK_SIZE; i += block_size) {
      const int tile_splat_idx = chunk_idx * CHUNK_SIZE + i;
      if (tile_splat_idx >= num_splats_this_tile) {
        break;
      }
      const int global_splat_idx = splat_idx_start + tile_splat_idx;

      const int gaussian_idx = gaussian_idx_by_splat_idx[global_splat_idx];
      _uvs[i * 2 + 0] = uvs[gaussian_idx * 2 + 0];
      _uvs[i * 2 + 1] = uvs[gaussian_idx * 2 + 1];
      _opacity[i] = opacity[gaussian_idx];

#pragma unroll
      for (int sh = 0; sh < N_SH; sh++) {
#pragma unroll
        for (int channel = 0; channel < 3; channel++) {
          // rgb dimensions = (splat_idx, channel_idx, sh_coeff_idx)
          _rgb[(i * 3 + channel) * N_SH + sh] =
              rgb[(gaussian_idx * 3 + channel) * N_SH + sh];
        }
      }

      _sigma_image[i * 4 + 0] = sigma_image[gaussian_idx * 4 + 0];
      _sigma_image[i * 4 + 1] = sigma_image[gaussian_idx * 4 + 1];
      _sigma_image[i * 4 + 2] = sigma_image[gaussian_idx * 4 + 2];
      _sigma_image[i * 4 + 3] = sigma_image[gaussian_idx * 4 + 3];
    }
    __syncthreads(); // wait for copying to complete before attempting to use
                     // data
    if (valid_pixel) {
      int chunk_start = chunk_idx * CHUNK_SIZE;
      int chunk_end = min((chunk_idx + 1) * CHUNK_SIZE, num_splats_this_tile);
      int num_splats_this_chunk = chunk_end - chunk_start;
      for (int i = 0; i < num_splats_this_chunk; i++) {
        if (alpha_accum > 0.999) {
          break;
        }
        const T u_mean = _uvs[i * 2 + 0];
        const T v_mean = _uvs[i * 2 + 1];

        const T u_diff = T(u_splat) - u_mean;
        const T v_diff = T(v_splat) - v_mean;

        // 2d covariance matrix
        const T a = _sigma_image[i * 4 + 0];
        const T b = _sigma_image[i * 4 + 1];
        const T c = _sigma_image[i * 4 + 2];
        const T d = _sigma_image[i * 4 + 3];
        T det = a * d - b * c;

        T alpha = 0.0;
        // skip any covariance matrices that are not positive definite
        if (det > 0.0) {
          if (det < 1e-14) {
            det += 1e-14;
          }
          // compute mahalanobis distance
          const T mh_sq = (d * u_diff * u_diff - (b + c) * u_diff * v_diff +
                           a * v_diff * v_diff) /
                          det;
          if (mh_sq > 0.0) {
            // probablity at this pixel normalized to have probability at the
            // center of the gaussian to be 1.0
            T norm_prob = 0.0;
            if (use_fast_exp) {
              norm_prob = __expf(-0.5 * mh_sq);
            } else {
              norm_prob = exp(-0.5 * mh_sq);
            }
            alpha = _opacity[i] * norm_prob;
          }
        }
        alpha_weight = 1.0 - alpha_accum;
        const T weight = alpha * (1.0 - alpha_accum);

        // compute rgb
        T computed_rgb[3];
        sh_to_rgb<T, N_SH>(_rgb + i * 3 * N_SH, sh_at_view_dir, computed_rgb);

#pragma unroll
        for (int channel = 0; channel < 3; channel++) {
          _image[(threadIdx.y * 16 + threadIdx.x) * 3 + channel] +=
              computed_rgb[channel] * weight;
        }

        alpha_accum += weight;
        num_splats++;
      } // end splat loop
    }   // valid pixel check
  }     // end chunk loop

  // copy back to global memory
  __syncthreads(); // wait for splatting to complete
  if (valid_pixel) {
    num_splats_per_pixel[v_splat * image_width + u_splat] = num_splats;
    final_weight_per_pixel[v_splat * image_width + u_splat] = alpha_weight;

#pragma unroll
    for (int channel = 0; channel < 3; channel++) {
      image[(v_splat * image_width + u_splat) * 3 + channel] =
          _image[(threadIdx.y * 16 + threadIdx.x) * 3 + channel];
    }
  }
}

void render_tiles_cuda(torch::Tensor uvs, torch::Tensor opacity,
                       torch::Tensor rgb, torch::Tensor sigma_image,
                       torch::Tensor view_dir_by_pixel,
                       torch::Tensor splat_start_end_idx_by_tile_idx,
                       torch::Tensor gaussian_idx_by_splat_idx,
                       torch::Tensor num_splats_per_pixel,
                       torch::Tensor final_weight_per_pixel,
                       torch::Tensor rendered_image) {
  CHECK_VALID_INPUT(uvs);
  CHECK_VALID_INPUT(opacity);
  CHECK_VALID_INPUT(rgb);
  CHECK_VALID_INPUT(sigma_image);
  CHECK_VALID_INPUT(view_dir_by_pixel);
  CHECK_VALID_INPUT(splat_start_end_idx_by_tile_idx);
  CHECK_VALID_INPUT(gaussian_idx_by_splat_idx);
  CHECK_VALID_INPUT(num_splats_per_pixel);
  CHECK_VALID_INPUT(final_weight_per_pixel);
  CHECK_VALID_INPUT(rendered_image);

  int N = uvs.size(0);
  TORCH_CHECK(uvs.size(1) == 2, "uvs must be Nx2 (u, v)");
  TORCH_CHECK(opacity.size(0) == N,
              "Opacity must have the same number of elements as uvs");
  TORCH_CHECK(opacity.size(1) == 1, "Opacity must be Nx1");
  TORCH_CHECK(rgb.size(0) == N,
              "RGB must have the same number of elements as uvs");
  TORCH_CHECK(rgb.size(1) == 3, "RGB must be Nx3");
  TORCH_CHECK(sigma_image.size(0) == N,
              "Sigma image must have the same number of elements as uvs");
  TORCH_CHECK(sigma_image.size(1) == 2, "Sigma image must be Nx2x2");
  TORCH_CHECK(sigma_image.size(1) == 2, "Sigma image must be Nx2x2");
  TORCH_CHECK(rendered_image.size(2) == 3, "Image must be HxWx3");

  int image_height = rendered_image.size(0);
  int image_width = rendered_image.size(1);
  TORCH_CHECK(view_dir_by_pixel.size(0) == image_height,
              "view_dir_by_pixel must have the same height as the image");
  TORCH_CHECK(view_dir_by_pixel.size(1) == image_width,
              "view_dir_by_pixel must have the same width as the image");
  TORCH_CHECK(view_dir_by_pixel.size(2) == 3,
              "view_dir_by_pixel must have 3 channels");

  int num_tiles_x = (image_width + 16 - 1) / 16;
  int num_tiles_y = (image_height + 16 - 1) / 16;

  dim3 block_size(16, 16, 1);
  dim3 grid_size(num_tiles_x, num_tiles_y, 1);

  int num_sh_coeff;
  if (rgb.dim() == 3) {
    num_sh_coeff = rgb.size(2);
  } else {
    num_sh_coeff = 1;
  }
  if (uvs.dtype() == torch::kFloat32) {
    CHECK_FLOAT_TENSOR(opacity);
    CHECK_FLOAT_TENSOR(rgb);
    CHECK_FLOAT_TENSOR(sigma_image);
    CHECK_FLOAT_TENSOR(view_dir_by_pixel);
    CHECK_INT_TENSOR(splat_start_end_idx_by_tile_idx);
    CHECK_INT_TENSOR(gaussian_idx_by_splat_idx);
    CHECK_INT_TENSOR(num_splats_per_pixel);
    CHECK_FLOAT_TENSOR(final_weight_per_pixel);
    CHECK_FLOAT_TENSOR(rendered_image);

    if (num_sh_coeff == 1) {
      render_tiles_kernel<float, 960, 1><<<grid_size, block_size>>>(
          uvs.data_ptr<float>(), opacity.data_ptr<float>(),
          rgb.data_ptr<float>(), sigma_image.data_ptr<float>(),
          view_dir_by_pixel.data_ptr<float>(),
          splat_start_end_idx_by_tile_idx.data_ptr<int>(),
          gaussian_idx_by_splat_idx.data_ptr<int>(), image_width, image_height,
          true, num_splats_per_pixel.data_ptr<int>(),
          final_weight_per_pixel.data_ptr<float>(),
          rendered_image.data_ptr<float>());
    } else if (num_sh_coeff == 4) {
      render_tiles_kernel<float, 576, 4><<<grid_size, block_size>>>(
          uvs.data_ptr<float>(), opacity.data_ptr<float>(),
          rgb.data_ptr<float>(), sigma_image.data_ptr<float>(),
          view_dir_by_pixel.data_ptr<float>(),
          splat_start_end_idx_by_tile_idx.data_ptr<int>(),
          gaussian_idx_by_splat_idx.data_ptr<int>(), image_width, image_height,
          true, num_splats_per_pixel.data_ptr<int>(),
          final_weight_per_pixel.data_ptr<float>(),
          rendered_image.data_ptr<float>());
    } else if (num_sh_coeff == 9) {
      render_tiles_kernel<float, 320, 9><<<grid_size, block_size>>>(
          uvs.data_ptr<float>(), opacity.data_ptr<float>(),
          rgb.data_ptr<float>(), sigma_image.data_ptr<float>(),
          view_dir_by_pixel.data_ptr<float>(),
          splat_start_end_idx_by_tile_idx.data_ptr<int>(),
          gaussian_idx_by_splat_idx.data_ptr<int>(), image_width, image_height,
          true, num_splats_per_pixel.data_ptr<int>(),
          final_weight_per_pixel.data_ptr<float>(),
          rendered_image.data_ptr<float>());
    } else if (num_sh_coeff == 16) {
      render_tiles_kernel<float, 160, 16><<<grid_size, block_size>>>(
          uvs.data_ptr<float>(), opacity.data_ptr<float>(),
          rgb.data_ptr<float>(), sigma_image.data_ptr<float>(),
          view_dir_by_pixel.data_ptr<float>(),
          splat_start_end_idx_by_tile_idx.data_ptr<int>(),
          gaussian_idx_by_splat_idx.data_ptr<int>(), image_width, image_height,
          true, num_splats_per_pixel.data_ptr<int>(),
          final_weight_per_pixel.data_ptr<float>(),
          rendered_image.data_ptr<float>());
    } else {
      AT_ERROR("Unsupported number of SH coefficients: ", num_sh_coeff);
    }
  } else if (uvs.dtype() == torch::kFloat64) {
    CHECK_DOUBLE_TENSOR(opacity);
    CHECK_DOUBLE_TENSOR(rgb);
    CHECK_DOUBLE_TENSOR(sigma_image);
    CHECK_INT_TENSOR(splat_start_end_idx_by_tile_idx);
    CHECK_INT_TENSOR(gaussian_idx_by_splat_idx);
    CHECK_INT_TENSOR(num_splats_per_pixel);
    CHECK_DOUBLE_TENSOR(final_weight_per_pixel);
    CHECK_DOUBLE_TENSOR(rendered_image);
    if (num_sh_coeff == 1) {
      render_tiles_kernel<double, 320, 1><<<grid_size, block_size>>>(
          uvs.data_ptr<double>(), opacity.data_ptr<double>(),
          rgb.data_ptr<double>(), sigma_image.data_ptr<double>(),
          view_dir_by_pixel.data_ptr<double>(),
          splat_start_end_idx_by_tile_idx.data_ptr<int>(),
          gaussian_idx_by_splat_idx.data_ptr<int>(), image_width, image_height,
          false, num_splats_per_pixel.data_ptr<int>(),
          final_weight_per_pixel.data_ptr<double>(),
          rendered_image.data_ptr<double>());
    } else if (num_sh_coeff == 4) {
      render_tiles_kernel<double, 160, 4><<<grid_size, block_size>>>(
          uvs.data_ptr<double>(), opacity.data_ptr<double>(),
          rgb.data_ptr<double>(), sigma_image.data_ptr<double>(),
          view_dir_by_pixel.data_ptr<double>(),
          splat_start_end_idx_by_tile_idx.data_ptr<int>(),
          gaussian_idx_by_splat_idx.data_ptr<int>(), image_width, image_height,
          false, num_splats_per_pixel.data_ptr<int>(),
          final_weight_per_pixel.data_ptr<double>(),
          rendered_image.data_ptr<double>());
    } else if (num_sh_coeff == 9) {
      render_tiles_kernel<double, 128, 9><<<grid_size, block_size>>>(
          uvs.data_ptr<double>(), opacity.data_ptr<double>(),
          rgb.data_ptr<double>(), sigma_image.data_ptr<double>(),
          view_dir_by_pixel.data_ptr<double>(),
          splat_start_end_idx_by_tile_idx.data_ptr<int>(),
          gaussian_idx_by_splat_idx.data_ptr<int>(), image_width, image_height,
          false, num_splats_per_pixel.data_ptr<int>(),
          final_weight_per_pixel.data_ptr<double>(),
          rendered_image.data_ptr<double>());
    } else if (num_sh_coeff == 16) {
      render_tiles_kernel<double, 64, 16><<<grid_size, block_size>>>(
          uvs.data_ptr<double>(), opacity.data_ptr<double>(),
          rgb.data_ptr<double>(), sigma_image.data_ptr<double>(),
          view_dir_by_pixel.data_ptr<double>(),
          splat_start_end_idx_by_tile_idx.data_ptr<int>(),
          gaussian_idx_by_splat_idx.data_ptr<int>(), image_width, image_height,
          false, num_splats_per_pixel.data_ptr<int>(),
          final_weight_per_pixel.data_ptr<double>(),
          rendered_image.data_ptr<double>());
    } else {
      AT_ERROR("Unsupported number of SH coefficients: ", num_sh_coeff);
    }
  } else {
    AT_ERROR("Inputs must be float32 or float64");
  }
  cudaDeviceSynchronize();
}
