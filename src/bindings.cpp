#include <torch/extension.h>

void render_tiles_cuda(torch::Tensor uvs, torch::Tensor opacity,
                       torch::Tensor rgb, torch::Tensor sigma_image,
                       torch::Tensor view_dir_by_pixel,
                       torch::Tensor splat_start_end_idx_by_tile_idx,
                       torch::Tensor gaussian_idx_by_splat_idx,
                       torch::Tensor num_splats_per_pixel,
                       torch::Tensor final_weight_per_pixel,
                       torch::Tensor rendered_image);

void render_tiles_backward_cuda(
    torch::Tensor uvs, torch::Tensor opacity, torch::Tensor rgb,
    torch::Tensor sigma_image, torch::Tensor view_dir_by_pixel,
    torch::Tensor splat_start_end_idx_by_tile_idx,
    torch::Tensor gaussian_idx_by_splat_idx, torch::Tensor num_splats_per_pixel,
    torch::Tensor final_weight_per_pixel, torch::Tensor grad_image,
    torch::Tensor grad_rgb, torch::Tensor grad_opacity, torch::Tensor grad_uvs,
    torch::Tensor grad_sigma_image);

void camera_projection_cuda(torch::Tensor xyz, torch::Tensor K,
                            torch::Tensor uv);

void camera_projection_backward_cuda(torch::Tensor xyz, torch::Tensor K,
                                     torch::Tensor uv_grad_out,
                                     torch::Tensor xyz_grad_in);

void compute_sigma_world_cuda(torch::Tensor quaternions, torch::Tensor scales,
                              torch::Tensor sigma_world);

void compute_sigma_world_backward_cuda(torch::Tensor quaternions,
                                       torch::Tensor scales,
                                       torch::Tensor sigma_world_grad_out,
                                       torch::Tensor quaternions_grad_in,
                                       torch::Tensor scales_grad_in);

void compute_projection_jacobian_cuda(torch::Tensor xyz, torch::Tensor K,
                                      torch::Tensor J);

void compute_projection_jacobian_backward_cuda(torch::Tensor xyz,
                                               torch::Tensor K,
                                               torch::Tensor jac_grad_out,
                                               torch::Tensor xyz_grad_in);

void compute_sigma_image_cuda(torch::Tensor sigma_world, torch::Tensor J,
                              torch::Tensor world_T_image, torch::Tensor);

void compute_sigma_image_backward_cuda(torch::Tensor sigma_world,
                                       torch::Tensor J,
                                       torch::Tensor world_T_image,
                                       torch::Tensor sigma_image_grad_out,
                                       torch::Tensor sigma_world_grad_in,
                                       torch::Tensor J_grad_in);

void compute_tiles_cuda(torch::Tensor uvs, torch::Tensor sigma_image,
                        int n_tiles_x, int n_tiles_y, float mh_dist,
                        torch::Tensor gaussian_indices_per_tile,
                        torch::Tensor num_gaussians_per_tile);

void compute_splat_to_gaussian_id_vector_cuda(
    torch::Tensor gaussian_indices_per_tile,
    torch::Tensor num_gaussians_per_tile,
    torch::Tensor splat_to_gaussian_id_vector_offsets,
    torch::Tensor splat_to_gaussian_id_vector,
    torch::Tensor tile_idx_by_splat_idx);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("render_tiles_cuda", &render_tiles_cuda, "Render tiles CUDA");
  m.def("render_tiles_backward_cuda", &render_tiles_backward_cuda,
        "Render tiles backward");
  m.def("camera_projection_cuda", &camera_projection_cuda,
        "project point into image CUDA");
  m.def("camera_projection_backward_cuda", &camera_projection_backward_cuda,
        "project point into image backward CUDA");
  m.def("compute_sigma_world_cuda", &compute_sigma_world_cuda,
        "compute sigma world CUDA");
  m.def("compute_sigma_world_backward_cuda", &compute_sigma_world_backward_cuda,
        "compute sigma world backward CUDA");
  m.def("compute_projection_jacobian_cuda", &compute_projection_jacobian_cuda,
        "compute projection jacobian CUDA");
  m.def("compute_projection_jacobian_backward_cuda",
        &compute_projection_jacobian_backward_cuda,
        "compute projection jacobian backward CUDA");
  m.def("compute_sigma_image_cuda", &compute_sigma_image_cuda,
        "compute sigma image CUDA");
  m.def("compute_sigma_image_backward_cuda", &compute_sigma_image_backward_cuda,
        "compute sigma image backward CUDA");
  m.def("compute_tiles_cuda", &compute_tiles_cuda, "compute tiles CUDA");
  m.def("compute_splat_to_gaussian_id_vector_cuda",
        &compute_splat_to_gaussian_id_vector_cuda,
        "compute tile to gaussian vector CUDA");
}
