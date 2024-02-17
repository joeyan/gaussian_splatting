#include <torch/extension.h>

void render_tiles_cuda(
        torch::Tensor uvs,
        torch::Tensor opacity,
        torch::Tensor rgb,
        torch::Tensor sigma_image,
        torch::Tensor gaussian_start_end_indices,
        torch::Tensor gaussian_indices_by_tile,
        torch::Tensor rendered_image);

void camera_projection_cuda(
    torch::Tensor xyz,
    torch::Tensor K,
    torch::Tensor uv
);

void compute_sigma_world_cuda(
    torch::Tensor quaternions,
    torch::Tensor scales,
    torch::Tensor sigma_world
);

void compute_projection_jacobian_cuda(
    torch::Tensor xyz,
    const float fx,
    const float fy,
    torch::Tensor J
);

void compute_sigma_image_cuda(
    torch::Tensor sigma_world,
    torch::Tensor J,
    torch::Tensor world_T_image,
    torch::Tensor 
);

void compute_tiles_cuda (
    torch::Tensor uvs,
    torch::Tensor sigma_image,
    int n_tiles_x,
    int n_tiles_y,
    float mh_dist,
    torch::Tensor gaussian_indices_per_tile,
    torch::Tensor num_gaussians_per_tile
);

void compute_tile_to_gaussian_vector(
    torch::Tensor gaussian_indices_per_tile,
    torch::Tensor num_gaussians_per_tile,
    torch::Tensor tile_to_gaussian_vector_offsets,
    torch::Tensor tile_to_gaussian_vector
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render_tiles_cuda", &render_tiles_cuda, "Render tiles CUDA");
    m.def("camera_projection_cuda", &camera_projection_cuda, "project point into image CUDA");
    m.def("compute_sigma_world_cuda", &compute_sigma_world_cuda, "compute sigma world CUDA");
    m.def("compute_projection_jacobian_cuda", &compute_projection_jacobian_cuda, "compute projection jacobian CUDA");
    m.def("compute_sigma_image_cuda", &compute_sigma_image_cuda, "compute sigma image CUDA");
    m.def("compute_tiles_cuda", &compute_tiles_cuda, "compute tiles CUDA");
    m.def("compute_tile_to_gaussian_vector", &compute_tile_to_gaussian_vector, "compute tile to gaussian vector CUDA");
}