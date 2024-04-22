#include <torch/extension.h>

void render_tiles_cuda(
    torch::Tensor uvs,
    torch::Tensor opacity,
    torch::Tensor rgb,
    torch::Tensor conic,
    torch::Tensor view_dir_by_pixel,
    torch::Tensor splat_start_end_idx_by_tile_idx,
    torch::Tensor gaussian_idx_by_splat_idx,
    torch::Tensor background_rgb,
    torch::Tensor num_splats_per_pixel,
    torch::Tensor final_weight_per_pixel,
    torch::Tensor rendered_image
);

void render_tiles_backward_cuda(
    torch::Tensor uvs,
    torch::Tensor opacity,
    torch::Tensor rgb,
    torch::Tensor conic,
    torch::Tensor view_dir_by_pixel,
    torch::Tensor splat_start_end_idx_by_tile_idx,
    torch::Tensor gaussian_idx_by_splat_idx,
    torch::Tensor background_rgb,
    torch::Tensor num_splats_per_pixel,
    torch::Tensor final_weight_per_pixel,
    torch::Tensor grad_image,
    torch::Tensor grad_rgb,
    torch::Tensor grad_opacity,
    torch::Tensor grad_uvs,
    torch::Tensor grad_conic
);

void camera_projection_cuda(torch::Tensor xyz, torch::Tensor K, torch::Tensor uv);

void camera_projection_backward_cuda(
    torch::Tensor xyz,
    torch::Tensor K,
    torch::Tensor uv_grad_out,
    torch::Tensor xyz_grad_in
);

void compute_sigma_world_cuda(
    torch::Tensor quaternion,
    torch::Tensor scale,
    torch::Tensor sigma_world
);

void compute_sigma_world_backward_cuda(
    torch::Tensor quaternion,
    torch::Tensor scale,
    torch::Tensor sigma_world_grad_out,
    torch::Tensor quaternion_grad_in,
    torch::Tensor scale_grad_in
);

void compute_projection_jacobian_cuda(torch::Tensor xyz, torch::Tensor K, torch::Tensor J);

void compute_projection_jacobian_backward_cuda(
    torch::Tensor xyz,
    torch::Tensor K,
    torch::Tensor jac_grad_out,
    torch::Tensor xyz_grad_in
);

void compute_conic_cuda(
    torch::Tensor sigma_world,
    torch::Tensor J,
    torch::Tensor camera_T_world,
    torch::Tensor conic
);

void compute_conic_backward_cuda(
    torch::Tensor sigma_world,
    torch::Tensor J,
    torch::Tensor camera_T_world,
    torch::Tensor conic_grad_out,
    torch::Tensor sigma_world_grad_in,
    torch::Tensor J_grad_in
);

void compute_tiles_cuda(
    torch::Tensor uvs,
    torch::Tensor conic,
    int n_tiles_x,
    int n_tiles_y,
    float mh_dist,
    torch::Tensor gaussian_indices_per_tile,
    torch::Tensor num_gaussians_per_tile
);

std::tuple<torch::Tensor, torch::Tensor> get_sorted_gaussian_list(
    const int max_tiles_per_gaussian,
    torch::Tensor uvs,
    torch::Tensor xyz_camera_frame,
    torch::Tensor conic,
    const int n_tiles_x,
    const int n_tiles_y,
    const float mh_dist
);

void compute_splat_to_gaussian_id_vector_cuda(
    torch::Tensor gaussian_indices_per_tile,
    torch::Tensor num_gaussians_per_tile,
    torch::Tensor splat_to_gaussian_id_vector_offsets,
    torch::Tensor splat_to_gaussian_id_vector,
    torch::Tensor tile_idx_by_splat_idx
);

void precompute_rgb_from_sh_cuda(
    const torch::Tensor xyz,
    const torch::Tensor sh_coeff,
    const torch::Tensor camera_T_world,
    torch::Tensor rgb
);

void precompute_rgb_from_sh_backward_cuda(
    const torch::Tensor xyz,
    const torch::Tensor camera_T_world,
    const torch::Tensor grad_rgb,
    torch::Tensor grad_sh
);

void render_depth_cuda(
    torch::Tensor xyz_camera_frame,
    torch::Tensor uvs,
    torch::Tensor opacity,
    torch::Tensor conic,
    torch::Tensor splat_start_end_idx_by_tile_idx,
    torch::Tensor gaussian_idx_by_splat_idx,
    const float alpha_threshold,
    torch::Tensor depth_image
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("render_tiles_cuda", &render_tiles_cuda, "Render tiles CUDA");
    m.def("render_tiles_backward_cuda", &render_tiles_backward_cuda, "Render tiles backward");
    m.def("camera_projection_cuda", &camera_projection_cuda, "project point into image CUDA");
    m.def(
        "camera_projection_backward_cuda",
        &camera_projection_backward_cuda,
        "project point into image backward CUDA"
    );
    m.def("compute_sigma_world_cuda", &compute_sigma_world_cuda, "compute sigma world CUDA");
    m.def(
        "compute_sigma_world_backward_cuda",
        &compute_sigma_world_backward_cuda,
        "compute sigma world backward CUDA"
    );
    m.def(
        "compute_projection_jacobian_cuda",
        &compute_projection_jacobian_cuda,
        "compute projection jacobian CUDA"
    );
    m.def(
        "compute_projection_jacobian_backward_cuda",
        &compute_projection_jacobian_backward_cuda,
        "compute projection jacobian backward CUDA"
    );
    m.def("compute_conic_cuda", &compute_conic_cuda, "compute conic CUDA");
    m.def(
        "compute_conic_backward_cuda", &compute_conic_backward_cuda, "compute conic backward CUDA"
    );
    m.def("compute_tiles_cuda", &compute_tiles_cuda, "compute tiles CUDA");
    m.def(
        "compute_splat_to_gaussian_id_vector_cuda",
        &compute_splat_to_gaussian_id_vector_cuda,
        "compute tile to gaussian vector CUDA"
    );
    m.def(
        "get_sorted_gaussian_list",
        &get_sorted_gaussian_list,
        "get sorted gaussian list"
    );
    m.def(
        "precompute_rgb_from_sh_cuda",
        &precompute_rgb_from_sh_cuda,
        "precompute rgb from sh per gaussian"
    );
    m.def(
        "precompute_rgb_from_sh_backward_cuda",
        &precompute_rgb_from_sh_backward_cuda,
        "precompute rgb from sh per gaussian backward"
    );
    m.def("render_depth_cuda", &render_depth_cuda, "Render depth CUDA");
}
