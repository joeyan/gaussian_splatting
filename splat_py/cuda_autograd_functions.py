import torch

from splat_cuda import (
    camera_projection_cuda,
    camera_projection_backward_cuda,
    compute_sigma_world_cuda,
    compute_sigma_world_backward_cuda,
    compute_projection_jacobian_cuda,
    compute_projection_jacobian_backward_cuda,
    compute_conic_cuda,
    compute_conic_backward_cuda,
    render_tiles_cuda,
    render_tiles_backward_cuda,
    precompute_rgb_from_sh_cuda,
    precompute_rgb_from_sh_backward_cuda,
)


class CameraPointProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz_camera, K):
        uv = torch.zeros(xyz_camera.shape[0], 2, dtype=xyz_camera.dtype, device=xyz_camera.device)
        camera_projection_cuda(xyz_camera, K, uv)
        ctx.save_for_backward(xyz_camera, K)
        return uv

    @staticmethod
    def backward(ctx, grad_uv):
        xyz_camera, K = ctx.saved_tensors
        grad_xyz_camera = torch.zeros(
            xyz_camera.shape, dtype=xyz_camera.dtype, device=xyz_camera.device
        )
        camera_projection_backward_cuda(xyz_camera, K, grad_uv, grad_xyz_camera)
        return grad_xyz_camera, None


class ComputeSigmaWorld(torch.autograd.Function):
    @staticmethod
    def forward(ctx, quaternion, scale):
        sigma_world = torch.zeros(
            quaternion.shape[0],
            3,
            3,
            dtype=quaternion.dtype,
            device=quaternion.device,
        )
        compute_sigma_world_cuda(quaternion, scale, sigma_world)
        ctx.save_for_backward(quaternion, scale)
        return sigma_world

    @staticmethod
    def backward(ctx, grad_sigma_world):
        quaternion, scale = ctx.saved_tensors
        grad_quaternion = torch.zeros(
            quaternion.shape, dtype=quaternion.dtype, device=quaternion.device
        )
        grad_scale = torch.zeros(scale.shape, dtype=scale.dtype, device=scale.device)
        compute_sigma_world_backward_cuda(
            quaternion, scale, grad_sigma_world, grad_quaternion, grad_scale
        )
        return grad_quaternion, grad_scale


class ComputeProjectionJacobian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz_camera, K):
        jacobian = torch.zeros(
            xyz_camera.shape[0], 2, 3, dtype=xyz_camera.dtype, device=xyz_camera.device
        )
        compute_projection_jacobian_cuda(xyz_camera, K, jacobian)
        ctx.save_for_backward(xyz_camera, K)
        return jacobian

    @staticmethod
    def backward(ctx, grad_jacobian):
        xyz_camera, K = ctx.saved_tensors
        grad_xyz_camera = torch.zeros(
            xyz_camera.shape, dtype=xyz_camera.dtype, device=xyz_camera.device
        )
        compute_projection_jacobian_backward_cuda(xyz_camera, K, grad_jacobian, grad_xyz_camera)
        return grad_xyz_camera, None


class ComputeConic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigma_world, J, camera_T_world):
        conic = torch.zeros(J.shape[0], 3, dtype=sigma_world.dtype, device=sigma_world.device)
        compute_conic_cuda(sigma_world, J, camera_T_world, conic)
        ctx.save_for_backward(sigma_world, camera_T_world, J)
        return conic

    @staticmethod
    def backward(ctx, grad_conic):
        sigma_world, camera_T_world, J = ctx.saved_tensors
        grad_sigma_world = torch.zeros(
            sigma_world.shape, dtype=sigma_world.dtype, device=sigma_world.device
        )
        grad_J = torch.zeros(J.shape, dtype=J.dtype, device=J.device)
        compute_conic_backward_cuda(
            sigma_world, J, camera_T_world, grad_conic, grad_sigma_world, grad_J
        )
        return grad_sigma_world, grad_J, None


class PrecomputeRGBFromSH(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sh_coeffs, xyz, camera_T_world):
        rgb = torch.zeros(xyz.shape[0], 3, dtype=sh_coeffs.dtype, device=sh_coeffs.device)
        precompute_rgb_from_sh_cuda(xyz, sh_coeffs, camera_T_world, rgb)

        if sh_coeffs.dim() == 2:
            num_sh_coeff = torch.tensor(1, dtype=torch.int, device=sh_coeffs.device)
        else:
            num_sh_coeff = torch.tensor(
                sh_coeffs.shape[2], dtype=torch.int, device=sh_coeffs.device
            )
        ctx.save_for_backward(xyz, camera_T_world, num_sh_coeff)
        return rgb

    @staticmethod
    def backward(ctx, grad_rgb):
        xyz, camera_T_world, num_sh_coeff = ctx.saved_tensors
        grad_sh_coeffs = torch.zeros(
            xyz.shape[0], 3, num_sh_coeff.item(), dtype=xyz.dtype, device=xyz.device
        )
        precompute_rgb_from_sh_backward_cuda(xyz, camera_T_world, grad_rgb, grad_sh_coeffs)
        return grad_sh_coeffs, None, None


class RenderImage(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rgb,
        opacity,
        uvs,
        conic,
        rays,
        splat_start_end_idx_by_tile_idx,
        sorted_gaussian_idx_by_splat_idx,
        image_size,
        background_rgb,
    ):
        rendered_image = torch.zeros(
            image_size[0], image_size[1], 3, dtype=rgb.dtype, device=rgb.device
        )
        num_splats_per_pixel = torch.zeros(
            image_size[0], image_size[1], dtype=torch.int, device=rgb.device
        )
        final_weight_per_pixel = torch.zeros(
            image_size[0], image_size[1], dtype=rgb.dtype, device=rgb.device
        )

        render_tiles_cuda(
            uvs,
            opacity,
            rgb,
            conic,
            rays,
            splat_start_end_idx_by_tile_idx,
            sorted_gaussian_idx_by_splat_idx,
            background_rgb,
            num_splats_per_pixel,
            final_weight_per_pixel,
            rendered_image,
        )
        ctx.save_for_backward(
            uvs,
            opacity,
            rgb,
            conic,
            rays,
            splat_start_end_idx_by_tile_idx,
            sorted_gaussian_idx_by_splat_idx,
            background_rgb,
            num_splats_per_pixel,
            final_weight_per_pixel,
        )
        return rendered_image

    @staticmethod
    def backward(ctx, grad_rendered_image):
        (
            uvs,
            opacity,
            rgb,
            conic,
            rays,
            splat_start_end_idx_by_tile_idx,
            sorted_gaussian_idx_by_splat_idx,
            background_rgb,
            num_splats_per_pixel,
            final_weight_per_pixel,
        ) = ctx.saved_tensors
        grad_rgb = torch.zeros_like(rgb)
        grad_opacity = torch.zeros_like(opacity)
        grad_uv = torch.zeros_like(uvs)
        grad_conic = torch.zeros_like(conic)

        # ensure input is contiguous
        grad_rendered_image = grad_rendered_image.contiguous()
        render_tiles_backward_cuda(
            uvs,
            opacity,
            rgb,
            conic,
            rays,
            splat_start_end_idx_by_tile_idx,
            sorted_gaussian_idx_by_splat_idx,
            background_rgb,
            num_splats_per_pixel,
            final_weight_per_pixel,
            grad_rendered_image,
            grad_rgb,
            grad_opacity,
            grad_uv,
            grad_conic,
        )
        return grad_rgb, grad_opacity, grad_uv, grad_conic, None, None, None, None, None
