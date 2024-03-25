import torch

from splat_cuda import (
    camera_projection_cuda,
    camera_projection_backward_cuda,
    compute_sigma_world_cuda,
    compute_sigma_world_backward_cuda,
    compute_projection_jacobian_cuda,
    compute_projection_jacobian_backward_cuda,
    compute_sigma_image_cuda,
    compute_sigma_image_backward_cuda,
    render_tiles_cuda,
    render_tiles_backward_cuda,
)


class CameraPointProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz_camera, K):
        uv = torch.zeros(
            xyz_camera.shape[0], 2, dtype=xyz_camera.dtype, device=xyz_camera.device
        )
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
    def forward(ctx, quaternions, scales):
        sigma_world = torch.zeros(
            quaternions.shape[0],
            3,
            3,
            dtype=quaternions.dtype,
            device=quaternions.device,
        )
        compute_sigma_world_cuda(quaternions, scales, sigma_world)
        ctx.save_for_backward(quaternions, scales)
        return sigma_world

    @staticmethod
    def backward(ctx, grad_sigma_world):
        quaternions, scales = ctx.saved_tensors
        grad_quaternions = torch.zeros(
            quaternions.shape, dtype=quaternions.dtype, device=quaternions.device
        )
        grad_scales = torch.zeros(
            scales.shape, dtype=scales.dtype, device=scales.device
        )
        compute_sigma_world_backward_cuda(
            quaternions, scales, grad_sigma_world, grad_quaternions, grad_scales
        )
        return grad_quaternions, grad_scales


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
        compute_projection_jacobian_backward_cuda(
            xyz_camera, K, grad_jacobian, grad_xyz_camera
        )
        return grad_xyz_camera, None


class ComputeSigmaImage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sigma_world, J, world_T_image):
        sigma_image = torch.zeros(
            J.shape[0], 2, 2, dtype=sigma_world.dtype, device=sigma_world.device
        )
        compute_sigma_image_cuda(sigma_world, J, world_T_image, sigma_image)
        ctx.save_for_backward(sigma_world, world_T_image, J)
        return sigma_image

    @staticmethod
    def backward(ctx, grad_sigma_image):
        sigma_world, world_T_image, J = ctx.saved_tensors
        grad_sigma_world = torch.zeros(
            sigma_world.shape, dtype=sigma_world.dtype, device=sigma_world.device
        )
        grad_J = torch.zeros(J.shape, dtype=J.dtype, device=J.device)
        compute_sigma_image_backward_cuda(
            sigma_world, J, world_T_image, grad_sigma_image, grad_sigma_world, grad_J
        )
        return grad_sigma_world, grad_J, None


class RenderImage(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        rgb,
        opacity,
        uvs,
        sigma_image,
        splat_start_end_idx_by_tile_idx,
        sorted_gaussian_idx_by_splat_idx,
        image_size,
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
            sigma_image,
            splat_start_end_idx_by_tile_idx,
            sorted_gaussian_idx_by_splat_idx,
            num_splats_per_pixel,
            final_weight_per_pixel,
            rendered_image,
        )
        ctx.save_for_backward(
            uvs,
            opacity,
            rgb,
            sigma_image,
            splat_start_end_idx_by_tile_idx,
            sorted_gaussian_idx_by_splat_idx,
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
            sigma_image,
            splat_start_end_idx_by_tile_idx,
            sorted_gaussian_idx_by_splat_idx,
            num_splats_per_pixel,
            final_weight_per_pixel,
        ) = ctx.saved_tensors
        grad_rgb = torch.zeros_like(rgb)
        grad_opacity = torch.zeros_like(opacity)
        grad_uv = torch.zeros_like(uvs)
        grad_sigma_image = torch.zeros_like(sigma_image)

        # ensure input is contiguous
        grad_rendered_image = grad_rendered_image.contiguous()
        render_tiles_backward_cuda(
            uvs,
            opacity,
            rgb,
            sigma_image,
            splat_start_end_idx_by_tile_idx,
            sorted_gaussian_idx_by_splat_idx,
            num_splats_per_pixel,
            final_weight_per_pixel,
            grad_rendered_image,
            grad_rgb,
            grad_opacity,
            grad_uv,
            grad_sigma_image,
        )
        if torch.sum(~torch.isfinite(grad_rgb)) > 0:
            print("num grad_rbg nan", torch.sum(~torch.isfinite(grad_rgb)))
        if torch.sum(~torch.isfinite(grad_opacity)) > 0:
            print("num grad_opacity nan", torch.sum(~torch.isfinite(grad_opacity)))
        if torch.sum(~torch.isfinite(grad_uv)) > 0:
            print("num grad_uv nan", torch.sum(~torch.isfinite(grad_uv)))
        if torch.sum(~torch.isfinite(grad_sigma_image)) > 0:
            print("num grad_sigma_image nan", torch.sum(~torch.isfinite(grad_sigma_image)))
        return grad_rgb, grad_opacity, grad_uv, grad_sigma_image, None, None, None
