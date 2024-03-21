import math
import torch

from splat_py.constants import *
from splat_py.utils import inverse_sigmoid, sample_pdf
from splat_cuda import compute_sigma_world_cuda


class GSTrainer:
    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.update_optimizer(iter_num=0)
        self.reset_grad_accum()

    def update_optimizer(self, iter_num):
        print("Updating optimizer")
        # add new params to optimizer
        new_lr = BASE_LR * (NUM_ITERS - iter_num * 0.9) / NUM_ITERS
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.gaussians.xyz, "lr": new_lr * XYZ_LR_MULTIPLIER},
                {
                    "params": self.gaussians.quaternions,
                    "lr": new_lr * QUAT_LR_MULTIPLIER,
                },
                {"params": self.gaussians.scales, "lr": new_lr * SCALE_LR_MULTIPLIER},
                {
                    "params": self.gaussians.opacities,
                    "lr": new_lr * OPACITY_LR_MULTIPLIER,
                },
                {"params": self.gaussians.rgb, "lr": new_lr * RGB_LR_MULTIPLIER},
            ],
        )

    def check_nans(self):
        if torch.any(~torch.isfinite(self.gaussians.xyz)):
            print("NaN or inf in xyz")

        if torch.any(~torch.isfinite(self.gaussians.quaternions)):
            print("NaN or inf in quaternions")

        if torch.any(~torch.isfinite(self.gaussians.scales)):
            print("NaN or inf in scales")

        if torch.any(~torch.isfinite(self.gaussians.opacities)):
            print("NaN or inf in opacities")

        if torch.any(~torch.isfinite(self.gaussians.rgb)):
            print("NaN or inf in rgb")

    def reset_grad_accum(self):
        # reset grad accumulators
        self.uv_grad_accum = torch.zeros(
            (self.gaussians.xyz.shape[0], 2),
            dtype=self.gaussians.xyz.dtype,
            device=self.gaussians.xyz.device,
        )
        self.xyz_grad_accum = torch.zeros(
            self.gaussians.xyz.shape,
            dtype=self.gaussians.xyz.dtype,
            device=self.gaussians.xyz.device,
        )
        self.grad_accum_count = torch.zeros(
            self.gaussians.xyz.shape[0],
            dtype=torch.int,
            device=self.gaussians.xyz.device,
        )

    def reset_opacities(self, iter):
        print("\t\tResetting opacities")
        self.gaussians.opacities = torch.nn.Parameter(
            torch.ones_like(self.gaussians.opacities) * inverse_sigmoid(INITIAL_OPACITY)
        )
        self.update_optimizer(iter)
        self.reset_grad_accum()

    def delete_param_from_optimizer(self, new_param, keep_mask, param_index):
        old_optimizer_param = self.optimizer.param_groups[param_index]["params"][0]
        optimizer_param_state = self.optimizer.state[old_optimizer_param]
        del self.optimizer.state[old_optimizer_param]

        optimizer_param_state["exp_avg"] = optimizer_param_state["exp_avg"][
            keep_mask, :
        ]
        optimizer_param_state["exp_avg_sq"] = optimizer_param_state["exp_avg_sq"][
            keep_mask, :
        ]

        del self.optimizer.param_groups[param_index]["params"][0]
        del self.optimizer.param_groups[param_index]["params"]

        self.optimizer.param_groups[param_index]["params"] = [new_param]
        self.optimizer.state[new_param] = optimizer_param_state

    def delete_gaussians_from_optimizer(self, keep_mask):
        self.delete_param_from_optimizer(self.gaussians.xyz, keep_mask, 0)
        self.delete_param_from_optimizer(self.gaussians.quaternions, keep_mask, 1)
        self.delete_param_from_optimizer(self.gaussians.scales, keep_mask, 2)
        self.delete_param_from_optimizer(self.gaussians.opacities, keep_mask, 3)
        self.delete_param_from_optimizer(self.gaussians.rgb, keep_mask, 4)

    def clone_params_in_optimizer(self, new_param, clone_mask, param_index):
        old_optimizer_param = self.optimizer.param_groups[param_index]["params"][0]
        optimizer_param_state = self.optimizer.state[old_optimizer_param]

        optimizer_param_state["exp_avg"] = torch.cat(
            [
                optimizer_param_state["exp_avg"],
                torch.zeros(
                    clone_mask.sum(),
                    optimizer_param_state["exp_avg"].shape[1],
                    device=optimizer_param_state["exp_avg"].device,
                    dtype=optimizer_param_state["exp_avg"].dtype,
                ),
            ],
            dim=0,
        )
        optimizer_param_state["exp_avg_sq"] = torch.cat(
            [
                optimizer_param_state["exp_avg_sq"],
                torch.zeros(
                    clone_mask.sum(),
                    optimizer_param_state["exp_avg_sq"].shape[1],
                    device=optimizer_param_state["exp_avg_sq"].device,
                    dtype=optimizer_param_state["exp_avg_sq"].dtype,
                ),
            ],
            dim=0,
        )

        del self.optimizer.state[old_optimizer_param]
        del old_optimizer_param
        self.optimizer.param_groups[param_index]["params"] = [new_param]
        self.optimizer.state[new_param] = optimizer_param_state

    def clone_gaussians_in_optimizer(self, clone_mask):
        self.clone_params_in_optimizer(self.gaussians.xyz, clone_mask, 0)
        self.clone_params_in_optimizer(self.gaussians.quaternions, clone_mask, 1)
        self.clone_params_in_optimizer(self.gaussians.scales, clone_mask, 2)
        self.clone_params_in_optimizer(self.gaussians.opacities, clone_mask, 3)
        self.clone_params_in_optimizer(self.gaussians.rgb, clone_mask, 4)

    def delete_gaussians(self, keep_mask):
        self.gaussians.xyz = torch.nn.Parameter(
            self.gaussians.xyz.detach()[keep_mask, :]
        )
        self.gaussians.quaternions = torch.nn.Parameter(
            self.gaussians.quaternions.detach()[keep_mask, :]
        )
        self.gaussians.scales = torch.nn.Parameter(
            self.gaussians.scales.detach()[keep_mask, :]
        )
        self.gaussians.opacities = torch.nn.Parameter(
            self.gaussians.opacities.detach()[keep_mask]
        )
        self.gaussians.rgb = torch.nn.Parameter(
            self.gaussians.rgb.detach()[keep_mask, :]
        )
        self.uv_grad_accum = self.uv_grad_accum[keep_mask, :]
        self.xyz_grad_accum = self.xyz_grad_accum[keep_mask, :]
        self.grad_accum_count = self.grad_accum_count[keep_mask]
        print("Deleted: ", torch.sum(~keep_mask).detach().cpu().numpy())

        # remove deleted gaussians from optimizer
        self.delete_gaussians_from_optimizer(keep_mask)

    def adaptive_density_control_update_adam(self, iter_num):
        if not (USE_DELETE or USE_CLONE or USE_SPLIT):
            return
        print("adaptive_density control update")
        self.check_nans()

        zero_view_mask = self.grad_accum_count == 0
        zero_grad_mask = torch.norm(self.uv_grad_accum, dim=1) == 0.0
        print("zero view mask: ", torch.sum(zero_view_mask).detach().cpu().numpy())
        print("zero grad mask: ", torch.sum(zero_grad_mask).detach().cpu().numpy())

        # low opacity
        keep_mask = self.gaussians.opacities > inverse_sigmoid(DELETE_OPACITY_THRESHOLD)
        keep_mask = keep_mask.squeeze(1)
        keep_mask &= ~zero_view_mask
        keep_mask &= ~zero_grad_mask

        # delete
        delete_count = torch.sum(~keep_mask).detach().cpu().numpy()
        if (delete_count > 0) and USE_DELETE:
            self.delete_gaussians(keep_mask)

        uv_grad_avg = self.uv_grad_accum / self.grad_accum_count.unsqueeze(1).float()
        xyz_grad_avg = self.xyz_grad_accum / self.grad_accum_count.unsqueeze(1).float()
        
        uv_grad_avg_norm = torch.norm(uv_grad_avg, dim=1)
        uv_split_val = torch.quantile(uv_grad_avg_norm, 0.95).item()
        densify_mask = uv_grad_avg_norm > uv_split_val
        print("Density mask: ", torch.sum(densify_mask).detach().cpu().numpy(), "split_val", uv_split_val)

        if densify_mask.any() and USE_CLONE or USE_SPLIT:
            xyz_list = []
            quaternions_list = []
            scales_list = []
            opacities_list = []
            rgb_list = []

            scale_norm = self.gaussians.scales.exp().norm(dim=-1)
            clone_mask = densify_mask & (
                scale_norm <= CLONE_SCALE_THRESHOLD
            )
            split_mask = densify_mask & (scale_norm > CLONE_SCALE_THRESHOLD) & (scale_norm <= MAX_SCALE_NORM)
            oversize_mask = densify_mask & (
                self.gaussians.scales.exp().norm(dim=-1) > MAX_SCALE_NORM
            )
            print(
                "Clone Mask: ",
                torch.sum(clone_mask).detach().cpu().numpy(),
                "Split Mask: ",
                torch.sum(split_mask).detach().cpu().numpy(),
                "Oversize Mask: ",
                torch.sum(oversize_mask).detach().cpu().numpy(),
            )

            if clone_mask.any() and USE_CLONE:
                # create cloned gaussians
                cloned_xyz = self.gaussians.xyz[clone_mask, :].clone().detach()
                cloned_xyz -= xyz_grad_avg[clone_mask, :].detach() * 0.01
                cloned_quaternions = (
                    self.gaussians.quaternions[clone_mask, :].clone().detach()
                )
                cloned_scales = self.gaussians.scales[clone_mask, :].clone().detach()
                cloned_opacities = self.gaussians.opacities[clone_mask].clone().detach()
                cloned_rgb = self.gaussians.rgb[clone_mask, :].clone().detach()

                xyz_list.append(cloned_xyz)
                quaternions_list.append(cloned_quaternions)
                scales_list.append(cloned_scales)
                opacities_list.append(cloned_opacities)
                rgb_list.append(cloned_rgb)
                print("\tCloned: ", torch.sum(clone_mask).detach().cpu().numpy())

            if split_mask.any() and USE_SPLIT:
                # create split gaussians
                split_quaternions = (
                    self.gaussians.quaternions[split_mask, :].clone().detach()
                )
                split_scales = self.gaussians.scales[split_mask, :].clone().detach()
                split_opacities = self.gaussians.opacities[split_mask].clone().detach()
                split_rgb = self.gaussians.rgb[split_mask, :].clone().detach()

                # compute sigma_world
                sigma_world = torch.zeros(
                    split_quaternions.shape[0],
                    3,
                    3,
                    device=split_quaternions.device,
                    dtype=split_quaternions.dtype,
                )
                compute_sigma_world_cuda(split_quaternions, split_scales, sigma_world)

                split_xyz_original = self.gaussians.xyz[split_mask, :].clone().detach()
                xyz_1, xyz_2 = sample_pdf(split_xyz_original, sigma_world)

                split_scales = split_scales - math.exp(1.6)

                # update original gaussian.xyz
                self.gaussians.xyz.detach()[split_mask] = xyz_1.detach()
                split_xyz = xyz_2.detach()

                xyz_list.append(split_xyz)
                quaternions_list.append(split_quaternions)
                scales_list.append(split_scales)
                opacities_list.append(split_opacities)
                rgb_list.append(split_rgb)
                print("\tSplit: ", torch.sum(split_mask).detach().cpu().numpy())

            if oversize_mask.any() and USE_SPLIT:
                # create split gaussians
                oversize_quaternions = (
                    self.gaussians.quaternions[oversize_mask, :].clone().detach()
                )
                oversize_scales = (
                    self.gaussians.scales[oversize_mask, :].clone().detach()
                )
                oversize_opacities = (
                    self.gaussians.opacities[oversize_mask].clone().detach()
                )
                oversize_rgb = self.gaussians.rgb[oversize_mask, :].clone().detach()

                # compute sigma_world
                sigma_world = torch.zeros(
                    oversize_quaternions.shape[0],
                    3,
                    3,
                    device=oversize_quaternions.device,
                    dtype=oversize_quaternions.dtype,
                )
                compute_sigma_world_cuda(
                    oversize_quaternions, oversize_scales, sigma_world
                )
                oversize_scales = oversize_scales - math.exp(5.0)

                oversize_xyz_original = (
                    self.gaussians.xyz[oversize_mask, :].clone().detach()
                )
                xyz_1, xyz_2 = sample_pdf(oversize_xyz_original, sigma_world)

                # update original gaussian.xyz
                self.gaussians.xyz.detach()[oversize_mask] = xyz_1.detach()
                oversize_xyz = xyz_2.detach()

                xyz_list.append(oversize_xyz)
                quaternions_list.append(oversize_quaternions)
                scales_list.append(oversize_scales)
                opacities_list.append(oversize_opacities)
                rgb_list.append(oversize_rgb)
                print("\tOversize: ", torch.sum(oversize_mask).detach().cpu().numpy())

            xyz_list.append(self.gaussians.xyz.clone().detach())
            quaternions_list.append(self.gaussians.quaternions.clone().detach())
            scales_list.append(self.gaussians.scales.clone().detach())
            opacities_list.append(self.gaussians.opacities.clone().detach())
            rgb_list.append(self.gaussians.rgb.clone().detach())

            # update gaussians
            self.gaussians.xyz = torch.nn.Parameter(torch.cat(xyz_list, dim=0))
            self.gaussians.quaternions = torch.nn.Parameter(
                torch.cat(quaternions_list, dim=0)
            )
            self.gaussians.scales = torch.nn.Parameter(torch.cat(scales_list, dim=0))
            self.gaussians.opacities = torch.nn.Parameter(
                torch.cat(opacities_list, dim=0)
            )
            self.gaussians.rgb = torch.nn.Parameter(torch.cat(rgb_list, dim=0))

            self.clone_gaussians_in_optimizer(densify_mask)

        self.reset_grad_accum()
        self.check_nans()
