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

        # exp_avg and exp_avg_sq are the same shape as the parameter
        # optimizer_param_state["exp_avg"] = torch.cat(
        #     [
        #         optimizer_param_state["exp_avg"],
        #         optimizer_param_state["exp_avg"][clone_mask, :].clone(),
        #     ],
        #     dim=0,
        # )
        # optimizer_param_state["exp_avg_sq"] = torch.cat(
        #     [
        #         optimizer_param_state["exp_avg_sq"],
        #         optimizer_param_state["exp_avg_sq"][clone_mask, :].clone(),
        #     ],
        #     dim=0,
        # )
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

    def adaptive_density_control_update_adam(self, iter_num):
        if not (USE_DELETE or USE_CLONE or USE_SPLIT):
            return
        print("adaptive_density control update")
        self.check_nans()

        finite_mask = torch.isfinite(self.uv_grad_accum).all(dim=1)

        # low opacity
        keep_mask = self.gaussians.opacities > inverse_sigmoid(DELETE_OPACITY_THRESHOLD)
        keep_mask = keep_mask.squeeze(1)
        keep_mask &= finite_mask

        delete_count = torch.sum(~keep_mask).detach().cpu().numpy()
        print("Delete Count: ", delete_count)

        if (delete_count > 0) and USE_DELETE:
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
            print("\tDeleted: ", delete_count)

            # remove deleted gaussians from optimizer
            self.delete_gaussians_from_optimizer(keep_mask)

        uv_grad_avg = self.uv_grad_accum / self.grad_accum_count.unsqueeze(1).float()
        xyz_grad_avg = self.xyz_grad_accum / self.grad_accum_count.unsqueeze(1).float()
        densify_mask = uv_grad_avg.norm(1) > DENSIFY_GRAD_THRESHOLD

        finite_mask = torch.isfinite(uv_grad_avg).all(dim=1)
        densify_mask = densify_mask & finite_mask

        if torch.any(~torch.isfinite(uv_grad_avg)):
            print("Num uv_grad_avg NaN", torch.sum(torch.isnan(uv_grad_avg)).item())
            print("Num uv_grad_avg inf", torch.sum(~torch.isfinite(uv_grad_avg)).item())

        if densify_mask.any() and USE_CLONE or USE_SPLIT:
            print("clone and split")
            xyz_list = []
            quaternions_list = []
            scales_list = []
            opacities_list = []
            rgb_list = []

            clone_mask = densify_mask & (
                self.gaussians.scales.exp().norm(dim=-1) > CLONE_SCALE_THRESHOLD
            )
            split_mask = densify_mask & (
                self.gaussians.scales.exp().norm(dim=-1) <= CLONE_SCALE_THRESHOLD
            )
            # too large
            oversize_mask = self.gaussians.scales.exp().norm(dim=-1) > MAX_SCALE_NORM
            print("Oversize: ", torch.sum(oversize_mask).detach().cpu().numpy())
            split_mask = split_mask | oversize_mask

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
                split_scales = self.gaussians.scales[
                    split_mask, :
                ].clone().detach() - math.exp(1.6)
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

                # update original gaussian.xyz
                self.gaussians.xyz.detach()[split_mask] = xyz_1.detach()
                split_xyz = xyz_2.detach()

                xyz_list.append(split_xyz)
                quaternions_list.append(split_quaternions)
                scales_list.append(split_scales)
                opacities_list.append(split_opacities)
                rgb_list.append(split_rgb)
                print("\tSplit: ", torch.sum(split_mask).detach().cpu().numpy())

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

    def adaptive_density_control(self, iter_num):
        if not (USE_DELETE or USE_CLONE or USE_SPLIT):
            return

        self.check_nans()

        finite_mask = torch.isfinite(self.uv_grad_accum).all(dim=1)

        # delete gaussians
        keep_mask = self.gaussians.opacities > inverse_sigmoid(DELETE_OPACITY_THRESHOLD)
        keep_mask = keep_mask.squeeze(1)
        keep_mask &= finite_mask

        delete_count = torch.sum(~keep_mask).detach().cpu().numpy()
        if (delete_count > 0) and USE_DELETE:
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
            print("\tDeleted: ", delete_count)

        uv_grad_avg = self.uv_grad_accum / self.grad_accum_count.unsqueeze(1).float()
        xyz_grad_avg = self.xyz_grad_accum / self.grad_accum_count.unsqueeze(1).float()
        densify_mask = uv_grad_avg.norm(1) > DENSIFY_GRAD_THRESHOLD

        finite_mask = torch.isfinite(uv_grad_avg).all(dim=1)
        densify_mask = densify_mask & finite_mask

        if torch.isnan(uv_grad_avg).any() or torch.any(~torch.isfinite(uv_grad_avg)):
            print(
                "Num uv_grad_avg inf", torch.sum(~torch.isfinite(uv_grad_avg)).detach()
            )
            print("Num uv_grad_avg NaN", torch.sum(torch.isnan(uv_grad_avg)).detach())

        if densify_mask.any() and USE_CLONE or USE_SPLIT:
            print("clone and split")
            xyz_list = []
            quaternions_list = []
            scales_list = []
            opacities_list = []
            rgb_list = []

            clone_mask = densify_mask & (
                self.gaussians.scales.exp().norm(dim=-1) > CLONE_SCALE_THRESHOLD
            )
            split_mask = densify_mask & (
                self.gaussians.scales.exp().norm(dim=-1) <= CLONE_SCALE_THRESHOLD
            )
            # too large
            oversize_mask = self.gaussians.scales.exp().norm(dim=-1) > MAX_SCALE_NORM
            print("Oversize: ", torch.sum(oversize_mask).detach().cpu().numpy())
            split_mask = split_mask | oversize_mask

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
                split_scales = self.gaussians.scales[
                    split_mask, :
                ].clone().detach() - math.exp(1.6)
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

                # update original gaussian.xyz
                self.gaussians.xyz.detach()[split_mask] = xyz_1.detach()
                split_xyz = xyz_2.detach()

                xyz_list.append(split_xyz)
                quaternions_list.append(split_quaternions)
                scales_list.append(split_scales)
                opacities_list.append(split_opacities)
                rgb_list.append(split_rgb)
                print("\tSplit: ", torch.sum(split_mask).detach().cpu().numpy())

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

        self.check_nans()

        # update optimizer
        self.update_optimizer(iter_num)
        self.reset_grad_accum()
