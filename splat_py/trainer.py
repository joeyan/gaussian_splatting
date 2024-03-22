import torch

from splat_py.constants import *
from splat_py.utils import inverse_sigmoid, quaternion_to_rotation_torch


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

    def add_params_to_optimizer(self, new_param, num_added, param_index):
        old_optimizer_param = self.optimizer.param_groups[param_index]["params"][0]
        optimizer_param_state = self.optimizer.state[old_optimizer_param]

        # set exp_avg and exp_avg_sq for cloned gaussians to zero
        optimizer_param_state["exp_avg"] = torch.cat(
            [
                optimizer_param_state["exp_avg"],
                torch.zeros(
                    num_added,
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
                    num_added,
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

    def add_gaussians_to_optimizer(self, clone_mask):
        self.add_params_to_optimizer(self.gaussians.xyz, clone_mask, 0)
        self.add_params_to_optimizer(self.gaussians.quaternions, clone_mask, 1)
        self.add_params_to_optimizer(self.gaussians.scales, clone_mask, 2)
        self.add_params_to_optimizer(self.gaussians.opacities, clone_mask, 3)
        self.add_params_to_optimizer(self.gaussians.rgb, clone_mask, 4)

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

        # remove deleted gaussians from optimizer
        self.delete_gaussians_from_optimizer(keep_mask)

    def adaptive_density_control_update_adam(self):
        if not (USE_DELETE or USE_CLONE or USE_SPLIT):
            return
        print("adaptive_density control update")
        self.check_nans()

        # Step 1. Delete gaussians
        # low opacity
        keep_mask = self.gaussians.opacities > inverse_sigmoid(DELETE_OPACITY_THRESHOLD)
        keep_mask = keep_mask.squeeze(1)
        print("low opacity mask: ", torch.sum(~keep_mask).detach().cpu().numpy())
        # no views or grad
        zero_view_mask = self.grad_accum_count == 0
        zero_grad_mask = torch.norm(self.uv_grad_accum, dim=1) == 0.0
        print("zero view mask: ", torch.sum(zero_view_mask).detach().cpu().numpy())
        print("zero grad mask: ", torch.sum(zero_grad_mask).detach().cpu().numpy())
        keep_mask &= ~zero_view_mask
        keep_mask &= ~zero_grad_mask

        # # too large
        # too_big_mask = self.gaussians.scales.exp().max(dim=1).values > MAX_SCALE_NORM
        # print("too big mask: ", torch.sum(too_big_mask).detach().cpu().numpy())
        # keep_mask &= ~too_big_mask

        delete_count = torch.sum(~keep_mask).detach().cpu().numpy()
        print("Deleting: ", delete_count)
        if (delete_count > 0) and USE_DELETE:
            self.delete_gaussians(keep_mask)

        # Step 2. Densify gaussians
        uv_grad_avg = self.uv_grad_accum / self.grad_accum_count.unsqueeze(1).float()
        xyz_grad_avg = self.xyz_grad_accum / self.grad_accum_count.unsqueeze(1).float()

        uv_grad_avg_norm = torch.norm(uv_grad_avg, dim=1)

        uv_split_val = torch.quantile(uv_grad_avg_norm, UV_GRAD_PERCENTILE).item()
        densify_mask = uv_grad_avg_norm > uv_split_val
        print(
            "Densify mask: ",
            torch.sum(densify_mask).detach().cpu().numpy(),
            "split_val",
            uv_split_val,
        )

        # densify_mask = uv_grad_avg_norm > DENSIFY_GRAD_THRESHOLD
        # print("Densify mask: ", torch.sum(densify_mask).detach().cpu().numpy(), "split_val", DENSIFY_GRAD_THRESHOLD)

        scale_max = self.gaussians.scales.exp().max(dim=-1).values
        clone_mask = densify_mask & (scale_max <= CLONE_SCALE_THRESHOLD)
        print("Clone Mask: ", torch.sum(clone_mask).detach().cpu().numpy())

        if clone_mask.any() and USE_CLONE:
            # create cloned gaussians
            cloned_xyz = self.gaussians.xyz[clone_mask, :].clone().detach()
            cloned_xyz -= xyz_grad_avg[clone_mask, :] * 0.01
            cloned_quaternions = (
                self.gaussians.quaternions[clone_mask, :].clone().detach()
            )
            cloned_scales = self.gaussians.scales[clone_mask, :].clone().detach()
            cloned_opacities = self.gaussians.opacities[clone_mask].clone().detach()
            cloned_rgb = self.gaussians.rgb[clone_mask, :].clone().detach()

            # keep masks up to date
            densify_mask = torch.cat([densify_mask, densify_mask[clone_mask]], dim=0)
            scale_max = torch.cat([scale_max, scale_max[clone_mask]], dim=0)

            # keep grads up to date
            self.uv_grad_accum = torch.cat(
                [self.uv_grad_accum, self.uv_grad_accum[clone_mask, :]], dim=0
            )
            self.xyz_grad_accum = torch.cat(
                [self.xyz_grad_accum, self.xyz_grad_accum[clone_mask, :]], dim=0
            )
            self.grad_accum_count = torch.cat(
                [self.grad_accum_count, self.grad_accum_count[clone_mask]], dim=0
            )

            # clone gaussians
            self.gaussians.xyz = torch.nn.Parameter(
                torch.cat([self.gaussians.xyz, cloned_xyz], dim=0)
            )
            self.gaussians.quaternions = torch.nn.Parameter(
                torch.cat([self.gaussians.quaternions, cloned_quaternions], dim=0)
            )
            self.gaussians.scales = torch.nn.Parameter(
                torch.cat([self.gaussians.scales, cloned_scales], dim=0)
            )
            self.gaussians.opacities = torch.nn.Parameter(
                torch.cat([self.gaussians.opacities, cloned_opacities], dim=0)
            )
            self.gaussians.rgb = torch.nn.Parameter(
                torch.cat([self.gaussians.rgb, cloned_rgb], dim=0)
            )
            self.add_gaussians_to_optimizer(
                torch.sum(clone_mask).detach().cpu().numpy()
            )

        split_mask = densify_mask & (scale_max > CLONE_SCALE_THRESHOLD)

        scale_split = torch.quantile(scale_max, SCALE_NORM_PERCENTILE).item()
        too_big_mask = scale_max > scale_split
        split_mask = split_mask | too_big_mask

        print("Split Mask: ", torch.sum(split_mask).detach().cpu().numpy())

        if split_mask.any() and USE_SPLIT:
            samples = NUM_SPLIT_SAMPLES
            # create split gaussians
            split_quaternions = (
                self.gaussians.quaternions[split_mask, :]
                .clone()
                .detach()
                .repeat(samples, 1)
            )
            split_scales = (
                self.gaussians.scales[split_mask, :].clone().detach().repeat(samples, 1)
            )
            split_opacities = (
                self.gaussians.opacities[split_mask].clone().detach().repeat(samples, 1)
            )
            split_rgb = (
                self.gaussians.rgb[split_mask, :].clone().detach().repeat(samples, 1)
            )
            split_xyz = (
                self.gaussians.xyz[split_mask, :].clone().detach().repeat(samples, 1)
            )

            # centered random samples
            random_samples = torch.rand(
                split_mask.sum() * samples, 3, device=self.gaussians.xyz.device
            )
            # scale by scale factors
            scale_factors = torch.exp(split_scales)
            random_samples = random_samples * scale_factors
            # rotate by quaternions
            split_quaternions = split_quaternions / torch.norm(
                split_quaternions, dim=1, keepdim=True
            )
            split_rotations = quaternion_to_rotation_torch(split_quaternions)

            random_samples = torch.bmm(
                split_rotations, random_samples.unsqueeze(-1)
            ).squeeze(-1)
            # translate by original mean locations
            split_xyz += random_samples

            # update scales
            split_scales = torch.log(torch.exp(split_scales) / SPLIT_SCALE_FACTOR)

            # delete original split gaussians
            self.delete_gaussians(~split_mask)

            # add split gaussians
            self.gaussians.xyz = torch.nn.Parameter(
                torch.cat([self.gaussians.xyz, split_xyz], dim=0)
            )
            self.gaussians.quaternions = torch.nn.Parameter(
                torch.cat([self.gaussians.quaternions, split_quaternions], dim=0)
            )
            self.gaussians.scales = torch.nn.Parameter(
                torch.cat([self.gaussians.scales, split_scales], dim=0)
            )
            self.gaussians.opacities = torch.nn.Parameter(
                torch.cat([self.gaussians.opacities, split_opacities], dim=0)
            )
            self.gaussians.rgb = torch.nn.Parameter(
                torch.cat([self.gaussians.rgb, split_rgb], dim=0)
            )
            self.add_gaussians_to_optimizer(
                torch.sum(split_mask).detach().cpu().numpy() * samples
            )

        self.reset_grad_accum()
        self.check_nans()
