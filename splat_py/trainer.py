import math
import torch

from splat_py.constants import *
from splat_py.utils import inverse_sigmoid, sample_pdf
from splat_cuda import compute_sigma_world_cuda


class GSTrainer:
    def __init__(self, gaussians):
        self.gaussians = gaussians
        self.update_optimizer(iter_num=0)

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
        # reset grad accumulators
        self.pos_grad_accum = torch.zeros(
            self.gaussians.xyz.shape,
            dtype=self.gaussians.xyz.dtype,
            device=self.gaussians.xyz.device,
        )
        self.grad_accum_count = torch.zeros(
            self.gaussians.xyz.shape[0],
            dtype=torch.int,
            device=self.gaussians.xyz.device,
        )

    def adaptive_density_control(self, iter_num):
        if not (USE_DELETE or USE_CLONE or USE_SPLIT):
            return

        if torch.any(torch.isnan(self.gaussians.xyz)) or torch.any(
            ~torch.isfinite(self.gaussians.xyz)
        ):
            print("NaN or inf in xyz BEFORE densify")

        if torch.any(torch.isnan(self.gaussians.quaternions)) or torch.any(
            ~torch.isfinite(self.gaussians.quaternions)
        ):
            print("NaN or inf in quaternions BEFORE densify")

        if torch.any(torch.isnan(self.gaussians.scales)) or torch.any(
            ~torch.isfinite(self.gaussians.scales)
        ):
            print("NaN or inf in scales BEFORE densify")

        if torch.any(torch.isnan(self.gaussians.opacities)) or torch.any(
            ~torch.isfinite(self.gaussians.opacities)
        ):
            print("NaN or inf in opacities BEFORE densify")

        if torch.any(torch.isnan(self.gaussians.rgb)) or torch.any(
            ~torch.isfinite(self.gaussians.rgb)
        ):
            print("NaN or inf in rgb BEFORE densify")

        # delete gaussians
        keep_mask = torch.zeros(
            self.gaussians.opacities.shape[0],
            dtype=torch.bool,
            device=self.gaussians.opacities.device,
        )
        keep_mask = self.gaussians.opacities > inverse_sigmoid(DELETE_OPACITY_THRESHOLD)
        keep_mask = keep_mask.squeeze(1)

        if not keep_mask.any() and USE_DELETE:
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
            self.pos_grad_accum = self.pos_grad_accum[keep_mask, :]
            self.grad_accum_count = self.grad_accum_count[keep_mask]
            print("\tDeleted: ", torch.sum(~keep_mask).detach().cpu().numpy())

        grad_avg = self.pos_grad_accum / self.grad_accum_count.unsqueeze(1).float()
        densify_mask = grad_avg.max(1)[0] > DENSIFY_GRAD_THRESHOLD

        finite_mask = torch.isfinite(grad_avg).all(dim=1)
        densify_mask = densify_mask & finite_mask

        if torch.isnan(grad_avg).any() or torch.any(~torch.isfinite(grad_avg)):
            print("Num grad_avg inf", torch.sum(~torch.isfinite(grad_avg)).detach())
            print("Num grad_avg NaN", torch.sum(torch.isnan(grad_avg)).detach())

        if densify_mask.any() and USE_CLONE or USE_SPLIT:
            print("clone and split")
            xyz_list = []
            quaternions_list = []
            scales_list = []
            opacities_list = []
            rgb_list = []

            # clone or split gaussians
            clone_mask = torch.zeros(
                self.gaussians.opacities.shape[0],
                dtype=torch.bool,
                device=self.gaussians.opacities.device,
            )
            split_mask = torch.zeros(
                self.gaussians.opacities.shape[0],
                dtype=torch.bool,
                device=self.gaussians.opacities.device,
            )

            clone_mask = densify_mask & (
                self.gaussians.scales.exp().norm(dim=-1) > CLONE_SCALE_THRESHOLD
            )
            split_mask = densify_mask & (
                self.gaussians.scales.exp().norm(dim=-1) <= CLONE_SCALE_THRESHOLD
            )

            if clone_mask.any() and USE_CLONE:
                # create cloned gaussians
                cloned_xyz = self.gaussians.xyz[clone_mask, :].clone().detach()
                # cloned_xyz -= grad_avg[clone_mask, :].detach() * 0.01
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

        if torch.any(torch.isnan(self.gaussians.xyz)) or torch.any(
            ~torch.isfinite(self.gaussians.xyz)
        ):
            print("NaN or inf in xyz AFTER densify")
            print(
                "Num XYZ inf", torch.sum(~torch.isfinite(self.gaussians.xyz)).detach()
            )

        if torch.any(torch.isnan(self.gaussians.quaternions)) or torch.any(
            ~torch.isfinite(self.gaussians.quaternions)
        ):
            print("NaN or inf in quaternions AFTER densify")

        if torch.any(torch.isnan(self.gaussians.scales)) or torch.any(
            ~torch.isfinite(self.gaussians.scales)
        ):
            print("NaN or inf in scales AFTER densify")

        if torch.any(torch.isnan(self.gaussians.opacities)) or torch.any(
            ~torch.isfinite(self.gaussians.opacities)
        ):
            print("NaN or inf in opacities AFTER densify")

        if torch.any(torch.isnan(self.gaussians.rgb)) or torch.any(
            ~torch.isfinite(self.gaussians.rgb)
        ):
            print("NaN or inf in rgb AFTER densify")

        # update optimizer
        self.update_optimizer(iter_num)
