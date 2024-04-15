import cv2
import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from splat_py.optimizer_manager import OptimizerManager
from splat_py.structs import GSMetrics
from splat_py.rasterize import rasterize
from splat_py.utils import (
    inverse_sigmoid,
    quaternion_to_rotation_torch,
)


class SplatTrainer:
    def __init__(self, gaussians, images, cameras, config):
        self.gaussians = gaussians
        self.images = images
        self.cameras = cameras
        self.config = config

        self.optimizer_manager = OptimizerManager(gaussians, self.config)
        self.metrics = GSMetrics()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.gaussians.xyz.device)

        self.reset_grad_accum()
        self.setup_test_train_split()
        self.setup_test_images()

    def setup_test_train_split(self):
        num_images = len(self.images)
        all_images = np.arange(num_images)
        self.test_split = np.arange(0, num_images, self.config.test_split_ratio)
        self.train_split = np.array(list(set(all_images) - set(self.test_split)))

    def setup_test_images(self):
        for image_idx in range(len(self.images)):
            self.images[image_idx].image = (
                self.images[image_idx].image.to(torch.float32) / self.config.saturated_pixel_value
            )
            self.images[image_idx].image = self.images[image_idx].image.to(torch.device("cuda"))

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

    def reset_opacity(self):
        print("\t\tResetting opacity")
        self.gaussians.opacity = torch.nn.Parameter(
            torch.ones_like(self.gaussians.opacity)
            * inverse_sigmoid(self.config.reset_opacity_value)
        )
        self.optimizer_manager.reset_opacity_exp_avg(self.gaussians)
        self.reset_grad_accum()

    def add_sh_band(self):
        num_gaussians = self.gaussians.xyz.shape[0]

        if self.gaussians.sh is None and self.config.max_sh_band > 0:
            new_sh = torch.zeros(
                num_gaussians,
                3,
                3,
                dtype=self.gaussians.rgb.dtype,
                device=self.gaussians.rgb.device,
            )
            self.gaussians.sh = torch.nn.Parameter(new_sh)
            self.optimizer_manager.add_sh_to_optimizer(self.gaussians)
        elif self.gaussians.sh.shape[2] == 3 and self.config.max_sh_band > 1:
            new_sh = torch.zeros(
                num_gaussians,
                3,
                8,
                dtype=self.gaussians.rgb.dtype,
                device=self.gaussians.rgb.device,
            )
            new_sh[:, :, :3] = self.gaussians.sh
            self.gaussians.sh = torch.nn.Parameter(new_sh)
            self.optimizer_manager.add_sh_band_to_optimizer(self.gaussians)
        elif self.gaussians.sh.shape[2] == 8 and self.config.max_sh_band > 2:
            new_sh = torch.zeros(
                num_gaussians,
                3,
                15,
                dtype=self.gaussians.rgb.dtype,
                device=self.gaussians.rgb.device,
            )
            new_sh[:, :, :8] = self.gaussians.sh
            self.gaussians.sh = torch.nn.Parameter(new_sh)
            self.optimizer_manager.add_sh_band_to_optimizer(self.gaussians)

    def delete_gaussians(self, keep_mask):
        self.gaussians.filter_in_place(keep_mask)
        self.uv_grad_accum = self.uv_grad_accum[keep_mask, :]
        self.xyz_grad_accum = self.xyz_grad_accum[keep_mask, :]
        self.grad_accum_count = self.grad_accum_count[keep_mask]

        # remove deleted gaussians from optimizer
        self.optimizer_manager.delete_gaussians_from_optimizer(self.gaussians, keep_mask)

    def clone_gaussians(self, clone_mask, xyz_grad_avg):
        # create cloned gaussians
        cloned_xyz = self.gaussians.xyz[clone_mask, :].clone().detach()
        cloned_xyz -= xyz_grad_avg[clone_mask, :] * 0.01
        cloned_quaternion = self.gaussians.quaternion[clone_mask, :].clone().detach()
        cloned_scale = self.gaussians.scale[clone_mask, :].clone().detach()
        cloned_opacity = self.gaussians.opacity[clone_mask].clone().detach()
        cloned_rgb = self.gaussians.rgb[clone_mask, :].clone().detach()
        if self.gaussians.sh is not None:
            cloned_sh = self.gaussians.sh[clone_mask, :].clone().detach()

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
        if self.gaussians.sh is not None:
            self.gaussians.append(
                cloned_xyz,
                cloned_rgb,
                cloned_opacity,
                cloned_scale,
                cloned_quaternion,
                cloned_sh,
            )
        else:
            self.gaussians.append(
                cloned_xyz, cloned_rgb, cloned_opacity, cloned_scale, cloned_quaternion
            )
        self.optimizer_manager.add_gaussians_to_optimizer(
            self.gaussians, torch.sum(clone_mask).detach().cpu().numpy()
        )

    def split_gaussians(self, split_mask):
        samples = self.config.num_split_samples
        # create split gaussians
        split_quaternion = (
            self.gaussians.quaternion[split_mask, :].clone().detach().repeat(samples, 1)
        )
        split_scale = self.gaussians.scale[split_mask, :].clone().detach().repeat(samples, 1)
        split_opacity = self.gaussians.opacity[split_mask].clone().detach().repeat(samples, 1)
        split_rgb = self.gaussians.rgb[split_mask, :].clone().detach().repeat(samples, 1)
        if self.gaussians.sh is not None:
            split_sh = self.gaussians.sh[split_mask, :].clone().detach().repeat(samples, 1, 1)
        split_xyz = self.gaussians.xyz[split_mask, :].clone().detach().repeat(samples, 1)

        # centered random samples
        random_samples = torch.rand(split_mask.sum() * samples, 3, device=self.gaussians.xyz.device)
        # scale by scale factors
        scale_factors = torch.exp(split_scale)
        random_samples = random_samples * scale_factors
        # rotate by quaternion
        split_quaternion = split_quaternion / torch.norm(split_quaternion, dim=1, keepdim=True)
        split_rotations = quaternion_to_rotation_torch(split_quaternion)

        random_samples = torch.bmm(split_rotations, random_samples.unsqueeze(-1)).squeeze(-1)
        # translate by original mean locations
        split_xyz += random_samples

        # update scale
        split_scale = torch.log(torch.exp(split_scale) / self.config.split_scale_factor)

        # delete original split gaussians
        self.delete_gaussians(~split_mask)

        # add split gaussians
        if self.gaussians.sh is not None:
            self.gaussians.append(
                split_xyz, split_rgb, split_opacity, split_scale, split_quaternion, split_sh
            )
        else:
            self.gaussians.append(
                split_xyz, split_rgb, split_opacity, split_scale, split_quaternion
            )
        self.optimizer_manager.add_gaussians_to_optimizer(
            self.gaussians, torch.sum(split_mask).detach().cpu().numpy() * samples
        )

    def adaptive_density_control(self):
        if not (self.config.use_delete or self.config.use_clone or self.config.use_split):
            return
        print("Adaptive_density control update")

        # Step 1. Delete gaussians
        # low opacity
        keep_mask = self.gaussians.opacity > inverse_sigmoid(self.config.delete_opacity_threshold)
        keep_mask = keep_mask.squeeze(1)
        print("\tlow opacity mask: ", torch.sum(~keep_mask).detach().cpu().numpy())
        # no views or grad
        zero_view_mask = self.grad_accum_count == 0
        zero_grad_mask = torch.norm(self.uv_grad_accum, dim=1) == 0.0
        print("\tzero view mask: ", torch.sum(zero_view_mask).detach().cpu().numpy())
        print("\tzero grad mask: ", torch.sum(zero_grad_mask).detach().cpu().numpy())
        keep_mask &= ~zero_view_mask
        keep_mask &= ~zero_grad_mask

        delete_count = torch.sum(~keep_mask).detach().cpu().numpy()
        print("\tDeleting: ", delete_count)
        if (delete_count > 0) and self.config.use_delete:
            self.delete_gaussians(keep_mask)

        # Step 2. Densify gaussians
        uv_grad_avg = self.uv_grad_accum / self.grad_accum_count.unsqueeze(1).float()
        xyz_grad_avg = self.xyz_grad_accum / self.grad_accum_count.unsqueeze(1).float()

        uv_grad_avg_norm = torch.norm(uv_grad_avg, dim=1)

        if self.config.use_fractional_densification:
            uv_split_val = torch.quantile(uv_grad_avg_norm, self.config.uv_grad_percentile).item()
        else:
            uv_split_val = self.config.uv_grad_threshold
        densify_mask = uv_grad_avg_norm > uv_split_val
        print(
            "\tDensify mask: ",
            torch.sum(densify_mask).detach().cpu().numpy(),
            "split_val",
            uv_split_val,
        )

        scale_max = self.gaussians.scale.exp().max(dim=-1).values
        clone_mask = densify_mask & (scale_max <= self.config.clone_scale_threshold)
        print("\tClone Mask: ", torch.sum(clone_mask).detach().cpu().numpy())

        # Step 2.1 clone gaussians
        if clone_mask.any() and self.config.use_clone:
            self.clone_gaussians(clone_mask, xyz_grad_avg)
            # keep masks up to date
            densify_mask = torch.cat([densify_mask, densify_mask[clone_mask]], dim=0)
            scale_max = torch.cat([scale_max, scale_max[clone_mask]], dim=0)

        split_mask = densify_mask & (scale_max > self.config.clone_scale_threshold)

        scale_split = torch.quantile(scale_max, self.config.scale_norm_percentile).item()
        too_big_mask = scale_max > scale_split
        split_mask = split_mask | too_big_mask

        print("\tSplit Mask: ", torch.sum(split_mask).detach().cpu().numpy())
        # Step 2.2 split gaussians
        if split_mask.any() and self.config.use_split:
            self.split_gaussians(split_mask)

        self.reset_grad_accum()

    def compute_test_psnr(self, save_test_images=False, iter=0):
        with torch.no_grad():
            test_psnrs = []
            test_ssim = []
            for test_img_idx in self.test_split:
                test_world_T_image = self.images[test_img_idx].world_T_image
                test_camera = self.cameras[self.images[test_img_idx].camera_id]

                (
                    test_image,
                    _,
                    _,
                ) = rasterize(
                    self.gaussians,
                    test_world_T_image,
                    test_camera,
                    near_thresh=self.config.near_thresh,
                    cull_mask_padding=self.config.cull_mask_padding,
                    mh_dist=self.config.mh_dist,
                    use_sh_precompute=self.config.use_sh_precompute,
                )
                gt_image = self.images[test_img_idx].image
                l2_loss = torch.nn.functional.mse_loss(test_image.clip(0, 1), gt_image)
                psnr = -10 * torch.log10(l2_loss).item()

                ssim = self.ssim(
                    test_image.unsqueeze(0).permute(0, 3, 1, 2).clip(0, 1),
                    gt_image.unsqueeze(0).permute(0, 3, 1, 2),
                )

                test_psnrs.append(psnr)
                test_ssim.append(ssim.item())

                if save_test_images:
                    debug_image = test_image.clip(0, 1).detach().cpu().numpy()
                    cv2.imwrite(
                        "{}/iter{}_test_image_{}.png".format(
                            self.config.output_dir, iter, test_img_idx
                        ),
                        (debug_image * self.config.saturated_pixel_value).astype(np.uint8)[
                            ..., ::-1
                        ],
                    )

        return torch.tensor(test_psnrs), torch.tensor(test_ssim)

    def splat_and_compute_loss(self, image_idx, world_T_image, camera):
        image, culling_mask, uv = rasterize(
            self.gaussians,
            world_T_image,
            camera,
            near_thresh=self.config.near_thresh,
            cull_mask_padding=self.config.cull_mask_padding,
            mh_dist=self.config.mh_dist,
            use_sh_precompute=self.config.use_sh_precompute,
        )
        uv.retain_grad()

        gt_image = self.images[image_idx].image
        l1_loss = torch.nn.functional.l1_loss(image, gt_image)

        # for debug only
        l2_loss = torch.nn.functional.mse_loss(image, gt_image)
        psnr = -10 * torch.log10(l2_loss)

        # channel first tensor for SSIM
        ssim_loss = 1.0 - self.ssim(
            image.unsqueeze(0).permute(0, 3, 1, 2),
            gt_image.unsqueeze(0).permute(0, 3, 1, 2),
        )
        loss = (1.0 - self.config.ssim_ratio) * l1_loss + self.config.ssim_ratio * ssim_loss
        loss.backward()
        self.optimizer_manager.optimizer.step()

        # scale uv grad back to world coordinates - this way, uv grad is consistent across multiple cameras
        uv_grad = uv.grad.detach()
        uv_grad[:, 0] = uv_grad[:, 0] * camera.K[0, 0]
        uv_grad[:, 1] = uv_grad[:, 1] * camera.K[1, 1]

        self.uv_grad_accum[~culling_mask] += torch.abs(uv_grad)
        self.xyz_grad_accum += torch.abs(self.gaussians.xyz.grad.detach())
        self.grad_accum_count += (~culling_mask).int()

        return image, psnr

    def train(self):
        for i in range(self.config.num_iters):
            image_idx = np.random.choice(self.train_split)
            world_T_image = self.images[image_idx].world_T_image
            camera = self.cameras[self.images[image_idx].camera_id]

            self.optimizer_manager.optimizer.zero_grad()

            image, psnr = self.splat_and_compute_loss(image_idx, world_T_image, camera)
            self.metrics.train_psnr.append(psnr.item())
            self.metrics.num_gaussians.append(self.gaussians.xyz.shape[0])

            if i % self.config.print_interval == 0:
                print(
                    "Iter: {}, PSNR: {}, N: {}".format(
                        i, psnr.detach().cpu().numpy(), self.gaussians.xyz.shape[0]
                    )
                )

            # make sure to perform test split eval before adaptive control
            # if adaptive control occurs in the same iteration, test psnr will be low
            if i % self.config.test_eval_interval == 0:
                test_psnrs, test_ssims = self.compute_test_psnr()
                self.metrics.test_psnr.append(test_psnrs.mean().item())
                print(
                    "\t\t\t\t\t\tTEST SPLIT PSNR: {}, SSIM: {}".format(
                        test_psnrs.mean().item(), test_ssims.mean().item()
                    )
                )

            if (
                i > self.config.adaptive_control_start
                and i % self.config.adaptive_control_interval == 0
                and i < self.config.adaptive_control_end
            ):
                self.adaptive_density_control()

            if (
                i > self.config.reset_opacity_start
                and i < self.config.reset_opacity_end
                and i % self.config.reset_opacity_interval == 0
            ):
                self.reset_opacity()

            if self.config.use_sh_coeff and i > 0 and i % self.config.add_sh_band_interval == 0:
                self.add_sh_band()

            if i % self.config.save_debug_image_interval == 0:
                debug_image = image.clip(0, 1).detach().cpu().numpy()
                cv2.imwrite(
                    "{}/iter{}_image_{}.png".format(self.config.output_dir, i, image_idx),
                    (debug_image * self.config.saturated_pixel_value).astype(np.uint8)[..., ::-1],
                )
        final_psnrs, final_ssim = self.compute_test_psnr(save_test_images=True, iter=i)
        print(
            "Final PSNR: {}, SSIM: {}".format(final_psnrs.mean().item(), final_ssim.mean().item())
        )
