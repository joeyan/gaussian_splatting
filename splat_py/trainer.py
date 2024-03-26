import cv2
import numpy as np
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from splat_py.constants import *
from splat_py.cuda_autograd_functions import (
    CameraPointProjection,
    ComputeSigmaWorld,
    ComputeProjectionJacobian,
    ComputeSigmaImage,
    RenderImage,
)
from splat_py.structs import Gaussians, Tiles, SimpleTimer
from splat_py.splat import splat
from splat_py.tile_culling import (
    match_gaussians_to_tiles_gpu,
    sort_gaussians,
)
from splat_py.utils import (
    inverse_sigmoid,
    quaternion_to_rotation_torch,
    transform_points_torch,
)


class GSTrainer:
    def __init__(self, gaussians, images, cameras):
        self.gaussians = gaussians
        self.images = images
        self.cameras = cameras

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(
            self.gaussians.xyz.device
        )

        self.update_optimizer(iter_num=0)
        self.reset_grad_accum()
        self.setup_test_train_split()

    def setup_test_train_split(self):
        num_images = len(self.images)
        all_images = np.arange(num_images)
        self.test_split = np.arange(0, num_images, TEST_SPLIT_RATIO)
        self.train_split = np.array(list(set(all_images) - set(self.test_split)))

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
        self.gaussians.filter_in_place(keep_mask)
        self.uv_grad_accum = self.uv_grad_accum[keep_mask, :]
        self.xyz_grad_accum = self.xyz_grad_accum[keep_mask, :]
        self.grad_accum_count = self.grad_accum_count[keep_mask]

        # remove deleted gaussians from optimizer
        self.delete_gaussians_from_optimizer(keep_mask)

    def clone_gaussians(self, clone_mask, xyz_grad_avg):
        # create cloned gaussians
        cloned_xyz = self.gaussians.xyz[clone_mask, :].clone().detach()
        cloned_xyz -= xyz_grad_avg[clone_mask, :] * 0.01
        cloned_quaternions = self.gaussians.quaternions[clone_mask, :].clone().detach()
        cloned_scales = self.gaussians.scales[clone_mask, :].clone().detach()
        cloned_opacities = self.gaussians.opacities[clone_mask].clone().detach()
        cloned_rgb = self.gaussians.rgb[clone_mask, :].clone().detach()

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
        self.gaussians.append(
            cloned_xyz, cloned_rgb, cloned_opacities, cloned_scales, cloned_quaternions
        )
        self.add_gaussians_to_optimizer(torch.sum(clone_mask).detach().cpu().numpy())

    def split_gaussians(self, split_mask):
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
        self.gaussians.append(
            split_xyz, split_rgb, split_opacities, split_scales, split_quaternions
        )
        self.add_gaussians_to_optimizer(
            torch.sum(split_mask).detach().cpu().numpy() * samples
        )

    def adaptive_density_control(self):
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

        scale_max = self.gaussians.scales.exp().max(dim=-1).values
        clone_mask = densify_mask & (scale_max <= CLONE_SCALE_THRESHOLD)
        print("Clone Mask: ", torch.sum(clone_mask).detach().cpu().numpy())

        # Step 2.1 clone gaussians
        if clone_mask.any() and USE_CLONE:
            self.clone_gaussians(clone_mask, xyz_grad_avg)
            # keep masks up to date
            densify_mask = torch.cat([densify_mask, densify_mask[clone_mask]], dim=0)
            scale_max = torch.cat([scale_max, scale_max[clone_mask]], dim=0)

        split_mask = densify_mask & (scale_max > CLONE_SCALE_THRESHOLD)

        scale_split = torch.quantile(scale_max, SCALE_NORM_PERCENTILE).item()
        too_big_mask = scale_max > scale_split
        split_mask = split_mask | too_big_mask

        print("Split Mask: ", torch.sum(split_mask).detach().cpu().numpy())
        # Step 2.2 split gaussians
        if split_mask.any() and USE_SPLIT:
            self.split_gaussians(split_mask)

        self.reset_grad_accum()
        self.check_nans()

    def compute_test_psnr(self):
        with torch.no_grad():
            test_psnrs = []
            for test_img_idx in self.test_split:
                test_world_T_image = self.images[test_img_idx].world_T_image
                test_camera = self.cameras[self.images[test_img_idx].camera_id]

                test_image, _ = splat(self.gaussians, test_world_T_image, test_camera)
                gt_image = (
                    self.images[test_img_idx].image.to(torch.float32)
                    / SATURATED_PIXEL_VALUE
                )
                gt_image = gt_image.to(torch.device("cuda"))
                l2_loss = torch.nn.functional.mse_loss(test_image, gt_image)
                psnr = -10 * torch.log10(l2_loss).item()
                test_psnrs.append(psnr)
        return torch.tensor(test_psnrs)

    def splat_and_compute_loss(self, image_idx, world_T_image, camera):
        with SimpleTimer("Splat {} Gaussians".format(self.gaussians.xyz.shape[0])):
            xyz_camera_frame = transform_points_torch(self.gaussians.xyz, world_T_image)
            uv = CameraPointProjection.apply(xyz_camera_frame, camera.K)
            uv.retain_grad()

            # perform frustrum culling
            culling_mask = torch.zeros(
                xyz_camera_frame.shape[0],
                dtype=torch.bool,
                device=self.gaussians.xyz.device,
            )
            culling_mask = culling_mask | (xyz_camera_frame[:, 2] < NEAR_THRESH)
            culling_mask = (
                culling_mask
                | (uv[:, 0] < -1 * CULL_MASK_PADDING)
                | (uv[:, 0] > camera.width + CULL_MASK_PADDING)
                | (uv[:, 1] < -1 * CULL_MASK_PADDING)
                | (uv[:, 1] > camera.height + CULL_MASK_PADDING)
            )

            # cull gaussians outside of camera frustrum
            culled_uv = uv[~culling_mask, :]
            culled_xyz_camera_frame = xyz_camera_frame[~culling_mask, :]

            culled_gaussians = Gaussians(
                xyz=self.gaussians.xyz[~culling_mask, :],
                quaternions=self.gaussians.quaternions[~culling_mask, :],
                scales=self.gaussians.scales[~culling_mask, :],
                opacities=torch.sigmoid(
                    self.gaussians.opacities[~culling_mask]
                ),  # apply sigmoid activation to opacities
                rgb=torch.sigmoid(
                    self.gaussians.rgb[~culling_mask, :]
                ),  # apply sigmoid activation to rgb
            )
            sigma_world = ComputeSigmaWorld.apply(
                culled_gaussians.quaternions, culled_gaussians.scales
            )
            J = ComputeProjectionJacobian.apply(culled_xyz_camera_frame, camera.K)
            sigma_image = ComputeSigmaImage.apply(sigma_world, J, world_T_image)

            # perform tile culling
            tiles = Tiles(camera.height, camera.width, culled_uv.device)
            (
                gaussian_idx_by_splat_idx,
                splat_start_end_idx_by_tile_idx,
                tile_idx_by_splat_idx,
            ) = match_gaussians_to_tiles_gpu(
                culled_uv, tiles, sigma_image, mh_dist=MH_DIST
            )

            sorted_gaussian_idx_by_splat_idx = sort_gaussians(
                culled_xyz_camera_frame,
                gaussian_idx_by_splat_idx,
                tile_idx_by_splat_idx,
            )
            image = RenderImage.apply(
                culled_gaussians.rgb,
                culled_gaussians.opacities,
                culled_uv,
                sigma_image,
                splat_start_end_idx_by_tile_idx,
                sorted_gaussian_idx_by_splat_idx,
                torch.tensor([camera.height, camera.width], device=culled_uv.device),
            )
        gt_image = (
            self.images[image_idx].image.to(torch.float32) / SATURATED_PIXEL_VALUE
        )
        gt_image = gt_image.to(torch.device("cuda"))

        l1_loss = torch.nn.functional.l1_loss(image, gt_image)

        # print PSNR
        l2_loss = torch.nn.functional.mse_loss(image, gt_image)
        psnr = -10 * torch.log10(l2_loss)

        # channel first tensor for SSIM
        ssim_loss = 1.0 - self.ssim(
            image.unsqueeze(0).permute(0, 3, 1, 2),
            gt_image.unsqueeze(0).permute(0, 3, 1, 2),
        )
        loss = (1.0 - SSIM_RATIO) * l1_loss + SSIM_RATIO * ssim_loss
        with SimpleTimer("Backward"):
            loss.backward()
        with SimpleTimer("Optimizer Step"):
            self.optimizer.step()

        self.uv_grad_accum += torch.abs(uv.grad.detach())
        self.xyz_grad_accum += torch.abs(self.gaussians.xyz.grad.detach())
        self.grad_accum_count += (~culling_mask).int()

        return image, psnr

    def train(self):
        for i in range(NUM_ITERS):
            image_idx = np.random.choice(self.train_split)
            world_T_image = self.images[image_idx].world_T_image
            camera = self.cameras[self.images[image_idx].camera_id]

            self.optimizer.zero_grad()

            image, psnr = self.splat_and_compute_loss(image_idx, world_T_image, camera)

            if i % PRINT_INTERVAL == 0:
                print(
                    "Iter: {}, PSNR: {}, N: {}".format(
                        i, psnr.detach().cpu().numpy(), self.gaussians.xyz.shape[0]
                    )
                )

            # make sure to perform test split eval before adaptive control
            # if adaptive control occurs in the same iteration, test psnr will be low
            if i % TEST_EVAL_INTERVAL == 0:
                test_psnrs = self.compute_test_psnr()
                print("\tTEST SPLIT PSNR: ", test_psnrs.mean().item())

            if (
                i > ADAPTIVE_CONTROL_START
                and i % ADAPTIVE_CONTROL_INVERVAL == 0
                and i < ADAPTIVE_CONTROL_END
            ):
                self.adaptive_density_control()

            if (
                i > RESET_OPACTIY_START
                and i < RESET_OPACITY_END
                and i % RESET_OPACITY_INTERVAL == 0
            ):
                self.reset_opacities(i)

            if i % SAVE_DEBUG_IMAGE_INTERVAL == 0:
                with SimpleTimer("Save Images"):
                    debug_image = image.clip(0, 1).detach().cpu().numpy()
                    cv2.imwrite(
                        "{}/iter{}_image_{}.png".format(OUTPUT_DIR, i, image_idx),
                        (debug_image * SATURATED_PIXEL_VALUE).astype(np.uint8)[
                            ..., ::-1
                        ],
                    )
        final_psnrs = self.compute_test_psnr()
        print("Final PSNR: ", final_psnrs.mean().item())
