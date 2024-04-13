import torch

from splat_py.constants import *


class OptimizerManager:
    """
    Manages adding/deleting gaussians and updating SH Bands
    """

    def __init__(self, gaussians):
        self.setup_optimizer(gaussians)

    def setup_optimizer(self, gaussians):
        # add new params to optimizer
        self.optimizer = torch.optim.Adam(
            [
                {"params": gaussians.xyz, "lr": BASE_LR * XYZ_LR_MULTIPLIER},
                {
                    "params": gaussians.quaternion,
                    "lr": BASE_LR * QUAT_LR_MULTIPLIER,
                },
                {"params": gaussians.scale, "lr": BASE_LR * SCALE_LR_MULTIPLIER},
                {
                    "params": gaussians.opacity,
                    "lr": BASE_LR * OPACITY_LR_MULTIPLIER,
                },
                {"params": gaussians.rgb, "lr": BASE_LR * RGB_LR_MULTIPLIER},
            ],
        )
        if gaussians.sh is not None:
            self.optimizer.add_param_group(
                {"params": gaussians.sh, "lr": BASE_LR * SH_LR_MULTIPLIER},
            )

    def reset_opacity_exp_avg(self, gaussians):
        # reset exp_avg and exp_avg_sq for opacity
        old_optimizer_param = self.optimizer.param_groups[3]["params"][0]
        optimizer_param_state = self.optimizer.state[old_optimizer_param]
        del self.optimizer.state[old_optimizer_param]

        optimizer_param_state["exp_avg"] = torch.zeros_like(optimizer_param_state["exp_avg"])
        optimizer_param_state["exp_avg_sq"] = torch.zeros_like(optimizer_param_state["exp_avg_sq"])

        del self.optimizer.param_groups[3]["params"][0]
        del self.optimizer.param_groups[3]["params"]

        self.optimizer.param_groups[3]["params"] = [gaussians.opacity]
        self.optimizer.state[3] = optimizer_param_state

    def add_sh_to_optimizer(self, gaussians):
        self.optimizer.add_param_group(
            {"params": gaussians.sh, "lr": BASE_LR * SH_LR_MULTIPLIER},
        )

    def add_sh_band_to_optimizer(self, gaussians):
        old_optimizer_param = self.optimizer.param_groups[5]["params"][0]
        optimizer_param_state = self.optimizer.state[old_optimizer_param]
        del self.optimizer.state[old_optimizer_param]

        optimizer_param_state["exp_avg"] = torch.zeros_like(gaussians.sh)
        optimizer_param_state["exp_avg_sq"] = torch.zeros_like(gaussians.sh)

        del self.optimizer.param_groups[5]["params"][0]
        del self.optimizer.param_groups[5]["params"]

        self.optimizer.param_groups[5]["params"] = [gaussians.sh]
        self.optimizer.state[5] = optimizer_param_state

    def delete_param_from_optimizer(self, new_param, keep_mask, param_index):
        old_optimizer_param = self.optimizer.param_groups[param_index]["params"][0]
        optimizer_param_state = self.optimizer.state[old_optimizer_param]
        del self.optimizer.state[old_optimizer_param]

        optimizer_param_state["exp_avg"] = optimizer_param_state["exp_avg"][keep_mask, :]
        optimizer_param_state["exp_avg_sq"] = optimizer_param_state["exp_avg_sq"][keep_mask, :]

        del self.optimizer.param_groups[param_index]["params"][0]
        del self.optimizer.param_groups[param_index]["params"]

        self.optimizer.param_groups[param_index]["params"] = [new_param]
        self.optimizer.state[new_param] = optimizer_param_state

    def delete_gaussians_from_optimizer(self, updated_gaussians, keep_mask):
        self.delete_param_from_optimizer(updated_gaussians.xyz, keep_mask, 0)
        self.delete_param_from_optimizer(updated_gaussians.quaternion, keep_mask, 1)
        self.delete_param_from_optimizer(updated_gaussians.scale, keep_mask, 2)
        self.delete_param_from_optimizer(updated_gaussians.opacity, keep_mask, 3)
        self.delete_param_from_optimizer(updated_gaussians.rgb, keep_mask, 4)
        if updated_gaussians.sh is not None:
            self.delete_param_from_optimizer(updated_gaussians.sh, keep_mask, 5)

    def add_params_to_optimizer(self, new_param, num_added, param_index):
        old_optimizer_param = self.optimizer.param_groups[param_index]["params"][0]
        optimizer_param_state = self.optimizer.state[old_optimizer_param]

        if new_param.dim() == 2:
            # set exp_avg and exp_avg_sq for cloned gaussians to zero
            optimizer_param_state["exp_avg"] = torch.cat(
                [
                    optimizer_param_state["exp_avg"],
                    torch.zeros(
                        num_added,
                        new_param.shape[1],
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
                        new_param.shape[1],
                        device=optimizer_param_state["exp_avg_sq"].device,
                        dtype=optimizer_param_state["exp_avg_sq"].dtype,
                    ),
                ],
                dim=0,
            )
        if new_param.dim() == 3:
            # set exp_avg and exp_avg_sq for cloned gaussians to zero
            optimizer_param_state["exp_avg"] = torch.cat(
                [
                    optimizer_param_state["exp_avg"],
                    torch.zeros(
                        num_added,
                        new_param.shape[1],
                        new_param.shape[2],
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
                        new_param.shape[1],
                        new_param.shape[2],
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

    def add_gaussians_to_optimizer(self, updated_gaussians, num_added):
        self.add_params_to_optimizer(updated_gaussians.xyz, num_added, 0)
        self.add_params_to_optimizer(updated_gaussians.quaternion, num_added, 1)
        self.add_params_to_optimizer(updated_gaussians.scale, num_added, 2)
        self.add_params_to_optimizer(updated_gaussians.opacity, num_added, 3)
        self.add_params_to_optimizer(updated_gaussians.rgb, num_added, 4)
        if updated_gaussians.sh is not None:
            self.add_params_to_optimizer(updated_gaussians.sh, num_added, 5)
