from dataclasses import dataclass
import tyro
from typing import Literal
import yaml


class yamlEnabled(object):
    """
    Decorator to enable yaml serialization for a class.
    from: https://stackoverflow.com/questions/74723634/how-do-you-use-a-frozen-dataclass-in-a-dictionary-and-export-it-to-yaml
    """

    def __init__(self, tag):
        self.tag = tag

    def __call__(self, cls):
        def to_yaml(dumper, data):
            return dumper.represent_mapping(self.tag, vars(data))

        yaml.SafeDumper.add_representer(cls, to_yaml)

        def from_yaml(loader, node):
            data = loader.construct_mapping(node)
            return cls(**data)

        yaml.SafeLoader.add_constructor(self.tag, from_yaml)
        return cls


@yamlEnabled("!SplatConfig")
@dataclass
class SplatConfig:
    """Path to dataset directory"""

    dataset_path: str = "garden"
    """downsample factor for the images - if applicable"""
    downsample_factor: int = 4
    """output directory for saving the results"""
    output_dir: str = "splat_output"

    """interval for saving checkpoints"""
    checkpoint_interval: int = 10000
    """initialize gaussians from checkpoint"""
    load_checkpoint: bool = False
    """path to saved gaussian checkpoint"""
    checkpoint_path: str = ""

    """interval for saving debug training images"""
    save_debug_image_interval: int = 200
    """interval to print debug information"""
    print_interval: int = 100

    """initial opacity for gaussians initialized from a point cloud"""
    initial_opacity: float = 0.2
    """number of neighbors used to compute the initial scale"""
    initial_scale_num_neighbors: int = 3
    """factor to scale the distance to the nearest neighbors"""
    initial_scale_factor: float = 0.8
    """maximum initial scale"""
    max_initial_scale: float = 0.1

    """gaussians closer than this are culled alongside points outside of fov"""
    near_thresh: float = 0.3
    """mahalanobis distance for tile culling 3.0 = 99.7%"""
    mh_dist: float = 3.0
    """keep gaussians that project within this padding of image during frustrum culling"""
    cull_mask_padding: int = 100
    """max rgb value for splatted image"""
    saturated_pixel_value: float = 255.0

    """number of iterations for training"""
    num_iters: int = 7000
    """fraction of ssim loss to l1 loss"""
    ssim_frac: float = 0.2
    "base learning rate"
    base_lr: float = 0.002
    """learning rate multiplier for xyz"""
    xyz_lr_multiplier: float = 0.1
    """learning rate multiplier for quaternion"""
    quat_lr_multiplier: float = 2
    """learning rate multiplier for scale"""
    scale_lr_multiplier: float = 5
    """learning rate multiplier for opacity"""
    opacity_lr_multiplier: float = 10
    """learning rate multiplier for rgb"""
    rgb_lr_multiplier: float = 2
    """learning rate multiplier for spherical harmonics"""
    sh_lr_multiplier: float = 0.1

    """interval to evaluate test images"""
    test_eval_interval: int = 500
    """select every nth image for the test split - 8 is same as GS and Mip-Nerf 360 papers"""
    test_split_ratio: int = 8

    """use background color"""
    use_background: bool = True
    """background color end interval"""
    use_background_end: int = 6600

    """interval to reset all opacities to a fixed value"""
    reset_opacity_interval: int = 3001
    """opacity value to reset to"""
    reset_opacity_value: float = 0.20
    """start iteration for reset opacity"""
    reset_opacity_start: int = 1050
    """end iteration for reset opacity"""
    reset_opacity_end: int = 6500

    """precompute SH to RGB for each gaussian - speeds up computation ~1.4-2x"""
    use_sh_precompute: bool = True
    """max SH band to use - 0 is no view dependent color"""
    max_sh_band: Literal[0, 1, 2, 3] = 3
    """add SH band every interval until all are added"""
    add_sh_band_interval: int = 1000

    """use split gaussians"""
    use_split: bool = True
    """use clone gaussians"""
    use_clone: bool = True
    """use delete gaussians"""
    use_delete: bool = True

    """start iteration for adaptive control"""
    adaptive_control_start: int = 750
    """end iteration for adaptive control"""
    adaptive_control_end: int = 6500
    """interval for adaptive control"""
    adaptive_control_interval: int = 100

    """max number of gaussians"""
    max_gaussians: int = 3250000

    """delete gaussians with opacity below this threshold"""
    delete_opacity_threshold: float = 0.1
    """clone gaussians with scale below this threshold"""
    clone_scale_threshold: float = 0.01
    """delete gaussians with scale norm above this threshold"""
    max_scale_norm: float = 0.5
    """densify a fixed fraction of gaussians every iteration"""
    use_fractional_densification: bool = True
    """front load densification - slower but slightly higher psnr"""
    use_adaptive_fractional_densification: bool = True

    """densify gaussians over this percentile - only used if use_fractional_densification is True"""
    uv_grad_percentile: float = 0.96
    """densify gaussians over this percentile - only used if use_fractional_densification is True"""
    scale_norm_percentile: float = 0.99

    """densify gaussians over this threshold - only used if use_fractional_densification is False"""
    uv_grad_threshold: float = 0.0002

    """decrease scale of split gaussians by this factor"""
    split_scale_factor: float = 1.6
    """number of samples to split gaussians into"""
    num_split_samples: int = 2


# allow user to choose from 7k or 30k config as base configuration
SplatConfigs = tyro.extras.subcommand_type_from_defaults(
    {
        "7k": SplatConfig(),  # default config is 7k
        "30k": SplatConfig(
            num_iters=30000,
            adaptive_control_start=1500,
            adaptive_control_end=27500,
            adaptive_control_interval=300,
            reset_opacity_end=27500,
            use_background_end=28000,
        ),
    }
)
