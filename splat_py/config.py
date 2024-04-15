from dataclasses import dataclass


@dataclass
class SplatConfig:
    # DATA OPTIONS
    dataset_path: str = "/home/joe/Downloads/garden"
    downsample_factor: int = 4
    output_dir: str = "splat_output"

    # CHECKPOINT OPTIONS
    checkpoint_interval: int = 10000
    load_checkpoint: bool = False
    checkpoint_path: str = ""

    # DEBUG OPTIONS
    save_debug_image_interval: int = 200
    print_interval: int = 100
    print_debug_timing: bool = False

    # INITIALIZATION OPTIONS
    initial_opacity: float = 0.2
    initial_scale_num_neighbors: int = 3
    initial_scale_factor: float = 0.8
    max_initial_scale: float = 0.1

    # SPLATTING OPTIONS
    near_thresh: float = 0.3
    mh_dist: float = 3.0
    cull_mask_padding: int = 100
    saturated_pixel_value: float = 255.0

    # TRAINING OPTIONS
    # 1. number of iterations
    num_iters: int = 7000
    # 2. SSIM ratio
    ssim_ratio: float = 0.2
    # 3. learning rates
    base_lr: float = 0.002
    xyz_lr_multiplier: float = 0.1
    quat_lr_multiplier: float = 2
    scale_lr_multiplier: float = 5
    opacity_lr_multiplier: float = 10
    rgb_lr_multiplier: float = 2
    sh_lr_multiplier: float = 0.1

    # TEST OPTIONS
    test_eval_interval: int = 500
    test_split_ratio: int = 8

    # RESET OPACITY OPTIONS
    reset_opacity_interval: int = 3001
    reset_opacity_value: float = 0.15
    reset_opacity_start: int = 1050
    reset_opacity_end: int = 6500

    # SPHERICAL HARMONICS OPTIONS
    use_sh_coeff: bool = True
    use_sh_precompute: bool = True
    max_sh_band: int = 3
    add_sh_band_interval: int = 1000

    # ADAPTIVE CONTROL OPTIONS
    # 1. which adaptive control options are enabled
    use_split: bool = True
    use_clone: bool = True
    use_delete: bool = True

    # 2. adaptive control intervals
    adaptive_control_start: int = 750
    adaptive_control_end: int = 6500
    adaptive_control_interval: int = 100

    # 3. thresholds for adaptive control
    delete_opacity_threshold: float = 0.1
    clone_scale_threshold: float = 0.01
    max_scale_norm: float = 0.5
    use_fractional_densification: bool = True

    # these are used if use_fractional_densification is True
    uv_grad_percentile: float = 0.96
    scale_norm_percentile: float = 0.99

    # these are used if use_fractional_densification is False
    uv_grad_threshold: float = 0.0002

    # split options
    split_scale_factor: float = 1.6
    num_split_samples: int = 2
