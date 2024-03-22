import os
import torch

from splat_py.constants import *
from splat_py.dataloader import ColmapData
from splat_py.structs import SimpleTimer

from splat_py.trainer import GSTrainer

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

torch.manual_seed(0)
with SimpleTimer("Load Colmap Data"):
    colmap_data = ColmapData(DATASET_PATH, torch.device("cuda"), downsample_factor=4)
    gaussians = colmap_data.create_gaussians()

    images = colmap_data.get_images()
    cameras = colmap_data.get_cameras()

    gaussians.xyz = torch.nn.Parameter(gaussians.xyz)
    gaussians.quaternions = torch.nn.Parameter(gaussians.quaternions)
    gaussians.scales = torch.nn.Parameter(gaussians.scales)
    gaussians.opacities = torch.nn.Parameter(gaussians.opacities)
    gaussians.rgb = torch.nn.Parameter(gaussians.rgb)

trainer = GSTrainer(gaussians, images, cameras)
trainer.train()
