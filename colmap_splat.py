import os
import torch
import time
import plotext as plt

from splat_py.constants import *
from splat_py.dataloader import ColmapData
from splat_py.structs import SimpleTimer
from splat_py.trainer import GSTrainer

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

torch.manual_seed(0)
with SimpleTimer("Load Colmap Data"):
    colmap_data = ColmapData(
        DATASET_PATH, torch.device("cuda"), downsample_factor=DOWNSAMPLE_FACTOR
    )
    gaussians = colmap_data.create_gaussians()

    images = colmap_data.get_images()
    cameras = colmap_data.get_cameras()

    gaussians.xyz = torch.nn.Parameter(gaussians.xyz)
    gaussians.quaternions = torch.nn.Parameter(gaussians.quaternions)
    gaussians.scales = torch.nn.Parameter(gaussians.scales)
    gaussians.opacities = torch.nn.Parameter(gaussians.opacities)
    gaussians.rgb = torch.nn.Parameter(gaussians.rgb)

start = time.time()
trainer = GSTrainer(gaussians, images, cameras)
trainer.train()
end = time.time()
print("Total training time: ", end - start)

import plotext as plt
import numpy as np

train_psnr = np.array(trainer.metrics.train_psnr)
num_gaussians = np.array(trainer.metrics.num_gaussians)

plt.plot(train_psnr, xside="lower", yside="left", label="Train PSNR")
plt.plot(num_gaussians, xside="upper", yside="right", label="Num Gaussians")

plt.xlabel("Iteration")
plt.ylabel("Train PSNR", yside="left")
plt.ylabel("Num Gaussians", yside="right")

plt.title("Gaussian Splatting")
plt.show()
