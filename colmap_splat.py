import os
import torch
import time

import numpy as np
import plotext as plt

from splat_py.constants import *
from splat_py.dataloader import ColmapData
from splat_py.structs import SimpleTimer
from splat_py.trainer import GSTrainer


def plot_metrics(metrics):
    x = np.arange(len(metrics.train_psnr))
    train_psnr = np.array(metrics.train_psnr)
    num_gaussians = np.array(metrics.num_gaussians)

    # test psnr has different x-axis
    test_psnr = np.array(metrics.test_psnr)
    x_test = np.arange(len(test_psnr)) * TEST_EVAL_INTERVAL

    # smooth train psnr for better visualization
    smoothing_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1])
    smoothing_weights /= np.sum(smoothing_weights)
    train_psnr = np.convolve(train_psnr, smoothing_weights, mode="valid")

    plt.plot(x, train_psnr, xside="lower", yside="left", label="Train PSNR")
    plt.plot(x_test, test_psnr, xside="lower", yside="left", label="Test PSNR")
    plt.plot(x, num_gaussians, xside="upper", yside="right", label="Num Gaussians")

    plt.xlabel("Iteration")
    plt.ylabel("Train PSNR", yside="left")
    plt.ylabel("Num Gaussians", yside="right")

    plt.title("Gaussian Splatting")
    plt.show()


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

# save gaussians
torch.save(gaussians, os.path.join(OUTPUT_DIR, "gaussians.pt"))

# training time
seconds = end - start
minutes, seconds = divmod(seconds, 60)
print("Total training time: {}min {}sec".format(int(minutes), int(seconds)))
print("Max Test PSNR: ", max(trainer.metrics.test_psnr))
plot_metrics(trainer.metrics)
