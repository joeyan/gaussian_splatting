import os
import torch
import time

import tyro
import numpy as np
import plotext as plt

from splat_py.config import SplatConfigs
from splat_py.dataloader import ColmapData
from splat_py.trainer import SplatTrainer


def plot_metrics(metrics, config):
    x = np.arange(len(metrics.train_psnr))
    train_psnr = np.array(metrics.train_psnr)
    num_gaussians = np.array(metrics.num_gaussians)

    # test psnr has different x-axis
    test_psnr = np.array(metrics.test_psnr)
    x_test = np.arange(len(test_psnr)) * config.test_eval_interval

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


config = tyro.cli(SplatConfigs)

if not os.path.exists(config.output_dir):
    os.makedirs(config.output_dir)

torch.manual_seed(0)
colmap_data = ColmapData(
    config.dataset_path,
    torch.device("cuda"),
    downsample_factor=config.downsample_factor,
    config=config,
)

if config.load_checkpoint:
    gaussians = torch.load(config.checkpoint_path)
else:
    gaussians = colmap_data.create_gaussians()
    gaussians.xyz = torch.nn.Parameter(gaussians.xyz)
    gaussians.quaternion = torch.nn.Parameter(gaussians.quaternion)
    gaussians.scale = torch.nn.Parameter(gaussians.scale)
    gaussians.opacity = torch.nn.Parameter(gaussians.opacity)
    gaussians.rgb = torch.nn.Parameter(gaussians.rgb)

images = colmap_data.get_images()
cameras = colmap_data.get_cameras()


start = time.time()
trainer = SplatTrainer(gaussians, images, cameras, config)
trainer.train()
end = time.time()

# save gaussians
torch.save(gaussians, os.path.join(config.output_dir, "gaussians.pt"))

# training time
seconds = end - start
minutes, seconds = divmod(seconds, 60)
print("Total training time: {}min {}sec".format(int(minutes), int(seconds)))
print("Max Test PSNR: ", max(trainer.metrics.test_psnr))
plot_metrics(trainer.metrics, config)
