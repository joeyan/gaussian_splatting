import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math

from splat_py.splat import splat
from splat_py.structs import Camera
from splat_py.constants import *
from splat_py.utils import inverse_sigmoid


def load_grads(path):
    grads = torch.load(path)
    uv_grad_accum = grads["uv_grad_accum"]
    xyz_grad_accum = grads["xyz_grad_accum"]
    grad_accum_count = grads["grad_accum_count"]

    return uv_grad_accum, xyz_grad_accum, grad_accum_count


def plot_uv_grad_hist(uv_grad_accum, grad_accum_count, max=None):
    num_bins = 100
    avg_uv_grad = uv_grad_accum.norm(dim=1) / grad_accum_count
    uv_grad_cpu = avg_uv_grad.cpu().numpy()

    if max is not None:
        uv_grad_cpu = uv_grad_cpu[uv_grad_cpu < max]
        print("Num uv grads above max: ", torch.sum(avg_uv_grad > max).item())

    plt.hist(uv_grad_cpu, bins=num_bins)
    plt.xlabel("Average UV Gradient")
    plt.ylabel("Frequency")
    plt.title("Histogram of Average UV Gradient Norm")
    plt.savefig("uv_grad_hist.png")
    plt.close()


def plot_xyz_grad_hist(xyz_grad_accum, grad_accum_count, max=None):
    num_bins = 100
    avg_xyz_grad = xyz_grad_accum.norm(dim=1) / grad_accum_count
    xyz_grad_cpu = avg_xyz_grad.cpu().numpy()

    if max is not None:
        xyz_grad_cpu = xyz_grad_cpu[xyz_grad_cpu < max]
        print("Num xyz grads above max: ", torch.sum(avg_xyz_grad > max).item())

    plt.hist(xyz_grad_cpu, bins=num_bins)
    plt.xlabel("Average XYZ Gradient")
    plt.ylabel("Frequency")
    plt.title("Histogram of Average XYZ Gradient Norm")
    plt.savefig("xyz_grad_hist.png")
    plt.close()


def plot_scale_hist(gaussians, max=None):
    num_bins = 100
    scale_norm = gaussians.scales.exp().norm(dim=1)
    scales_cpu = scale_norm.detach().cpu().numpy()

    if max is not None:
        scales_cpu = scales_cpu[scales_cpu < max]
        print("Num scales above max: ", torch.sum(scale_norm > max).item())

    plt.hist(scales_cpu, bins=num_bins)
    plt.xlabel("Scale")
    plt.ylabel("Frequency")
    plt.title("Histogram of Scale Norm")
    plt.savefig("scale_hist.png")
    plt.close()


def plot_opacity_hist(gaussians, max=None):
    num_bins = 100
    opacities_cpu = torch.sigmoid(gaussians.opacities).detach().cpu().numpy()

    if max is not None:
        opacities_cpu = opacities_cpu[opacities_cpu < max]
        print("Num opacities above max: ", torch.sum(gaussians.opacities > max).item())

    plt.hist(opacities_cpu, bins=num_bins)
    plt.xlabel("Opacity")
    plt.ylabel("Frequency")
    plt.title("Histogram of Opacity")
    plt.savefig("opacity_hist.png")
    plt.close()


iter_num = 2500
uv_grad_accum, xyz_grad_accum, grad_accum_count = load_grads(
    "splat_output/iter{}_adaptive_control_debug_data.pth".format(iter_num)
)
gaussians = torch.load(
    "splat_output/iter{}_adaptive_control_gaussians.pth".format(iter_num)
)


# filter by zero view count
zero_view_mask = grad_accum_count == 0
print("Num Points with zero view count", torch.sum(zero_view_mask).item())
zero_uv_grad_mask = torch.norm(uv_grad_accum, dim=1) == 0.0
print("Num Points with zero uv grad", torch.sum(zero_uv_grad_mask).item())

keep_mask = ~zero_view_mask & ~zero_uv_grad_mask

uv_grad_accum = uv_grad_accum[keep_mask, :]
xyz_grad_accum = xyz_grad_accum[keep_mask, :]
grad_accum_count = grad_accum_count[keep_mask]
gaussians.filter_in_place(keep_mask)


# plot uv_grad_hist
plot_uv_grad_hist(uv_grad_accum, grad_accum_count, 5e-6)
plot_xyz_grad_hist(xyz_grad_accum, grad_accum_count, 0.0006)
plot_scale_hist(gaussians, 0.001)
plot_opacity_hist(gaussians, 1.0)


avg_uv_grad = uv_grad_accum.norm(dim=1) / grad_accum_count
scale_norm = gaussians.scales.exp().norm(dim=1)

scale_split_val = torch.quantile(scale_norm, 0.95).item()
high_scale_mask = scale_norm > scale_split_val
print("scale_split_val: ", scale_split_val)

uv_split_val = torch.quantile(avg_uv_grad, 0.95).item()
densify_mask = avg_uv_grad > uv_split_val
print("uv_grad_split_val: ", uv_split_val)

filter_mask = high_scale_mask & densify_mask
gaussians.filter_in_place(filter_mask)
print("Num gaussians remaining: ", gaussians.xyz.shape[0])

with torch.no_grad():
    # make all filter points red and 90% opaque
    gaussians.rgb[filter_mask, :] = torch.tensor(
        [1.0, 0.0, 0.0], dtype=torch.float32, device=gaussians.rgb.device
    )
    opa = inverse_sigmoid(0.9)
    gaussians.opacities[filter_mask, :] = torch.tensor(
        [opa], dtype=torch.float32, device=gaussians.opacities.device
    )
    cam = Camera(1297, 840, torch.load("splat_output/camera_K.pth"))
    image_idx = 7
    world_T_image = torch.load("splat_output/world_T_image_{}.pth".format(image_idx))

    image, culling_mask = splat(gaussians, world_T_image, cam)
    print("Num Points after culling", torch.sum(culling_mask).item())
    debug_image = image.clip(0, 1).detach().cpu().numpy()
    cv2.imwrite(
        "debug_image_{}.png".format(image_idx),
        (debug_image * 255).astype(np.uint8)[..., ::-1],
    )
