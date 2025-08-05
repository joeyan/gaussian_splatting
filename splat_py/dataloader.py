import os
import cv2
import torch

from splat_py.config import SplatConfig
from splat_py.read_colmap import (
    read_images_binary,
    read_points3D_binary,
    read_cameras_binary,
    qvec2rotmat,
)
from splat_py.utils import inverse_sigmoid, compute_initial_scale_from_sparse_points
from splat_py.structs import Gaussians, Image, Camera


class GaussianSplattingDataset:
    """
    Generic Gaussian Splatting Dataset class

    Classes that inherit from this class should have the following variables:

    device: torch device
    xyz: Nx3 tensor of points
    rgb: Nx3 tensor of rgb values

    images: list of Image objects
    cameras: dict of Camera objects

    """

    def __init__(self, config):
        self.config = config

    def verify_loaded_points(self):
        """
        Verify that the values loaded from the dataset are consistent
        """
        N = self.xyz.shape[0]
        assert self.xyz.shape[1] == 3
        assert self.rgb.shape[0] == N
        assert self.rgb.shape[1] == 3

    def create_gaussians(self):
        """
        Create gaussians object from the dataset
        """
        self.verify_loaded_points()

        N = self.xyz.shape[0]
        initial_opacity = torch.ones(N, 1) * inverse_sigmoid(self.config.initial_opacity)
        # compute scale based on the density of the points around each point
        initial_scale = compute_initial_scale_from_sparse_points(
            self.xyz,
            num_neighbors=self.config.initial_scale_num_neighbors,
            neighbor_dist_to_scale_factor=self.config.initial_scale_factor,
            max_initial_scale=self.config.max_initial_scale,
        )
        initial_quaternion = torch.zeros(N, 4)
        initial_quaternion[:, 0] = 1.0

        return Gaussians(
            xyz=self.xyz.to(self.device),
            rgb=self.rgb.to(self.device),
            opacity=initial_opacity.to(self.device),
            scale=initial_scale.to(self.device),
            quaternion=initial_quaternion.to(self.device),
        )

    def get_images(self):
        """
        get images from the dataset
        """

        return self.images

    def get_cameras(self):
        """
        get cameras from the dataset
        """

        return self.cameras


class ColmapData(GaussianSplattingDataset):
    """
    This class loads data similar to Mip-Nerf 360 Dataset generated with colmap

    Format:

    dataset_dir:
        images: full resoloution images
            ...
        images_N: downsampled images by a factor of N
            ...
        poses_bounds.npy: currently unused
        sparse:
            0:
                cameras.bin
                images.bin
                points3D.bin
    """

    def __init__(
        self,
        colmap_directory_path: str,
        device: torch.device,
        downsample_factor: int,
        config: SplatConfig,
    ) -> None:
        super().__init__(config)

        self.colmap_directory_path = colmap_directory_path
        self.device = device
        self.downsample_factor = downsample_factor

        # load sparse points
        points_path = os.path.join(colmap_directory_path, "sparse", "0", "points3D.bin")
        sparse_points = read_points3D_binary(points_path)
        num_points = len(sparse_points)

        self.xyz = torch.zeros(num_points, 3)
        self.rgb = torch.zeros(num_points, 3)
        row = 0
        for _, point in sparse_points.items():
            self.xyz[row] = torch.tensor(point.xyz, dtype=torch.float32)
            self.rgb[row] = torch.tensor(
                (point.rgb / 255.0 - 0.5) / 0.28209479177387814, dtype=torch.float32
            )
            row += 1

        # load images
        image_info_path = os.path.join(colmap_directory_path, "sparse", "0", "images.bin")
        self.image_info = read_images_binary(image_info_path)

        self.images = []
        for _, image_info in self.image_info.items():
            # load image
            image_path = os.path.join(
                colmap_directory_path,
                f"images_{self.downsample_factor}",
                image_info.name,
            )
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # load transform
            camera_T_world = torch.eye(4)
            camera_T_world[:3, :3] = torch.tensor(qvec2rotmat(image_info.qvec), dtype=torch.float32)
            camera_T_world[:3, 3] = torch.tensor(image_info.tvec, dtype=torch.float32)

            self.images.append(
                Image(
                    image=torch.from_numpy(image).to(torch.uint8).to(self.device),
                    camera_id=image_info.camera_id,
                    camera_T_world=camera_T_world.to(self.device),
                )
            )

        # load cameras
        cameras_path = os.path.join(colmap_directory_path, "sparse", "0", "cameras.bin")
        cameras = read_cameras_binary(cameras_path)

        self.cameras = {}
        for camera_id, camera in cameras.items():
            K = torch.zeros((3, 3), dtype=torch.float32, device=self.device)
            if camera.model == "SIMPLE_PINHOLE":
                # colmap params [f, cx, cy]
                K[0, 0] = camera.params[0] / float(self.downsample_factor)
                K[1, 1] = camera.params[0] / float(self.downsample_factor)
                K[0, 2] = camera.params[1] / float(self.downsample_factor)
                K[1, 2] = camera.params[2] / float(self.downsample_factor)
                K[2, 2] = 1.0
            elif camera.model == "PINHOLE":
                # colmap params [fx, fy, cx, cy]
                K[0, 0] = camera.params[0] / float(self.downsample_factor)
                K[1, 1] = camera.params[1] / float(self.downsample_factor)
                K[0, 2] = camera.params[2] / float(self.downsample_factor)
                K[1, 2] = camera.params[3] / float(self.downsample_factor)
                K[2, 2] = 1.0
            else:
                raise NotImplementedError("Only Pinhole and Simple Pinhole cameras are supported")

            self.cameras[camera_id] = Camera(
                width=self.images[0].image.shape[1],
                height=self.images[0].image.shape[0],
                K=K,
            )
