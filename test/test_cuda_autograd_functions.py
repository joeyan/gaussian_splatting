import unittest
import torch
from splat_py.cuda_autograd_functions import (
    CameraPointProjection,
    ComputeProjectionJacobian,
    ComputeSigmaWorld,
    ComputeConic,
    PrecomputeRGBFromSH,
)


class TestAutogradFunctions(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.device = torch.device("cuda")
        self.xyz = torch.tensor(
            [
                [1.0, 2.0, 15.0],
                [2.5, -1.0, 4.0],
                [-1.0, -2.0, 10.0],
            ],
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        self.K = torch.tensor(
            [
                [430.0, 0.0, 320.0],
                [0.0, 410.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
            device=self.device,
            requires_grad=False,
        )
        self.quaternion = torch.tensor(
            [
                [0.8, 0.2, 0.2, 0.2],
                [0.714, -0.002, -0.664, 0.221],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        self.scale = torch.tensor(
            [
                [0.02, 0.03, 0.04],
                [0.09, 0.03, 0.01],
                [2.0, 1.0, 0.1],
            ],
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        self.camera_T_world = torch.tensor(
            [
                [0.9999, 0.0089, 0.0073, -0.3283],
                [-0.0106, 0.9568, 0.2905, -1.9260],
                [-0.0044, -0.2906, 0.9568, 2.9581],
                [0.0000, 0.0000, 0.0000, 1.0000],
            ],
            dtype=torch.float64,
            device=self.device,
            requires_grad=False,
        )

    def test_camera_point_projection(self):
        test = torch.autograd.gradcheck(
            CameraPointProjection.apply, (self.xyz, self.K), raise_exception=True
        )
        self.assertTrue(test)

    def test_compute_gaussian_projection_jacobian(self):
        test = torch.autograd.gradcheck(
            ComputeProjectionJacobian.apply,
            (self.xyz, self.K),
            raise_exception=True,
        )
        self.assertTrue(test)

    def test_compute_sigma_world(self):
        test = torch.autograd.gradcheck(
            ComputeSigmaWorld.apply,
            (self.quaternion, self.scale),
            raise_exception=True,
        )
        self.assertTrue(test)

    def test_compute_conic(self):
        sigma_world = torch.rand(
            1,
            3,
            3,
            dtype=self.quaternion.dtype,
            device=self.quaternion.device,
            requires_grad=True,
        )
        projection_jacobian = torch.rand(
            1,
            2,
            3,
            dtype=self.quaternion.dtype,
            device=self.quaternion.device,
            requires_grad=True,
        )
        test = torch.autograd.gradcheck(
            ComputeConic.apply,
            (sigma_world, projection_jacobian, self.camera_T_world),
            raise_exception=True,
        )
        self.assertTrue(test)

    def test_compute_rgb_from_sh_1(self):
        N_gaussians = 100
        sh_coeffs = torch.ones(
            N_gaussians,
            3,
            1,
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        xyz = torch.rand(
            N_gaussians,
            3,
            dtype=torch.float64,
            device=self.device,
            requires_grad=False,
        )
        camera_T_world = torch.zeros(
            4,
            4,
            dtype=torch.float64,
            device=self.device,
            requires_grad=False,
        )
        test = torch.autograd.gradcheck(
            PrecomputeRGBFromSH.apply,
            (sh_coeffs, xyz, camera_T_world),
            raise_exception=True,
        )
        self.assertTrue(test)

    def test_compute_rgb_from_sh_4(self):
        N_gaussians = 100
        sh_coeffs = torch.ones(
            N_gaussians,
            3,
            4,
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        xyz = torch.rand(
            N_gaussians,
            3,
            dtype=torch.float64,
            device=self.device,
            requires_grad=False,
        )
        camera_T_world = torch.zeros(
            4,
            4,
            dtype=torch.float64,
            device=self.device,
            requires_grad=False,
        )
        test = torch.autograd.gradcheck(
            PrecomputeRGBFromSH.apply,
            (sh_coeffs, xyz, camera_T_world),
            raise_exception=True,
        )
        self.assertTrue(test)

    def test_compute_rgb_from_sh_9(self):
        N_gaussians = 100
        sh_coeffs = torch.ones(
            N_gaussians,
            3,
            9,
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        xyz = torch.rand(
            N_gaussians,
            3,
            dtype=torch.float64,
            device=self.device,
            requires_grad=False,
        )
        camera_T_world = torch.zeros(
            4,
            4,
            dtype=torch.float64,
            device=self.device,
            requires_grad=False,
        )
        test = torch.autograd.gradcheck(
            PrecomputeRGBFromSH.apply,
            (sh_coeffs, xyz, camera_T_world),
            raise_exception=True,
        )
        self.assertTrue(test)

    def test_compute_rgb_from_sh_16(self):
        N_gaussians = 100
        sh_coeffs = torch.ones(
            N_gaussians,
            3,
            16,
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        xyz = torch.rand(
            N_gaussians,
            3,
            dtype=torch.float64,
            device=self.device,
            requires_grad=False,
        )
        camera_T_world = torch.zeros(
            4,
            4,
            dtype=torch.float64,
            device=self.device,
            requires_grad=False,
        )
        test = torch.autograd.gradcheck(
            PrecomputeRGBFromSH.apply,
            (sh_coeffs, xyz, camera_T_world),
            raise_exception=True,
        )
        self.assertTrue(test)


if __name__ == "__main__":
    unittest.main()
