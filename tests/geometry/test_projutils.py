import pytest
import torch
from torch.testing import assert_allclose

import gradslam as gs

from tests.common import default_to_cpu_if_no_gpu


class TestHomogenizePoints:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_homogenize_points(self, device):

        device = default_to_cpu_if_no_gpu(device)

        # Points to homogenize
        pts = torch.tensor(
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            device=device,
        )

        pts_gt = torch.tensor(
            [
                [1.0, 2.0, 3.0, 1.0],
                [3.0, 2.0, 1.0, 1.0],
                [-1.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
        )

        pts_pred = gs.homogenize_points(pts)
        assert_allclose(pts_pred, pts_gt)

    def test_raises_type_error(self):
        pts = [1, 2, 3]
        with pytest.raises(TypeError):
            gs.homogenize_points(pts)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_dim_error(self, device):
        device = default_to_cpu_if_no_gpu(device)
        pts = torch.rand(3, device=device)
        with pytest.raises(ValueError):
            gs.homogenize_points(pts)


class TestUnhomogenizePoints:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_unhomogenize_points(self, device):
        device = default_to_cpu_if_no_gpu(device)
        # Copied Kornia testcase (and added a few).
        # Points to unhomogenize
        pts = torch.tensor(
            [
                [1.0, 2.0, 1.0],
                [0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0],
                [-1.0, -2.0, -1.0],
                [0.0, 1.0, -2.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            device=device,
        )

        pts_gt = torch.tensor(
            [
                [1.0, 2.0],
                [0.0, 0.5],
                [2.0, 1.0],
                [1.0, 2.0],
                [0.0, -0.5],
                [0.0, 0.0],
                [1.0, 0.0],
            ],
            device=device,
        )

        pts_pred = gs.unhomogenize_points(pts)
        assert_allclose(pts_pred, pts_gt)

    def test_raises_type_error(self):
        pts = [1, 2, 3.0]
        with pytest.raises(TypeError):
            gs.unhomogenize_points(pts)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_dim_error(self, device):
        device = default_to_cpu_if_no_gpu(device)
        pts = torch.rand(3, device=device)
        with pytest.raises(ValueError):
            gs.unhomogenize_points(pts)


class TestProjectPoints:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (3, 4))
    def test_cases_1_and_4(self, device, lastdim):
        device = default_to_cpu_if_no_gpu(device)
        cam_coords = torch.rand(10, lastdim, device=device)
        proj_mat = torch.rand(4, 4, device=device)
        pixel_coords = gs.project_points(cam_coords, proj_mat)
        assert pixel_coords.shape == (10, 2)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (3, 4))
    def test_cases_2_and_5(self, device, lastdim):
        device = default_to_cpu_if_no_gpu(device)
        cam_coords = torch.rand(2, 10, lastdim, device=device)
        proj_mat = torch.rand(4, 4, device=device)
        pixel_coords = gs.project_points(cam_coords, proj_mat)
        assert pixel_coords.shape == (2, 10, 2)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (3, 4))
    def test_cases_3_and_6(self, device, lastdim):
        device = default_to_cpu_if_no_gpu(device)
        cam_coords = torch.rand(2, 10, lastdim, device=device)
        proj_mat = torch.rand(2, 4, 4, device=device)
        pixel_coords = gs.project_points(cam_coords, proj_mat)
        assert pixel_coords.shape == (2, 10, 2)

    def test_type_error_cam_coords(self):
        cam_coords = [1, 2, 3]
        proj_mat = [1, 2, 3]
        with pytest.raises(TypeError):
            gs.project_points(cam_coords, proj_mat)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (3, 4))
    def test_type_error_proj_mat(self, device, lastdim):
        device = default_to_cpu_if_no_gpu(device)
        cam_coords = torch.rand(2, 10, lastdim, device=device)
        proj_mat = [1, 2, 3]
        with pytest.raises(TypeError):
            gs.project_points(cam_coords, proj_mat)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_value_error_cam_coords(self, device):
        device = default_to_cpu_if_no_gpu(device)
        cam_coords = torch.rand(2, device=device)
        proj_mat = torch.rand(4, 4, device=device)
        with pytest.raises(ValueError):
            gs.project_points(cam_coords, proj_mat)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_value_error_cam_coords_2(self, device):
        device = default_to_cpu_if_no_gpu(device)
        cam_coords = torch.rand(2, 2, device=device)
        proj_mat = torch.rand(4, 4, device=device)
        with pytest.raises(ValueError):
            gs.project_points(cam_coords, proj_mat)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_value_error_cam_coords_3(self, device):
        device = default_to_cpu_if_no_gpu(device)
        cam_coords = torch.rand(2, 2, device=device)
        proj_mat = torch.rand(4, 4, device=device)
        with pytest.raises(ValueError):
            gs.project_points(cam_coords, proj_mat)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_value_error_proj_mat(self, device):
        device = default_to_cpu_if_no_gpu(device)
        cam_coords = torch.rand(2, 2, device=device)
        proj_mat = torch.rand(3, device=device)
        with pytest.raises(ValueError):
            gs.project_points(cam_coords, proj_mat)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("dims", ((4, 3), (3, 4)))
    def test_value_error_proj_mat_2(self, device, dims):
        device = default_to_cpu_if_no_gpu(device)
        cam_coords = torch.rand(2, 2, device=device)
        proj_mat = torch.rand(dims[0], dims[1], device=device)
        with pytest.raises(ValueError):
            gs.project_points(cam_coords, proj_mat)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_value_error_batchsize(self, device):
        device = default_to_cpu_if_no_gpu(device)
        cam_coords = torch.rand(2, 1, 10, 3, device=device)
        proj_mat = torch.rand(1, 4, 4, device=device)
        with pytest.raises(ValueError):
            gs.project_points(cam_coords, proj_mat)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_value_error_batchsize_2(self, device):
        device = default_to_cpu_if_no_gpu(device)
        cam_coords = torch.rand(2, 10, 3, device=device)
        proj_mat = torch.rand(1, 4, 4, device=device)
        with pytest.raises(ValueError):
            gs.project_points(cam_coords, proj_mat)


class TestUnprojectPoints:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (2, 3))
    def test_cases_1_and_4(self, device, lastdim):
        device = default_to_cpu_if_no_gpu(device)
        pixel_coords = torch.rand(10, lastdim, device=device)
        intrinsics_inv = torch.rand(3, 3, device=device)
        depths = torch.rand(10, device=device)
        cam_coords = gs.unproject_points(pixel_coords, intrinsics_inv, depths)
        assert cam_coords.shape == (10, 3)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (2, 3))
    def test_cases_2_and_5(self, device, lastdim):
        device = default_to_cpu_if_no_gpu(device)
        pixel_coords = torch.rand(2, 10, lastdim, device=device)
        intrinsics_inv = torch.rand(3, 3, device=device)
        depths = torch.rand(2, 10, device=device)
        cam_coords = gs.unproject_points(pixel_coords, intrinsics_inv, depths)
        assert cam_coords.shape == (2, 10, 3)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (2, 3))
    def test_cases_3_and_6(self, device, lastdim):
        device = default_to_cpu_if_no_gpu(device)
        pixel_coords = torch.rand(2, 10, lastdim, device=device)
        intrinsics_inv = torch.rand(2, 3, 3, device=device)
        depths = torch.rand(2, 10, device=device)
        cam_coords = gs.unproject_points(pixel_coords, intrinsics_inv, depths)
        assert cam_coords.shape == (2, 10, 3)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (2, 3))
    def test_type_errors(self, device, lastdim):
        device = default_to_cpu_if_no_gpu(device)
        pixel_coords = [1, 2, 3]
        intrinsics_inv = [1, 2, 3]
        depths = [1, 2, 3]
        with pytest.raises(TypeError):
            gs.unproject_points(pixel_coords, intrinsics_inv, depths)
        pixel_coords = torch.rand(2, 10, lastdim, device=device)
        with pytest.raises(TypeError):
            gs.unproject_points(pixel_coords, intrinsics_inv, depths)
        intrinsics_inv = torch.rand(2, 3, 3, device=device)
        with pytest.raises(TypeError):
            gs.unproject_points(pixel_coords, intrinsics_inv, depths)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_value_errors(self, device):
        device = default_to_cpu_if_no_gpu(device)
        pixel_coords = torch.rand(2, device=device)
        intrinsics_inv = torch.rand(3, 3, device=device)
        depths = torch.rand(2, 10, device=device)
        with pytest.raises(ValueError):
            gs.unproject_points(pixel_coords, intrinsics_inv, depths)
        pixel_coords = torch.rand(2, 3, device=device)
        intrinsics_inv = torch.rand(3, device=device)
        with pytest.raises(ValueError):
            gs.unproject_points(pixel_coords, intrinsics_inv, depths)
        intrinsics_inv = torch.rand(3, 3, device=device)
        pixel_coords = torch.rand(2, 3, device=device)
        depths = torch.rand(1, device=device)
        with pytest.raises(ValueError):
            gs.unproject_points(pixel_coords, intrinsics_inv, depths)
        pixel_coords = torch.rand(2, 1, 2, 3, device=device)
        intrinsics_inv = torch.rand(1, 3, 3, device=device)
        with pytest.raises(ValueError):
            gs.unproject_points(pixel_coords, intrinsics_inv, depths)
        pixel_coords = torch.rand(2, 1, 2, 3, device=device)
        intrinsics_inv = torch.rand(1, 3, 3, device=device)
        with pytest.raises(ValueError):
            gs.unproject_points(pixel_coords, intrinsics_inv, depths)


class TestInverseIntrinsics:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (3, 4))
    def test_output_shape(self, device, lastdim):
        device = default_to_cpu_if_no_gpu(device)
        rand_vals = torch.rand(10, 4, device=device)
        intrinsics = torch.zeros(10, lastdim, lastdim, device=device)
        intrinsics[..., 0, 0] = rand_vals[:, 0]  # fx
        intrinsics[..., 1, 1] = rand_vals[:, 1]  # fy
        intrinsics[..., 0, 2] = rand_vals[:, 2]  # cx
        intrinsics[..., 1, 2] = rand_vals[:, 3]  # cy
        intrinsics[..., 2, 2] = 1
        intrinsics[..., -1, -1] = 1
        test_res = gs.inverse_intrinsics(intrinsics)

        assert test_res.shape == intrinsics.shape

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (3, 4))
    def test_output_values(self, device, lastdim):
        device = default_to_cpu_if_no_gpu(device)
        rand_vals = torch.rand(4, device=device)
        intrinsics = torch.zeros(lastdim, lastdim, device=device)
        intrinsics[0, 0] = rand_vals[0]  # fx
        intrinsics[1, 1] = rand_vals[1]  # fy
        intrinsics[0, 2] = rand_vals[2]  # cx
        intrinsics[1, 2] = rand_vals[3]  # cy
        intrinsics[2, 2] = 1
        intrinsics[-1, -1] = 1
        test_res = gs.inverse_intrinsics(intrinsics)
        correct_res = torch.inverse(intrinsics)

        assert test_res.shape == intrinsics.shape
        assert (
            (test_res - correct_res).abs().sum() / correct_res.abs().sum()
        ).item() < 1e-2

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (3, 4))
    def test_output_values_more_dims(self, device, lastdim):
        device = default_to_cpu_if_no_gpu(device)
        rand_vals = torch.rand(5, 10, 4, device=device)
        intrinsics = torch.zeros(5, 10, lastdim, lastdim, device=device)
        intrinsics[..., 0, 0] = rand_vals[..., 0]  # fx
        intrinsics[..., 1, 1] = rand_vals[..., 1]  # fy
        intrinsics[..., 0, 2] = rand_vals[..., 2]  # cx
        intrinsics[..., 1, 2] = rand_vals[..., 3]  # cy
        intrinsics[..., 2, 2] = 1
        intrinsics[..., -1, -1] = 1
        test_res = gs.inverse_intrinsics(intrinsics)
        correct_res = []
        for b in range(5):
            res = [torch.inverse(intrinsics[b, s]) for s in range(10)]
            correct_res.append(torch.stack(res, 0))
        correct_res = torch.stack(correct_res, 0)

        assert test_res.shape == intrinsics.shape
        assert (
            (test_res - correct_res).abs().sum() / correct_res.abs().sum()
        ).item() < 1e-2

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (2, 3))
    def test_type_errors(self, device, lastdim):
        device = default_to_cpu_if_no_gpu(device)
        intrinsics = [1, 2, 3]
        with pytest.raises(TypeError):
            gs.inverse_intrinsics(intrinsics)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_value_errors(self, device):
        device = default_to_cpu_if_no_gpu(device)
        intrinsics = torch.rand(3, device=device)
        with pytest.raises(ValueError):
            gs.inverse_intrinsics(intrinsics)
        intrinsics = torch.rand(2, 3, device=device)
        with pytest.raises(ValueError):
            gs.inverse_intrinsics(intrinsics)
        intrinsics = torch.rand(3, 4, device=device)
        with pytest.raises(ValueError):
            gs.inverse_intrinsics(intrinsics)
        intrinsics = torch.rand(5, 3, 4, device=device)
        with pytest.raises(ValueError):
            gs.inverse_intrinsics(intrinsics)
