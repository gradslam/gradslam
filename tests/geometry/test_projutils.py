import pytest

import torch
from torch.testing import assert_allclose

import gradslam as gs


class TestHomogenizePoints:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_homogenize_points(self, device):
        # Points to homogenize
        pts = torch.tensor(
            [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0],],
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
        pts = torch.rand(3, device=device)
        with pytest.raises(ValueError):
            gs.homogenize_points(pts)


class TestUnhomogenizePoints:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_unhomogenize_points(self, device):
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
        pts = torch.rand(3, device=device)
        with pytest.raises(ValueError):
            gs.unhomogenize_points(pts)


class TestProjectPoints:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (3, 4))
    def test_cases_1_and_4(self, device, lastdim):
        cam_coords = torch.rand(10, lastdim, device=device)
        proj_mat = torch.rand(4, 4, device=device)
        pixel_coords = gs.project_points(cam_coords, proj_mat)
        assert pixel_coords.shape == (10, 2)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (3, 4))
    def test_cases_2_and_5(self, device, lastdim):
        cam_coords = torch.rand(2, 10, lastdim, device=device)
        proj_mat = torch.rand(4, 4, device=device)
        pixel_coords = gs.project_points(cam_coords, proj_mat)
        assert pixel_coords.shape == (2, 10, 2)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (3, 4))
    def test_cases_3_and_6(self, device, lastdim):
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
        cam_coords = torch.rand(2, 10, lastdim, device=device)
        proj_mat = [1, 2, 3]
        with pytest.raises(TypeError):
            gs.project_points(cam_coords, proj_mat)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_value_error_cam_coords(self, device):
        cam_coords = torch.rand(2, device=device)
        proj_mat = torch.rand(4, 4, device=device)
        with pytest.raises(ValueError):
            gs.project_points(cam_coords, proj_mat)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_value_error_cam_coords_2(self, device):
        cam_coords = torch.rand(2, 2, device=device)
        proj_mat = torch.rand(4, 4, device=device)
        with pytest.raises(ValueError):
            gs.project_points(cam_coords, proj_mat)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_value_error_cam_coords_3(self, device):
        cam_coords = torch.rand(2, 2, device=device)
        proj_mat = torch.rand(4, 4, device=device)
        with pytest.raises(ValueError):
            gs.project_points(cam_coords, proj_mat)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_value_error_proj_mat(self, device):
        cam_coords = torch.rand(2, 2, device=device)
        proj_mat = torch.rand(3, device=device)
        with pytest.raises(ValueError):
            gs.project_points(cam_coords, proj_mat)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("dims", ((4, 3), (3, 4)))
    def test_value_error_proj_mat_2(self, device, dims):
        cam_coords = torch.rand(2, 2, device=device)
        proj_mat = torch.rand(dims[0], dims[1], device=device)
        with pytest.raises(ValueError):
            gs.project_points(cam_coords, proj_mat)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_value_error_batchsize(self, device):
        cam_coords = torch.rand(2, 1, 10, 3, device=device)
        proj_mat = torch.rand(1, 4, 4, device=device)
        with pytest.raises(ValueError):
            gs.project_points(cam_coords, proj_mat)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_value_error_batchsize_2(self, device):
        cam_coords = torch.rand(2, 10, 3, device=device)
        proj_mat = torch.rand(1, 4, 4, device=device)
        with pytest.raises(ValueError):
            gs.project_points(cam_coords, proj_mat)


class TestUnprojectPoints:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (2, 3))
    def test_cases_1_and_4(self, device, lastdim):
        pixel_coords = torch.rand(10, lastdim, device=device)
        intrinsics_inv = torch.rand(3, 3, device=device)
        depths = torch.rand(10, device=device)
        cam_coords = gs.unproject_points(pixel_coords, intrinsics_inv, depths)
        assert cam_coords.shape == (10, 3)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (2, 3))
    def test_cases_2_and_5(self, device, lastdim):
        pixel_coords = torch.rand(2, 10, lastdim, device=device)
        intrinsics_inv = torch.rand(3, 3, device=device)
        depths = torch.rand(2, 10, device=device)
        cam_coords = gs.unproject_points(pixel_coords, intrinsics_inv, depths)
        assert cam_coords.shape == (2, 10, 3)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (2, 3))
    def test_cases_3_and_6(self, device, lastdim):
        pixel_coords = torch.rand(2, 10, lastdim, device=device)
        intrinsics_inv = torch.rand(2, 3, 3, device=device)
        depths = torch.rand(2, 10, device=device)
        cam_coords = gs.unproject_points(pixel_coords, intrinsics_inv, depths)
        assert cam_coords.shape == (2, 10, 3)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    @pytest.mark.parametrize("lastdim", (2, 3))
    def test_type_errors(self, device, lastdim):
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
