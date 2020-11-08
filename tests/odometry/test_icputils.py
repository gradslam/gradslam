import numpy as np
import pytest
import torch

# from torch.autograd import gradcheck
from torch.testing import assert_allclose

from gradslam.odometry.icputils import *
from gradslam.structures.rgbdimages import RGBDImages
from gradslam.structures.pointclouds import Pointclouds
from gradslam.structures.utils import pointclouds_from_rgbdimages

from tests.common import default_to_cpu_if_no_gpu, load_test_data

CUDA_NOT_AVAILABLE = "No CUDA devices available"


class TestSolveLinearSystem:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_solve_linear_system(self, device):
        device = default_to_cpu_if_no_gpu(device)
        A = torch.tensor(
            [
                [0.1, 0.7, 0.3, 0.6],
                [0.5, 0.2, 0.4, 0.8],
                [0.3, 0.9, 0.5, 0.2],
                [0.8, 0.2, 0.3, 0.4],
                [0.7, 0.9, 0.3, 0.8],
            ],
            dtype=torch.float32,
            device=device,
        )
        b = torch.tensor(
            [
                [0.7],
                [0.2],
                [0.9],
                [0.2],
                [0.9],
            ],
            dtype=torch.float32,
            device=device,
        )
        damp = 1e-8

        x = solve_linear_system(A, b, damp)
        res = A @ x
        assert res.shape == b.shape
        assert_allclose(res, b)

    def test_solve_linear_system_raises_type_error(self):
        device = default_to_cpu_if_no_gpu("cuda")
        A = torch.tensor(
            [
                [0.1, 0.7, 0.3, 0.6],
                [0.5, 0.2, 0.4, 0.8],
                [0.3, 0.9, 0.5, 0.2],
                [0.8, 0.2, 0.3, 0.4],
                [0.7, 0.9, 0.3, 0.8],
            ],
            dtype=torch.float32,
            device=device,
        )
        b = torch.tensor(
            [
                [0.7],
                [0.2],
                [0.9],
                [0.2],
                [0.9],
            ],
            dtype=torch.float32,
            device=device,
        )
        damp = torch.tensor(1e-8, device=device)

        with pytest.raises(TypeError):
            x = solve_linear_system(np.ones((5, 4)), b, damp)
        with pytest.raises(TypeError):
            x = solve_linear_system(A, np.ones((5, 1)), damp)
        with pytest.raises(TypeError):
            x = solve_linear_system(A, b, 5)

    def test_solve_linear_system_raises_value_error(self):
        device = default_to_cpu_if_no_gpu("cuda")
        A = torch.tensor(
            [
                [0.1, 0.7, 0.3, 0.6],
                [0.5, 0.2, 0.4, 0.8],
                [0.3, 0.9, 0.5, 0.2],
                [0.8, 0.2, 0.3, 0.4],
                [0.7, 0.9, 0.3, 0.8],
            ],
            dtype=torch.float32,
            device=device,
        )
        b = torch.tensor(
            [
                [0.7],
                [0.2],
                [0.9],
                [0.2],
                [0.9],
            ],
            dtype=torch.float32,
            device=device,
        )
        damp = torch.tensor(1e-8, device=device)

        with pytest.raises(ValueError):
            x = solve_linear_system(A.unsqueeze(0), b, damp)
        with pytest.raises(ValueError):
            x = solve_linear_system(A, b.unsqueeze(0), damp)
        with pytest.raises(ValueError):
            x = solve_linear_system(A, b, damp.unsqueeze(0))
        with pytest.raises(ValueError):
            x = solve_linear_system(A, b.repeat(1, 2), damp)
        with pytest.raises(ValueError):
            x = solve_linear_system(A, b.repeat(2, 1), damp)


class TestGaussNewtonSolve:
    # Functionality tests in TestPointToPlaneGradICP and TestPointToPlaneICP
    def test_gauss_newton_raises_type_error(self):
        device = default_to_cpu_if_no_gpu("cuda")
        src_pc = torch.tensor(
            [
                [0.1, 0.7, 0.3],
                [0.5, 0.2, 0.4],
                [0.3, 0.9, 0.5],
                [0.8, 0.2, 0.3],
                [0.7, 0.9, 0.3],
            ],
            dtype=torch.float32,
            device=device,
        )
        rad = 0.2
        transform = torch.tensor(
            [
                [np.cos(rad), -np.sin(rad), 0.0, 0.05],
                [np.sin(rad), np.cos(rad), 0.0, 0.03],
                [0.0, 0.0, 1.0, 0.01],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=src_pc.dtype,
        )
        tgt_pc = ((transform[:3, :3] @ src_pc.T) + transform[:-1, -1:]).T
        tgt_normals = src_pc.clone()
        src_pc = src_pc.unsqueeze(0)
        tgt_pc = tgt_pc.unsqueeze(0)
        tgt_normals = tgt_normals.unsqueeze(0)
        dist_thresh = 0.2

        with pytest.raises(TypeError):
            x = gauss_newton_solve(
                src_pc.cpu().detach().numpy(), tgt_pc, tgt_normals, dist_thresh
            )
        with pytest.raises(TypeError):
            x = gauss_newton_solve(
                src_pc, tgt_pc.cpu().detach().numpy(), tgt_normals, dist_thresh
            )
        with pytest.raises(TypeError):
            x = gauss_newton_solve(
                src_pc, tgt_pc, tgt_normals.cpu().detach().numpy(), dist_thresh
            )
        with pytest.raises(TypeError):
            x = gauss_newton_solve(
                src_pc, tgt_pc, tgt_normals, torch.tensor(0.2, device=device)
            )

    def test_gauss_newton_raises_value_error(self):
        device = default_to_cpu_if_no_gpu("cuda")
        src_pc = torch.tensor(
            [
                [0.1, 0.7, 0.3],
                [0.5, 0.2, 0.4],
                [0.3, 0.9, 0.5],
                [0.8, 0.2, 0.3],
                [0.7, 0.9, 0.3],
            ],
            dtype=torch.float32,
            device=device,
        )
        rad = 0.2
        transform = torch.tensor(
            [
                [np.cos(rad), -np.sin(rad), 0.0, 0.05],
                [np.sin(rad), np.cos(rad), 0.0, 0.03],
                [0.0, 0.0, 1.0, 0.01],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=src_pc.dtype,
        )
        tgt_pc = ((transform[:3, :3] @ src_pc.T) + transform[:-1, -1:]).T
        tgt_normals = src_pc.clone()
        src_pc = src_pc.unsqueeze(0)
        tgt_pc = tgt_pc.unsqueeze(0)
        tgt_normals = tgt_normals.unsqueeze(0)
        dist_thresh = 0.2

        with pytest.raises(ValueError):
            x = gauss_newton_solve(
                src_pc.unsqueeze(0), tgt_pc, tgt_normals, dist_thresh
            )
        with pytest.raises(ValueError):
            x = gauss_newton_solve(
                src_pc, tgt_pc.unsqueeze(0), tgt_normals, dist_thresh
            )
        with pytest.raises(ValueError):
            x = gauss_newton_solve(
                src_pc, tgt_pc, tgt_normals.unsqueeze(0), dist_thresh
            )
        with pytest.raises(ValueError):
            x = gauss_newton_solve(
                src_pc.repeat(2, 1, 1), tgt_pc, tgt_normals, dist_thresh
            )
        with pytest.raises(ValueError):
            x = gauss_newton_solve(
                src_pc, tgt_pc.repeat(2, 1, 1), tgt_normals, dist_thresh
            )
        with pytest.raises(ValueError):
            x = gauss_newton_solve(
                src_pc, tgt_pc, tgt_normals.repeat(2, 1, 1), dist_thresh
            )
        with pytest.raises(ValueError):
            x = gauss_newton_solve(
                src_pc, tgt_pc, tgt_normals.repeat(1, 2, 1), dist_thresh
            )
        with pytest.raises(ValueError):
            x = gauss_newton_solve(
                src_pc.repeat(1, 1, 2), tgt_pc, tgt_normals, dist_thresh
            )
        with pytest.raises(ValueError):
            x = gauss_newton_solve(
                src_pc, tgt_pc.repeat(1, 1, 2), tgt_normals, dist_thresh
            )
        with pytest.raises(ValueError):
            x = gauss_newton_solve(
                src_pc, tgt_pc, tgt_normals.repeat(1, 1, 2), dist_thresh
            )

    # def test_gauss_newton_gradcheck(self):
    #     device = default_to_cpu_if_no_gpu("cuda")
    #     src_pc = torch.tensor(
    #         [
    #             [0.1, 0.7, 0.3],
    #             [0.5, 0.2, 0.4],
    #             [0.3, 0.9, 0.5],
    #             [0.8, 0.2, 0.3],
    #             [0.7, 0.9, 0.3],
    #         ],
    #         dtype=torch.float32,
    #         device=device,
    #     )
    #     rad = 0.2
    #     transform = torch.tensor(
    #         [
    #             [np.cos(rad), -np.sin(rad), 0.0, 0.05],
    #             [np.sin(rad), np.cos(rad), 0.0, 0.03],
    #             [0.0, 0.0, 1.0, 0.01],
    #             [0.0, 0.0, 0.0, 1.0],
    #         ],
    #         device=device,
    #         dtype=src_pc.dtype,
    #     )
    #     tgt_pc = ((transform[:3, :3] @ src_pc.T) + transform[:-1, -1:]).T
    #     tgt_normals = src_pc.clone()
    #     src_pc = src_pc.unsqueeze(0)
    #     tgt_pc = tgt_pc.unsqueeze(0)
    #     tgt_normals = tgt_normals.unsqueeze(0)
    #     dist_thresh = 0.2

    #     # evaluate function gradient
    #     src_pc = src_pc.requires_grad_().type(torch.float64)
    #     tgt_pc = tgt_pc.requires_grad_().type(torch.float64)
    #     tgt_normals = tgt_normals.requires_grad_().type(torch.float64)
    #     assert gradcheck(
    #         gauss_newton_solve, (src_pc, tgt_pc, tgt_normals), raise_exception=True
    #     )


class TestPointToPlaneICP:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason=CUDA_NOT_AVAILABLE)
    def test_point_to_plane_ICP_transform1(self):
        device = torch.device("cuda")
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first, batch_size=1)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )
        sigma = 0.6
        src_pointclouds = pointclouds_from_rgbdimages(rgbdimages[:, 0]).to(device)
        rad = 0.2
        transform = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.05],
                [0.0, np.cos(rad), -np.sin(rad), 0.03],
                [0.0, np.sin(rad), np.cos(rad), 0.01],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=colors.dtype,
        )
        # transform = torch.tensor(
        #     [
        #         [np.cos(rad), -np.sin(rad), 0.0, 0.05],
        #         [np.sin(rad), np.cos(rad), 0.0, 0.03],
        #         [0.0, 0.0, 1.0, 0.01],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ],
        #     device=device,
        #     dtype=colors.dtype,
        # )
        tgt_pointclouds = src_pointclouds.transform(transform)

        src_pc = src_pointclouds.points_padded
        tgt_pc = tgt_pointclouds.points_padded
        tgt_normals = tgt_pointclouds.normals_padded
        initial_transform = torch.eye(4, device=device)
        numiters = 100
        damp = 1e-8
        dist_thresh = None
        t, idx = point_to_plane_ICP(
            src_pc,
            tgt_pc,
            tgt_normals,
            initial_transform,
            numiters,
            damp,
            dist_thresh,
        )

        assert t.shape == transform.shape
        assert_allclose(t, transform)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason=CUDA_NOT_AVAILABLE)
    def test_point_to_plane_ICP_transform2(self):
        device = torch.device("cuda")
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first, batch_size=1)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )
        sigma = 0.6
        src_pointclouds = pointclouds_from_rgbdimages(rgbdimages[:, 0]).to(device)
        rad = 0.2
        transform = torch.tensor(
            [
                [np.cos(rad), -np.sin(rad), 0.0, 0.05],
                [np.sin(rad), np.cos(rad), 0.0, 0.03],
                [0.0, 0.0, 1.0, 0.01],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=colors.dtype,
        )
        tgt_pointclouds = src_pointclouds.transform(transform)

        src_pc = src_pointclouds.points_padded
        tgt_pc = tgt_pointclouds.points_padded
        tgt_normals = tgt_pointclouds.normals_padded
        initial_transform = torch.eye(4, device=device)
        numiters = 100
        damp = 1e-8
        dist_thresh = None
        t, idx = point_to_plane_ICP(
            src_pc,
            tgt_pc,
            tgt_normals,
            initial_transform,
            numiters,
            damp,
            dist_thresh,
        )

        assert t.shape == transform.shape
        assert_allclose(t, transform)

    def test_point_to_plane_ICP_raises_type_error(self):
        dtype = torch.float32
        device = default_to_cpu_if_no_gpu("cuda")
        rad = 0.2
        transform = torch.tensor(
            [
                [np.cos(rad), -np.sin(rad), 0.0, 0.05],
                [np.sin(rad), np.cos(rad), 0.0, 0.03],
                [0.0, 0.0, 1.0, 0.01],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
        src_pc = torch.rand((1, 5, 3), dtype=dtype, device=device)
        tgt_pc = torch.rand((1, 8, 3), dtype=dtype, device=device)
        tgt_normals = torch.rand((1, 8, 3), dtype=dtype, device=device)
        initial_transform = torch.eye(4, device=device)
        numiters = 20
        damp = 1e-8
        dist_thresh = None
        point_to_plane_ICP(
            src_pc,
            tgt_pc,
            tgt_normals,
            initial_transform,
            numiters,
            damp,
            dist_thresh,
        )
        with pytest.raises(TypeError):
            point_to_plane_ICP(
                "a",
                tgt_pc,
                tgt_normals,
                initial_transform,
                numiters,
                damp,
                dist_thresh,
            )
        with pytest.raises(TypeError):
            point_to_plane_ICP(
                src_pc,
                "a",
                tgt_normals,
                initial_transform,
                numiters,
                damp,
                dist_thresh,
            )
        with pytest.raises(TypeError):
            point_to_plane_ICP(
                src_pc, tgt_pc, "a", initial_transform, numiters, damp, dist_thresh
            )
        with pytest.raises(TypeError):
            point_to_plane_ICP(
                src_pc, tgt_pc, tgt_normals, "a", numiters, damp, dist_thresh
            )
        with pytest.raises(TypeError):
            point_to_plane_ICP(
                src_pc,
                tgt_pc,
                tgt_normals,
                initial_transform,
                "a",
                damp,
                dist_thresh,
            )
        with pytest.raises(TypeError):
            point_to_plane_ICP(
                src_pc,
                tgt_pc,
                tgt_normals,
                initial_transform,
                numiters,
                "a",
                dist_thresh,
            )
        with pytest.raises(TypeError):
            point_to_plane_ICP(
                src_pc, tgt_pc, tgt_normals, initial_transform, numiters, damp, "a"
            )

    def test_point_to_plane_ICP_raises_value_error(self):
        dtype = torch.float32
        device = default_to_cpu_if_no_gpu("cuda")
        rad = 0.2
        transform = torch.tensor(
            [
                [np.cos(rad), -np.sin(rad), 0.0, 0.05],
                [np.sin(rad), np.cos(rad), 0.0, 0.03],
                [0.0, 0.0, 1.0, 0.01],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
        src_pc = torch.rand((1, 5, 3), dtype=dtype, device=device)
        tgt_pc = torch.rand((1, 8, 3), dtype=dtype, device=device)
        tgt_normals = torch.rand((1, 8, 3), dtype=dtype, device=device)
        initial_transform = torch.eye(4, device=device)

        with pytest.raises(ValueError):
            point_to_plane_ICP(
                src_pc.unsqueeze(0), tgt_pc, tgt_normals, initial_transform
            )
        with pytest.raises(ValueError):
            point_to_plane_ICP(
                src_pc, tgt_pc.unsqueeze(0), tgt_normals, initial_transform
            )
        with pytest.raises(ValueError):
            point_to_plane_ICP(
                src_pc, tgt_pc, tgt_normals.unsqueeze(0), initial_transform
            )
        with pytest.raises(ValueError):
            point_to_plane_ICP(
                src_pc, tgt_pc, tgt_normals, initial_transform.unsqueeze(0)
            )
        with pytest.raises(ValueError):
            point_to_plane_ICP(src_pc, tgt_pc, tgt_normals, initial_transform[:3, :3])

    # def test_point_to_plane_ICP_gradcheck(self):
    #     dtype = torch.float32
    #     device = default_to_cpu_if_no_gpu("cuda")
    #     rad = 0.2
    #     transform = torch.tensor(
    #         [
    #             [np.cos(rad), -np.sin(rad), 0.0, 0.05],
    #             [np.sin(rad), np.cos(rad), 0.0, 0.03],
    #             [0.0, 0.0, 1.0, 0.01],
    #             [0.0, 0.0, 0.0, 1.0],
    #         ],
    #         device=device,
    #         dtype=dtype,
    #     )
    #     src_pc = torch.rand((1, 5, 3), dtype=dtype, device=device)
    #     tgt_pc = torch.rand((1, 8, 3), dtype=dtype, device=device)
    #     tgt_normals = torch.rand((1, 8, 3), dtype=dtype, device=device)
    #     initial_transform = torch.eye(4, device=device)

    #     # evaluate function gradient
    #     src_pc = src_pc.requires_grad_().type(torch.float64)
    #     tgt_pc = tgt_pc.requires_grad_().type(torch.float64)
    #     tgt_normals = tgt_normals.requires_grad_().type(torch.float64)
    #     initial_transform = initial_transform.requires_grad_().type(torch.float64)
    #     assert gradcheck(point_to_plane_ICP, (src_pc, tgt_pc, tgt_normals, initial_transform), raise_exception=True)


class TestPointToPlaneGradICP:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason=CUDA_NOT_AVAILABLE)
    def test_point_to_plane_gradICP_transform1(self):
        device = torch.device("cuda")
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first, batch_size=1)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )
        sigma = 0.6
        src_pointclouds = pointclouds_from_rgbdimages(rgbdimages[:, 0]).to(device)
        rad = 0.2
        transform = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.05],
                [0.0, np.cos(rad), -np.sin(rad), 0.03],
                [0.0, np.sin(rad), np.cos(rad), 0.01],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=colors.dtype,
        )
        # transform = torch.tensor(
        #     [
        #         [np.cos(rad), -np.sin(rad), 0.0, 0.05],
        #         [np.sin(rad), np.cos(rad), 0.0, 0.03],
        #         [0.0, 0.0, 1.0, 0.01],
        #         [0.0, 0.0, 0.0, 1.0],
        #     ],
        #     device=device,
        #     dtype=colors.dtype,
        # )
        tgt_pointclouds = src_pointclouds.transform(transform)

        src_pc = src_pointclouds.points_padded
        tgt_pc = tgt_pointclouds.points_padded
        tgt_normals = tgt_pointclouds.normals_padded
        initial_transform = torch.eye(4, device=device)
        numiters = 100
        damp = 1e-8
        dist_thresh = None
        t, idx = point_to_plane_gradICP(
            src_pc,
            tgt_pc,
            tgt_normals,
            initial_transform,
            numiters,
            damp,
            dist_thresh,
        )

        assert t.shape == transform.shape
        assert_allclose(t, transform)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason=CUDA_NOT_AVAILABLE)
    def test_point_to_plane_gradICP_transform2(self):
        device = torch.device("cuda")
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first, batch_size=1)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )
        sigma = 0.6
        src_pointclouds = pointclouds_from_rgbdimages(rgbdimages[:, 0]).to(device)
        rad = 0.2
        transform = torch.tensor(
            [
                [np.cos(rad), -np.sin(rad), 0.0, 0.05],
                [np.sin(rad), np.cos(rad), 0.0, 0.03],
                [0.0, 0.0, 1.0, 0.01],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=colors.dtype,
        )
        tgt_pointclouds = src_pointclouds.transform(transform)

        src_pc = src_pointclouds.points_padded
        tgt_pc = tgt_pointclouds.points_padded
        tgt_normals = tgt_pointclouds.normals_padded
        initial_transform = torch.eye(4, device=device)
        numiters = 100
        damp = 1e-8
        dist_thresh = None
        t, idx = point_to_plane_gradICP(
            src_pc,
            tgt_pc,
            tgt_normals,
            initial_transform,
            numiters,
            damp,
            dist_thresh,
        )

        assert t.shape == transform.shape
        assert_allclose(t, transform)

    def test_point_to_plane_gradICP_raises_type_error(self):
        dtype = torch.float32
        device = default_to_cpu_if_no_gpu("cuda")
        rad = 0.2
        transform = torch.tensor(
            [
                [np.cos(rad), -np.sin(rad), 0.0, 0.05],
                [np.sin(rad), np.cos(rad), 0.0, 0.03],
                [0.0, 0.0, 1.0, 0.01],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
        src_pc = torch.rand((1, 5, 3), dtype=dtype, device=device)
        tgt_pc = torch.rand((1, 8, 3), dtype=dtype, device=device)
        tgt_normals = torch.rand((1, 8, 3), dtype=dtype, device=device)
        initial_transform = torch.eye(4, device=device)
        numiters = 20
        damp = 1e-8
        dist_thresh = None
        point_to_plane_gradICP(
            src_pc,
            tgt_pc,
            tgt_normals,
            initial_transform,
            numiters,
            damp,
            dist_thresh,
        )
        with pytest.raises(TypeError):
            point_to_plane_gradICP(
                "a",
                tgt_pc,
                tgt_normals,
                initial_transform,
                numiters,
                damp,
                dist_thresh,
            )
        with pytest.raises(TypeError):
            point_to_plane_gradICP(
                src_pc,
                "a",
                tgt_normals,
                initial_transform,
                numiters,
                damp,
                dist_thresh,
            )
        with pytest.raises(TypeError):
            point_to_plane_gradICP(
                src_pc, tgt_pc, "a", initial_transform, numiters, damp, dist_thresh
            )
        with pytest.raises(TypeError):
            point_to_plane_gradICP(
                src_pc, tgt_pc, tgt_normals, "a", numiters, damp, dist_thresh
            )
        with pytest.raises(TypeError):
            point_to_plane_gradICP(
                src_pc,
                tgt_pc,
                tgt_normals,
                initial_transform,
                "a",
                damp,
                dist_thresh,
            )
        with pytest.raises(TypeError):
            point_to_plane_gradICP(
                src_pc,
                tgt_pc,
                tgt_normals,
                initial_transform,
                numiters,
                "a",
                dist_thresh,
            )
        with pytest.raises(TypeError):
            point_to_plane_gradICP(
                src_pc, tgt_pc, tgt_normals, initial_transform, numiters, damp, "a"
            )
        with pytest.raises(TypeError):
            point_to_plane_gradICP(src_pc, tgt_pc, tgt_normals, lambda_max="a")
        with pytest.raises(TypeError):
            point_to_plane_gradICP(src_pc, tgt_pc, tgt_normals, B="a")
        with pytest.raises(TypeError):
            point_to_plane_gradICP(src_pc, tgt_pc, tgt_normals, B2="a")
        with pytest.raises(TypeError):
            point_to_plane_gradICP(src_pc, tgt_pc, tgt_normals, nu="a")

    def test_point_to_plane_gradICP_raises_value_error(self):
        dtype = torch.float32
        device = default_to_cpu_if_no_gpu("cuda")
        rad = 0.2
        transform = torch.tensor(
            [
                [np.cos(rad), -np.sin(rad), 0.0, 0.05],
                [np.sin(rad), np.cos(rad), 0.0, 0.03],
                [0.0, 0.0, 1.0, 0.01],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
        src_pc = torch.rand((1, 5, 3), dtype=dtype, device=device)
        tgt_pc = torch.rand((1, 8, 3), dtype=dtype, device=device)
        tgt_normals = torch.rand((1, 8, 3), dtype=dtype, device=device)
        initial_transform = torch.eye(4, device=device)

        with pytest.raises(ValueError):
            point_to_plane_gradICP(
                src_pc.unsqueeze(0), tgt_pc, tgt_normals, initial_transform
            )
        with pytest.raises(ValueError):
            point_to_plane_gradICP(
                src_pc, tgt_pc.unsqueeze(0), tgt_normals, initial_transform
            )
        with pytest.raises(ValueError):
            point_to_plane_gradICP(
                src_pc, tgt_pc, tgt_normals.unsqueeze(0), initial_transform
            )
        with pytest.raises(ValueError):
            point_to_plane_gradICP(
                src_pc, tgt_pc, tgt_normals, initial_transform.unsqueeze(0)
            )
        with pytest.raises(ValueError):
            point_to_plane_gradICP(
                src_pc, tgt_pc, tgt_normals, initial_transform[:3, :3]
            )

    # def test_point_to_plane_gradICP_gradcheck(self):
    #     dtype = torch.float32
    #     device = default_to_cpu_if_no_gpu("cuda")
    #     rad = 0.2
    #     transform = torch.tensor(
    #         [
    #             [np.cos(rad), -np.sin(rad), 0.0, 0.05],
    #             [np.sin(rad), np.cos(rad), 0.0, 0.03],
    #             [0.0, 0.0, 1.0, 0.01],
    #             [0.0, 0.0, 0.0, 1.0],
    #         ],
    #         device=device,
    #         dtype=dtype,
    #     )
    #     src_pc = torch.rand((1, 5, 3), dtype=dtype, device=device)
    #     tgt_pc = torch.rand((1, 8, 3), dtype=dtype, device=device)
    #     tgt_normals = torch.rand((1, 8, 3), dtype=dtype, device=device)
    #     initial_transform = torch.eye(4, device=device)

    #     # evaluate function gradient
    #     src_pc = src_pc.requires_grad_().type(torch.float64)
    #     tgt_pc = tgt_pc.requires_grad_().type(torch.float64)
    #     tgt_normals = tgt_normals.requires_grad_().type(torch.float64)
    #     initial_transform = initial_transform.requires_grad_().type(torch.float64)
    #     assert gradcheck(point_to_plane_gradICP, (src_pc, tgt_pc, tgt_normals, initial_transform), raise_exception=True)


class TestDownsamplePointclouds:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_downsample_pointclouds(self, device):
        device = default_to_cpu_if_no_gpu(device)
        points = torch.tensor(
            [
                [5.0, 5.0, 5.0],
                [3.0, 3.0, 3.0],
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
            dtype=torch.float,
        ).unsqueeze(0)
        normals = points * -1
        colors = points * 2
        pc2im_bnhw = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 1, 4, 2],
                [0, 2, 3, 1],
                [0, 3, 0, 3],
                [0, 4, 3, 3],
                [0, 5, 3, 6],
            ],
            device=device,
            dtype=torch.int64,
        )

        # Example1: with normals, with colors, ds_ratio=3
        pointclouds = Pointclouds(points, normals, colors)
        ds_ratio = 3
        ds_pointclouds = downsample_pointclouds(pointclouds, pc2im_bnhw, ds_ratio)
        groundtruth_points = torch.tensor(
            [
                [5.0, 5.0, 5.0],
                [3.0, 2.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
            dtype=torch.float,
        ).unsqueeze(0)
        groundtruth_normals = groundtruth_points * -1
        groundtruth_colors = groundtruth_points * 2

        assert ds_pointclouds.points_padded.shape == groundtruth_points.shape
        assert ds_pointclouds.normals_padded.shape == groundtruth_normals.shape
        assert ds_pointclouds.colors_padded.shape == groundtruth_colors.shape
        assert_allclose(ds_pointclouds.points_padded, groundtruth_points)
        assert_allclose(ds_pointclouds.normals_padded, groundtruth_normals)
        assert_allclose(ds_pointclouds.colors_padded, groundtruth_colors)

        # Example2: no normals, no colors, ds_ratio=2
        pointclouds = Pointclouds(points)
        ds_ratio = 2
        ds_pointclouds = downsample_pointclouds(pointclouds, pc2im_bnhw, ds_ratio)
        groundtruth_points = torch.tensor(
            [[5.0, 5.0, 5.0], [3.0, 3.0, 3.0]],
            device=device,
            dtype=torch.float,
        ).unsqueeze(0)

        assert ds_pointclouds.points_padded.shape == groundtruth_points.shape
        assert ds_pointclouds.normals_padded is None
        assert_allclose(ds_pointclouds.points_padded, groundtruth_points)

    def test_downsample_pointclouds_raises_type_error(self):
        device = default_to_cpu_if_no_gpu("cuda")
        points = torch.tensor(
            [
                [5.0, 5.0, 5.0],
                [3.0, 3.0, 3.0],
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
            dtype=torch.float,
        ).unsqueeze(0)
        normals = points * -1
        pc2im_bnhw = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 1, 4, 2],
                [0, 2, 3, 1],
                [0, 3, 0, 3],
                [0, 4, 3, 3],
                [0, 5, 3, 6],
            ],
            device=device,
            dtype=torch.int64,
        )
        pointclouds = Pointclouds(points, normals)
        ds_ratio = 3
        ds_pointclouds = downsample_pointclouds(pointclouds, pc2im_bnhw, ds_ratio)
        with pytest.raises(TypeError):
            downsample_pointclouds("a", pc2im_bnhw, ds_ratio)
        with pytest.raises(TypeError):
            downsample_pointclouds(pointclouds, "a", ds_ratio)
        with pytest.raises(TypeError):
            downsample_pointclouds(pointclouds, pc2im_bnhw, "a")

    def test_downsample_pointclouds_raises_value_error(self):
        device = default_to_cpu_if_no_gpu("cuda")
        points = torch.tensor(
            [
                [5.0, 5.0, 5.0],
                [3.0, 3.0, 3.0],
                [1.0, 2.0, 3.0],
                [3.0, 2.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
            dtype=torch.float,
        ).unsqueeze(0)
        normals = points * -1
        pc2im_bnhw = torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 1, 4, 2],
                [0, 2, 3, 1],
                [0, 3, 0, 3],
                [0, 4, 3, 3],
                [0, 5, 3, 6],
            ],
            device=device,
            dtype=torch.int64,
        )
        pointclouds = Pointclouds(points, normals)
        ds_ratio = 3
        ds_pointclouds = downsample_pointclouds(pointclouds, pc2im_bnhw, ds_ratio)
        with pytest.raises(ValueError):
            downsample_pointclouds(pointclouds, pc2im_bnhw.unsqueeze(1), ds_ratio)
        with pytest.raises(ValueError):
            downsample_pointclouds(pointclouds, pc2im_bnhw.repeat(1, 2), ds_ratio)


class TestDownsampleRGBDImages:
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_downsample_rgbdimages(self, device):
        device = default_to_cpu_if_no_gpu(device)
        image = (
            torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0],
                        [3.0, 3.0, 3.0],
                    ],
                    [
                        [4.0, 4.0, 4.0],
                        [5.0, 5.0, 5.0],
                        [6.0, 6.0, 6.0],
                        [7.0, 7.0, 7.0],
                    ],
                    [
                        [8.0, 8.0, 8.0],
                        [9.0, 9.0, 9.0],
                        [10.0, 10.0, 10.0],
                        [11.0, 11.0, 11.0],
                    ],
                ],
                device=device,
                dtype=torch.float,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        depth = torch.ones_like(image[..., :1])
        intrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).to(device)
        poses = torch.eye(4).unsqueeze(0).unsqueeze(0).to(device)
        rgbdimages = RGBDImages(
            image, depth, intrinsics, poses, channels_first=False
        ).to(device)
        ds_ratio = 2
        ds_pointclouds = downsample_rgbdimages(rgbdimages, ds_ratio)
        groundtruth_points = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [0.0, 2.0, 1.0],
                [2.0, 2.0, 1.0],
            ],
            device=device,
            dtype=torch.float,
        ).unsqueeze(0)
        groundtruth_colors = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [2.0, 2.0, 2.0],
                [8.0, 8.0, 8.0],
                [10.0, 10.0, 10.0],
            ],
            device=device,
            dtype=torch.float,
        ).unsqueeze(0)
        groundtruth_normals = rgbdimages.normal_map[..., ::ds_ratio, ::ds_ratio, :]
        groundtruth_normals = groundtruth_normals.reshape(
            1, ds_pointclouds.normals_padded.shape[1], 3
        )

        assert ds_pointclouds.points_padded.shape == groundtruth_points.shape
        assert ds_pointclouds.colors_padded.shape == groundtruth_colors.shape
        assert ds_pointclouds.normals_padded.shape == groundtruth_normals.shape
        assert_allclose(ds_pointclouds.points_padded, groundtruth_points)
        assert_allclose(ds_pointclouds.colors_padded, groundtruth_colors)
        assert_allclose(ds_pointclouds.normals_padded, groundtruth_normals)

    def test_downsample_rgbdimages_raises_type_error(self):
        device = default_to_cpu_if_no_gpu("cuda")
        image = (
            torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0],
                        [3.0, 3.0, 3.0],
                    ],
                    [
                        [4.0, 4.0, 4.0],
                        [5.0, 5.0, 5.0],
                        [6.0, 6.0, 6.0],
                        [7.0, 7.0, 7.0],
                    ],
                    [
                        [8.0, 8.0, 8.0],
                        [9.0, 9.0, 9.0],
                        [10.0, 10.0, 10.0],
                        [11.0, 11.0, 11.0],
                    ],
                ],
                device=device,
                dtype=torch.float,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )

        depth = torch.ones_like(image[..., :1])
        intrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).to(device)
        poses = torch.eye(4).unsqueeze(0).unsqueeze(0).to(device)
        rgbdimages = RGBDImages(
            image, depth, intrinsics, poses, channels_first=False
        ).to(device)
        ds_ratio = 2
        ds_pointclouds = downsample_rgbdimages(rgbdimages, ds_ratio)
        with pytest.raises(TypeError):
            downsample_rgbdimages("a", ds_ratio)
        with pytest.raises(TypeError):
            downsample_rgbdimages(rgbdimages, "a")

    def test_downsample_rgbdimages_raises_value_error(self):
        device = default_to_cpu_if_no_gpu("cuda")
        image = (
            torch.tensor(
                [
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0],
                        [3.0, 3.0, 3.0],
                    ],
                    [
                        [4.0, 4.0, 4.0],
                        [5.0, 5.0, 5.0],
                        [6.0, 6.0, 6.0],
                        [7.0, 7.0, 7.0],
                    ],
                    [
                        [8.0, 8.0, 8.0],
                        [9.0, 9.0, 9.0],
                        [10.0, 10.0, 10.0],
                        [11.0, 11.0, 11.0],
                    ],
                ],
                device=device,
                dtype=torch.float,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        ).repeat(1, 2, 1, 1, 1)

        depth = torch.ones_like(image[..., :1])
        intrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).to(device)
        poses = torch.eye(4).unsqueeze(0).unsqueeze(0).to(device).repeat(1, 2, 1, 1)
        rgbdimages = RGBDImages(
            image, depth, intrinsics, poses, channels_first=False
        ).to(device)
        ds_ratio = 2
        ds_pointclouds = downsample_rgbdimages(rgbdimages[:, 0], ds_ratio)
        with pytest.raises(ValueError):
            downsample_rgbdimages(rgbdimages, ds_ratio)
