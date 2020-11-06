import numpy as np
import pytest
import torch
from torch.testing import assert_allclose

from gradslam.odometry.gradicp import GradICPOdometryProvider
from gradslam.structures.rgbdimages import RGBDImages
from gradslam.structures.pointclouds import Pointclouds
from gradslam.structures.utils import pointclouds_from_rgbdimages
from tests.common import load_test_data

CUDA_NOT_AVAILABLE = "No CUDA devices available"


class TestGradICP:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason=CUDA_NOT_AVAILABLE)
    def test_gradICP_provide(self):
        device = "cuda:0"
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
        rad = 0.1
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

        numiters = 30
        damp = 1e-8
        dist_thresh = 0.2
        lambda_max = 2.0
        B = 1.0
        B2 = 1.0
        nu = 200.0
        odom = GradICPOdometryProvider(
            numiters=numiters,
            damp=damp,
            dist_thresh=dist_thresh,
            lambda_max=lambda_max,
            B=B,
            B2=B2,
            nu=nu,
        )
        odom_transform = odom.provide(tgt_pointclouds, src_pointclouds)
        odom_transform = odom_transform.squeeze(1).squeeze(0)

        assert odom_transform.shape == transform.shape
        assert_allclose(odom_transform, transform)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason=CUDA_NOT_AVAILABLE)
    def test_gradICP_raises_value_error(self):
        device = "cuda:0"
        odom = GradICPOdometryProvider()
        pts = torch.tensor(
            [
                [5.0, 5.0, 5.0],
                [3.0, 3.0, 3.0],
            ],
            device=device,
        )
        normals = pts.clone()
        with pytest.raises(ValueError):
            pointclouds1 = Pointclouds([pts])
            pointclouds2 = Pointclouds([pts], [normals])
            transform = odom.provide(pointclouds1, pointclouds2)
        with pytest.raises(ValueError):
            pointclouds1 = Pointclouds([pts], [normals])
            pointclouds2 = Pointclouds([pts, pts], [normals, normals])
            transform = odom.provide(pointclouds1, pointclouds2)

    def test_gradICP_raises_no_cuda_error(self):
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError):
                odom = GradICPOdometryProvider()
        else:
            device = "cpu"
            odom = GradICPOdometryProvider()
            pts = torch.tensor(
                [
                    [3.0, 2.0, 5.0],
                    [3.0, 3.0, 4.0],
                    [3.0, 13.0, 4.0],
                    [3.0, 3.0, 24.0],
                ],
                device=device,
            )
            normals = pts.clone()
            with pytest.raises(RuntimeError):
                pointclouds1 = Pointclouds([pts], [normals])
                pointclouds2 = Pointclouds([pts], [normals])
                transform = odom.provide(pointclouds1, pointclouds2)

            device = "cuda:0"
            pointclouds1 = Pointclouds([pts], [normals])
            pointclouds2 = Pointclouds([pts.to(device)])
            with pytest.raises(RuntimeError):
                pointclouds1 = Pointclouds([pts], [normals.to(device)])
                pointclouds2 = Pointclouds([pts.to(device)])
                transform = odom.provide(pointclouds1, pointclouds2)
            with pytest.raises(RuntimeError):
                pointclouds1 = Pointclouds([pts], [normals]).to(device)
                pointclouds2 = Pointclouds([pts])
                transform = odom.provide(pointclouds1, pointclouds2)
