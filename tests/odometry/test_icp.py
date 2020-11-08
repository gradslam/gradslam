import numpy as np
import pytest
import torch
from torch.testing import assert_allclose

from gradslam.odometry.icp import ICPOdometryProvider
from gradslam.structures.rgbdimages import RGBDImages
from gradslam.structures.pointclouds import Pointclouds
from gradslam.structures.utils import pointclouds_from_rgbdimages
from tests.common import load_test_data


class TestICP:
    def test_ICP_provide(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        odom = ICPOdometryProvider(
            numiters=numiters,
            damp=damp,
            dist_thresh=dist_thresh,
        )
        odom_transform = odom.provide(tgt_pointclouds, src_pointclouds)
        odom_transform = odom_transform.squeeze(1).squeeze(0)

        assert odom_transform.shape == transform.shape
        assert_allclose(odom_transform, transform)

    def test_ICP_raises_value_error(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        odom = ICPOdometryProvider()
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
