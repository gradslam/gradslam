from pathlib import Path

import numpy as np
import pytest
import torch
from torch.testing import assert_allclose
from torch.utils.data import DataLoader

from gradslam.datasets.scannet import Scannet
from gradslam.odometry.gradicp import GradICPOdometryProvider
from gradslam.odometry.icp import ICPOdometryProvider
from gradslam.slam import fusionutils
from gradslam.structures.rgbdimages import RGBDImages
from gradslam.structures.pointclouds import Pointclouds

SCANNET_ROOT = "D:/Soroush-LFS/datasets/ScanNet-gradSLAM/extractions/scans"
SCANNET_META_ROOT = (
    "D:/Soroush-LFS/datasets/ScanNet-gradSLAM/extractions/sequence_associations"
)

# Tests below can only be run if a Scannet dataset is available
SCANNET_NOT_FOUND = "Scannet scans not found at default location: {}".format(
    SCANNET_ROOT
)
SCANNET_META_NOT_FOUND = "Scannet metadata not found at default location: {}".format(
    SCANNET_META_ROOT
)
CUDA_NOT_AVAILABLE = "No CUDA devices available"


class TestGradICP:
    @pytest.mark.skipif(not Path(SCANNET_ROOT).exists(), reason=SCANNET_NOT_FOUND)
    @pytest.mark.skipif(
        not Path(SCANNET_META_ROOT).exists(), reason=SCANNET_META_NOT_FOUND
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason=CUDA_NOT_AVAILABLE)
    def test_gradICP_provide(self):
        device = "cuda:0"
        channels_first = False
        dataset = Scannet(
            SCANNET_ROOT,
            SCANNET_META_ROOT,
            (
                "scene0333_00",
                "scene0636_00",
            ),
            start=0,
            end=4,
            height=240,
            width=320,
            channels_first=channels_first,
        )
        loader = DataLoader(dataset=dataset, batch_size=1)
        colors, depths, intrinsics, poses, *_ = next(iter(loader))
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )
        sigma = 0.6
        src_pointclouds = fusionutils.rgbdimages_to_pointclouds(
            rgbdimages[:, 0], sigma=sigma
        ).to(device)
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

        downsample_ratio = 1
        numiters = 30
        damp = 1e-8
        dist_thresh = 0.2
        lambda_max = 2.0
        B = 1.0
        B2 = 1.0
        nu = 200.0
        odom = GradICPOdometryProvider(
            downsample_ratio=downsample_ratio,
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
