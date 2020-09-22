import numpy as np
import pytest
import torch
from torch.testing import assert_allclose

from gradslam.odometry.groundtruth import GroundTruthOdometryProvider
from gradslam.structures.rgbdimages import RGBDImages

from tests.common import default_to_cpu_if_no_gpu


class TestGroundTruth:
    @staticmethod
    def init_rgbdimages(
        channels_first: bool = False,
        device: str = "cpu",
    ):
        device = torch.device(device)
        channels_first = False
        colors = torch.rand(1, 2, 32, 32, 3)
        depths = torch.rand(1, 2, 32, 32, 1)
        intrinsics = torch.rand(1, 1, 4, 4)
        rad1 = 0.1
        rad2 = 0.7
        poses = torch.tensor(
            [
                [
                    [np.cos(rad1), -np.sin(rad1), 0.0, 0.05],
                    [np.sin(rad1), np.cos(rad1), 0.0, 0.03],
                    [0.0, 0.0, 1.0, 0.01],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [np.cos(rad2), -np.sin(rad2), 0.0, 0.05],
                    [np.sin(rad2), np.cos(rad2), 0.0, 0.03],
                    [0.0, 0.0, 1.0, 0.01],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            device=device,
            dtype=colors.dtype,
        ).unsqueeze(0)
        return RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        ).to(device)

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_groundtruth_provide(self, device):
        device = default_to_cpu_if_no_gpu(device)
        rgbdimages = TestGroundTruth.init_rgbdimages(device=device)
        odom = GroundTruthOdometryProvider()
        t = 0
        odom_transform = odom.provide(rgbdimages[:, t], rgbdimages[:, t + 1])
        new_pose = rgbdimages[:, t].poses.squeeze() @ odom_transform.squeeze()

        assert odom_transform.shape == rgbdimages[:, t + 1].poses.shape
        assert_allclose(new_pose.squeeze(), rgbdimages[:, t + 1].poses.squeeze())

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_groundtruth_raises_type_error(self, device):
        device = default_to_cpu_if_no_gpu(device)
        rgbdimages = TestGroundTruth.init_rgbdimages(device=device)
        tensor = torch.rand(1, 1, 4, 4)
        odom = GroundTruthOdometryProvider()
        with pytest.raises(TypeError):
            odom = odom.provide(rgbdimages[:, 0], tensor)
        with pytest.raises(TypeError):
            odom = odom.provide(tensor, rgbdimages[:, 0])

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_groundtruth_raises_value_error(self, device):
        device = default_to_cpu_if_no_gpu(device)
        channels_first = False
        rgbdimages = TestGroundTruth.init_rgbdimages(
            device=device, channels_first=channels_first
        )
        nopose_rgbdimages = TestGroundTruth.init_rgbdimages(device=device)
        nopose_rgbdimages.poses = None
        batch_rgbdimages = RGBDImages(
            rgbdimages.rgb_image.repeat(2, 1, 1, 1, 1),
            rgbdimages.depth_image.repeat(2, 1, 1, 1, 1),
            rgbdimages.intrinsics.repeat(2, 1, 1, 1),
            rgbdimages.poses.repeat(2, 1, 1, 1),
            channels_first=channels_first,
        ).to(device)
        tensor = torch.rand(1, 1, 4, 4)
        odom = GroundTruthOdometryProvider()
        with pytest.raises(ValueError):
            odom = odom.provide(rgbdimages[:, 0], nopose_rgbdimages[:, 1])
        with pytest.raises(ValueError):
            odom = odom.provide(nopose_rgbdimages[:, 0], rgbdimages[:, 1])
        with pytest.raises(ValueError):
            odom = odom.provide(rgbdimages[:, 0], rgbdimages)
        with pytest.raises(ValueError):
            odom = odom.provide(rgbdimages, rgbdimages[:, 1])
        with pytest.raises(ValueError):
            odom = odom.provide(rgbdimages, batch_rgbdimages)
