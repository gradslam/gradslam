import logging
import unittest

import pytest
import torch
from torch.testing import assert_allclose

from gradslam.structures.utils import *
from gradslam.geometry.geometryutils import create_meshgrid
from gradslam.structures.rgbdimages import RGBDImages

from tests.common import default_to_cpu_if_no_gpu, load_test_data


class TestPointcloudsFromRGBDImages:
    @pytest.mark.parametrize("channels_first", (False, True))
    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_pointclouds_from_rgbdimages(self, channels_first, device):
        device = default_to_cpu_if_no_gpu(device)
        colors, depths, intrinsics, poses = load_test_data(channels_first)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )

        pointclouds = pointclouds_from_rgbdimages(rgbdimages[:, 0]).to(device)
        projected_pointclouds = pointclouds.pinhole_projection(
            intrinsics.to(device).squeeze(1)
        )
        proj0 = projected_pointclouds.points_list[0][..., :-1]
        meshgrid = (
            create_meshgrid(rgbdimages.shape[2], rgbdimages.shape[3], False)
            .to(device)
            .squeeze(0)
        )
        meshgrid = torch.cat(
            [
                meshgrid[..., 1:],
                meshgrid[..., 0:1],
            ],
            -1,
        )
        groundtruth = meshgrid[rgbdimages[0, 0].valid_depth_mask.squeeze()]

        assert_allclose(proj0.round().float(), groundtruth.float())

        # without filtering missing depths
        pointclouds2 = pointclouds_from_rgbdimages(
            rgbdimages[:, 0], filter_missing_depths=False
        ).to(device)

        for b in range(len(pointclouds)):
            filtered_points = pointclouds.points_list[b]
            unfiltered_points = pointclouds2.points_list[b]
            m = 0
            for n in range(len(filtered_points)):
                while (
                    not ((filtered_points[n] - unfiltered_points[m]) ** 2).sum() < 1e-12
                ):
                    m += 1
                assert ((filtered_points[n] - unfiltered_points[m]) ** 2).sum() < 1e-12
                m += 1

    @pytest.mark.parametrize("device", ("cpu", "cuda:0"))
    def test_raises_errors(self, device):
        device = default_to_cpu_if_no_gpu(device)
        channels_first = False
        colors, depths, intrinsics, poses = load_test_data(channels_first)
        rgbdimages = RGBDImages(
            colors.to(device),
            depths.to(device),
            intrinsics.to(device),
            poses.to(device),
            channels_first=channels_first,
        )  # .to(device)

        sigma = 0.6
        with pytest.raises(
            TypeError, match="Expected rgbdimages to be of type gradslam.RGBDImages"
        ):
            pointclouds = pointclouds_from_rgbdimages(depths).to(device)

        with pytest.raises(
            ValueError, match="Expected rgbdimages to have sequence length of 1"
        ):
            pointclouds = pointclouds_from_rgbdimages(rgbdimages).to(device)


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger("config")
    logger.setLevel(logging.DEBUG)
    unittest.main()
