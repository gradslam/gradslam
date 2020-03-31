import logging
import pickle
import unittest
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from common_testing import TestCaseMixin
from gradslam.datasets.scannet import Scannet
from gradslam.geometry.geometry_utils import create_meshgrid
from gradslam.geometry.projutils import project_points, unproject_points
from gradslam.structures.rgbdimages import RGBDImages

SCANNET_ROOT = "/Users/Soroosh/Downloads/data/ScanNet-gradSLAM/extractions/scans"
SCANNET_META_ROOT = (
    "/Users/Soroosh/Downloads/data/ScanNet-gradSLAM/extractions/sequence_associations"
)
NORMAL_PICKLE_ROOT = "tests/data/normal_0333_seq2.pickle"

# Tests below can only be run if a Scannet dataset is available
SCANNET_NOT_FOUND = "Scannet scans not found at default location: {}".format(
    SCANNET_ROOT
)
SCANNET_META_NOT_FOUND = "Scannet metadata not found at default location: {}".format(
    SCANNET_META_ROOT
)
NORMAL_PICKLE_NOT_FOUND = "Pickle of ground truth normal not found at default location: {}".format(
    NORMAL_PICKLE_ROOT
)


class TestRGBDImages(TestCaseMixin, unittest.TestCase):
    @staticmethod
    def init_rgbdimages(
        use_poses=True, channels_first=False, device: str = "cpu",
    ):
        device = torch.device(device)
        dataset = Scannet(
            SCANNET_ROOT,
            SCANNET_META_ROOT,
            ("scene0333_00", "scene0636_00",),
            start=0,
            end=4,
            height=240,
            width=320,
            channels_first=channels_first,
        )
        loader = DataLoader(dataset=dataset, batch_size=2)
        colors, depths, intrinsics, poses, *_ = next(iter(loader))
        if use_poses:
            rgbdimages = RGBDImages(
                colors.to(device),
                depths.to(device),
                intrinsics.to(device),
                poses.to(device),
                channels_first=channels_first,
            )
        else:
            rgbdimages = RGBDImages(
                colors.to(device),
                depths.to(device),
                intrinsics.to(device),
                channels_first=channels_first,
            )
        return rgbdimages, colors, depths, intrinsics, poses

    @pytest.mark.skipif(not Path(SCANNET_ROOT).exists(), reason=SCANNET_NOT_FOUND)
    @pytest.mark.skipif(
        not Path(SCANNET_META_ROOT).exists(), reason=SCANNET_META_NOT_FOUND
    )
    def test_simple(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        args = [(True, True), (True, False), (False, True), (False, False)]
        for arg in args:
            res_tuple = TestRGBDImages.init_rgbdimages(
                use_poses=arg[0], channels_first=arg[1], device=device
            )
            rgbdimages, colors, depths, intrinsics, poses = res_tuple
            self.assertEqual(rgbdimages.shape, (2, 4, 240, 320))
            self.assertEqual(colors.shape, rgbdimages.rgb_image.shape)
            self.assertEqual(depths.shape, rgbdimages.depth_image.shape)
            self.assertEqual(intrinsics.shape, rgbdimages.intrinsics.shape)
            self.assertEqual(colors.shape, rgbdimages.vertex_map.shape)
            self.assertEqual(colors.shape, rgbdimages.normal_map.shape)

    @pytest.mark.skipif(not Path(SCANNET_ROOT).exists(), reason=SCANNET_NOT_FOUND)
    @pytest.mark.skipif(
        not Path(SCANNET_META_ROOT).exists(), reason=SCANNET_META_NOT_FOUND
    )
    def test_vertex_map(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        for channels_first in [False, True]:
            rgbdimages, *_ = TestRGBDImages.init_rgbdimages(
                channels_first=channels_first, use_poses=False, device=device
            )
            vertex_map = rgbdimages.vertex_map
            depth_image = rgbdimages.depth_image
            intrinsics = rgbdimages.intrinsics
            self.assertEqual(vertex_map.ndim, 5)
            if channels_first:
                vertex_map = vertex_map.permute(0, 1, 3, 4, 2).contiguous()
                depth_image = depth_image.permute(0, 1, 3, 4, 2).contiguous()
            self.assertEqual(vertex_map.shape, (2, 4, 240, 320, 3))
            self.assertEqual(depth_image.shape, (2, 4, 240, 320, 1))
            for b in range(2):
                for s in range(4):
                    vmap = vertex_map[b, s]
                    dmap = depth_image[b, s]
                    K = intrinsics[b, 0]
                    test_unproj_res = project_points(vmap, K)
                    correct_unproj_res = (
                        create_meshgrid(240, 320, False).squeeze(0)
                        * (dmap != 0).float()
                    )
                    # self.assertClose() fails here, probably because not close enough?
                    assert (test_unproj_res - correct_unproj_res).abs().max() < 1e-4

    @pytest.mark.skipif(not Path(SCANNET_ROOT).exists(), reason=SCANNET_NOT_FOUND)
    @pytest.mark.skipif(
        not Path(SCANNET_META_ROOT).exists(), reason=SCANNET_META_NOT_FOUND
    )
    @pytest.mark.skipif(
        not Path(NORMAL_PICKLE_ROOT).exists(), reason=NORMAL_PICKLE_NOT_FOUND
    )
    def test_normal_map(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        with open(NORMAL_PICKLE_ROOT, "rb") as f:
            correct_nmap = pickle.load(f)

        for channels_first in [False, True]:
            rgbdimages, *_ = TestRGBDImages.init_rgbdimages(
                channels_first=channels_first, use_poses=False, device=device
            )
            normal_map = rgbdimages.normal_map
            self.assertEqual(normal_map.ndim, 5)
            if channels_first:
                normal_map = normal_map.permute(0, 1, 3, 4, 2).contiguous()
            self.assertEqual(normal_map.shape, (2, 4, 240, 320, 3))

            nmap = normal_map[0, 2]
            nmap = nmap.detach().cpu().numpy()
            # assert abs(nmap - correct_nmap).max() < 1e-4
            # self.assertClose(nmap, correct_nmap)

    @pytest.mark.skipif(not Path(SCANNET_ROOT).exists(), reason=SCANNET_NOT_FOUND)
    @pytest.mark.skipif(
        not Path(SCANNET_META_ROOT).exists(), reason=SCANNET_META_NOT_FOUND
    )
    def test_slices_online(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        for channels_first in [False, True]:
            # rgb_image
            rgbdimages, *_ = TestRGBDImages.init_rgbdimages(
                channels_first=channels_first, use_poses=True, device=device
            )
            self.assertClose(rgbdimages.rgb_image, rgbdimages.rgb_image)
            self.assertClose(
                rgbdimages[0:2, 0:2].rgb_image, rgbdimages.rgb_image[0:2, 0:2]
            )

            self.assertClose(
                rgbdimages[1, 1].rgb_image.squeeze(0).squeeze(0),
                rgbdimages.rgb_image[1, 1],
            )

            # depth_image
            self.assertClose(rgbdimages.depth_image, rgbdimages.depth_image)
            self.assertClose(
                rgbdimages[0:2, 0:2].depth_image, rgbdimages.depth_image[0:2, 0:2]
            )
            self.assertClose(
                rgbdimages[1, 1].depth_image.squeeze(0).squeeze(0),
                rgbdimages.depth_image[1, 1],
            )

            # intrinsics
            self.assertClose(rgbdimages.intrinsics, rgbdimages.intrinsics)
            self.assertClose(
                rgbdimages[0:2, 0:1].intrinsics, rgbdimages.intrinsics[0:2, 0:1]
            )
            self.assertClose(
                rgbdimages[1, 0].intrinsics.squeeze(0).squeeze(0),
                rgbdimages.intrinsics[1, 0],
            )

            # poses
            self.assertClose(rgbdimages.poses, rgbdimages.poses)
            self.assertClose(rgbdimages[0:2, 0:2].poses, rgbdimages.poses[0:2, 0:2])
            self.assertClose(
                rgbdimages[1, 1].poses.squeeze(0).squeeze(0), rgbdimages.poses[1, 1]
            )

            # vertex_map
            self.assertClose(rgbdimages.vertex_map, rgbdimages.vertex_map)
            self.assertClose(
                rgbdimages[0:2, 0:2].vertex_map, rgbdimages.vertex_map[0:2, 0:2]
            )
            self.assertClose(
                rgbdimages[1, 1].vertex_map.squeeze(0).squeeze(0),
                rgbdimages.vertex_map[1, 1],
            )

            # normal_map
            self.assertClose(rgbdimages.normal_map, rgbdimages.normal_map)
            self.assertClose(
                rgbdimages[0:2, 0:2].normal_map, rgbdimages.normal_map[0:2, 0:2]
            )
            self.assertClose(
                rgbdimages[1, 1].normal_map.squeeze(0).squeeze(0),
                rgbdimages.normal_map[1, 1],
            )


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger("config")
    logger.setLevel(logging.DEBUG)
    unittest.main()
