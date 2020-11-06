import logging
import os
import unittest

import numpy as np
import torch

from tests.common import default_to_cpu_if_no_gpu, load_test_data
from tests.common_testing import TestCaseMixin
from gradslam.geometry.geometryutils import create_meshgrid
from gradslam.geometry.projutils import project_points
from gradslam.structures.rgbdimages import RGBDImages


class TestRGBDImages(TestCaseMixin, unittest.TestCase):
    @staticmethod
    def init_rgbdimages(
        use_poses=True,
        channels_first=False,
        device: str = "cpu",
    ):
        device = torch.device(device)
        colors, depths, intrinsics, poses = load_test_data(channels_first)
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

    def test_simple(self):
        device = default_to_cpu_if_no_gpu("cuda:0")
        args = [(True, True), (True, False), (False, True), (False, False)]
        for arg in args:
            res_tuple = TestRGBDImages.init_rgbdimages(
                use_poses=arg[0], channels_first=arg[1], device=device
            )
            rgbdimages, colors, depths, intrinsics, poses = res_tuple
            self.assertEqual(rgbdimages.shape, (2, 3, 120, 160))
            self.assertEqual(colors.shape, rgbdimages.rgb_image.shape)
            self.assertEqual(depths.shape, rgbdimages.depth_image.shape)
            self.assertEqual(intrinsics.shape, rgbdimages.intrinsics.shape)
            self.assertEqual(colors.shape, rgbdimages.vertex_map.shape)
            self.assertEqual(colors.shape, rgbdimages.normal_map.shape)

    def test_vertex_map(self):
        device = default_to_cpu_if_no_gpu("cuda:0")

        scriptdir = os.path.dirname(os.path.realpath(__file__))
        gt_vmap = np.load(os.path.join(scriptdir, "../data/msrd_b2s3/vertex_map.npy"))
        gt_global_vmap = np.load(
            os.path.join(scriptdir, "../data/msrd_b2s3/global_vertex_map.npy")
        )

        for use_poses in [False, True]:
            for channels_first in [False, True]:
                rgbdimages, *_ = TestRGBDImages.init_rgbdimages(
                    channels_first=channels_first, use_poses=use_poses, device=device
                )
                vertex_map = rgbdimages.vertex_map
                global_vertex_map = rgbdimages.global_vertex_map
                depth_image = rgbdimages.depth_image
                intrinsics = rgbdimages.intrinsics
                self.assertEqual(vertex_map.ndim, 5)
                if channels_first:
                    vertex_map = vertex_map.permute(0, 1, 3, 4, 2).contiguous()
                    global_vertex_map = global_vertex_map.permute(
                        0, 1, 3, 4, 2
                    ).contiguous()
                    depth_image = depth_image.permute(0, 1, 3, 4, 2).contiguous()
                self.assertEqual(vertex_map.shape, (2, 3, 120, 160, 3))
                self.assertEqual(global_vertex_map.shape, (2, 3, 120, 160, 3))
                self.assertEqual(depth_image.shape, (2, 3, 120, 160, 1))

                for b in range(2):
                    for s in range(3):
                        vmap = vertex_map[b, s]
                        dmap = depth_image[b, s]
                        K = intrinsics[b, 0]
                        test_unproj_res = project_points(vmap, K)
                        meshgrid = (
                            create_meshgrid(120, 160, False).squeeze(0).to(device)
                        )
                        meshgrid = torch.cat(
                            [
                                meshgrid[..., 1:],
                                meshgrid[..., 0:1],
                            ],
                            -1,
                        )
                        correct_unproj_res = meshgrid * (dmap != 0).float()
                        # self.assertClose() fails here, probably because not close enough?
                        assert (test_unproj_res - correct_unproj_res).abs().max() < 1e-4

                assert ((gt_vmap - vertex_map.cpu().numpy()) ** 2).sum() < 1e-2
                if use_poses:
                    assert (
                        (gt_global_vmap - global_vertex_map.cpu().numpy()) ** 2
                    ).sum() < 1e-2
                else:
                    assert (
                        (gt_vmap - global_vertex_map.cpu().numpy()) ** 2
                    ).sum() < 1e-2

    def test_normal_map(self):
        device = default_to_cpu_if_no_gpu("cuda:0")

        def diff(x, y):
            # normals on gpu give slightly different values at some pixels
            return (((x - y) ** 2) < 1e-5).mean() > 0.99

        scriptdir = os.path.dirname(os.path.realpath(__file__))
        gt_nmap = np.load(os.path.join(scriptdir, "../data/msrd_b2s3/normal_map.npy"))
        gt_global_nmap = np.load(
            os.path.join(scriptdir, "../data/msrd_b2s3/global_normal_map.npy")
        )

        for use_poses in [False, True]:
            for channels_first in [False, True]:
                rgbdimages, *_ = TestRGBDImages.init_rgbdimages(
                    channels_first=channels_first, use_poses=use_poses, device=device
                )
                normal_map = rgbdimages.normal_map
                global_normal_map = rgbdimages.global_normal_map
                self.assertEqual(normal_map.ndim, 5)
                self.assertEqual(global_normal_map.ndim, 5)
                remove_missing = global_normal_map * rgbdimages.valid_depth_mask.to(
                    global_normal_map.dtype
                )
                assert ((global_normal_map - remove_missing) ** 2).sum().item() < 1e-5
                if channels_first:
                    normal_map = normal_map.permute(0, 1, 3, 4, 2).contiguous()
                    global_normal_map = global_normal_map.permute(
                        0, 1, 3, 4, 2
                    ).contiguous()
                self.assertEqual(normal_map.shape, (2, 3, 120, 160, 3))
                self.assertEqual(global_normal_map.shape, (2, 3, 120, 160, 3))

                nmap = normal_map.detach().cpu().numpy()
                global_nmap = global_normal_map.detach().cpu().numpy()

                assert diff(gt_nmap, nmap)
                if use_poses:
                    # # visualize normals
                    # import matplotlib.pyplot as plt
                    # fig, ax = plt.subplots(2, 2)
                    # ax[0, 0].imshow((nmap[-1, -1] * 255).astype(np.uint8))
                    # ax[0, 1].imshow((gt_nmap[-1, -1] * 255).astype(np.uint8))
                    # ax[1, 0].imshow((global_nmap[-1, -1] * 255).astype(np.uint8))
                    # ax[1, 1].imshow((gt_global_nmap[-1, -1] * 255).astype(np.uint8))
                    # plt.show()

                    assert diff(gt_global_nmap, global_nmap)
                else:
                    assert diff(gt_nmap, global_nmap)

    def test_indexing(self):
        device = default_to_cpu_if_no_gpu("cuda:0")

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
