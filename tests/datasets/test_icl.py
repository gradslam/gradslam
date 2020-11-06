import os
import logging
import unittest
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.testing import assert_allclose
from torch.utils.data import DataLoader

from gradslam.datasets.icl import ICL

ICL_ROOT = "G:/Datasets/ICL"

# Tests below can only be run if a ICL dataset is available
ICL_NOT_FOUND = "ICL dataset not found at default location: {}".format(ICL_ROOT)


class TestICL(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)

    @staticmethod
    def init_ICL(
        basedir=ICL_ROOT,
        trajectories=None,
        seqlen=4,
        dilation=None,
        stride=None,
        start=0,
        end=None,
        height=480,
        width=640,
        channels_first=False,
        normalize_color=False,
        return_depth=True,
        return_intrinsics=True,
        return_pose=True,
        return_transform=True,
        return_names=True,
        batch_size=4,
        shuffle=False,
    ):
        """
        Initializes ICL dataset
        """

        dataset = ICL(
            basedir=basedir,
            trajectories=trajectories,
            seqlen=seqlen,
            stride=stride,
            dilation=dilation,
            start=start,
            end=end,
            height=height,
            width=width,
            channels_first=channels_first,
            normalize_color=normalize_color,
            return_depth=return_depth,
            return_intrinsics=return_intrinsics,
            return_pose=return_pose,
            return_transform=return_transform,
            return_names=return_names,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return dataset, loader

    @pytest.mark.skipif(not Path(ICL_ROOT).exists(), reason=ICL_NOT_FOUND)
    def test_simple_shapes(self):
        B = 8
        L = 6
        H = 480
        W = 640
        channels_first = False
        dataset, loader = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            trajectories=(
                "living_room_traj2_frei_png",
                "living_room_traj3_frei_png",
            ),
        )
        colors, depths, intrinsics, poses, transforms, names = next(iter(loader))

        # test types
        self.assertTrue(torch.is_tensor(colors))
        self.assertTrue(torch.is_tensor(depths))
        self.assertTrue(torch.is_tensor(intrinsics))
        self.assertTrue(torch.is_tensor(poses))
        self.assertTrue(torch.is_tensor(transforms))
        self.assertTrue(isinstance(names, tuple))

        # test shapes
        self.assertTrue(len(dataset) > 100)
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)

        L = 12
        channels_first = True
        dataset, loader = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            trajectories=("living_room_traj0_frei_png",),
        )
        colors, depths, intrinsics, poses, transforms, names = next(iter(loader))

        # test shapes
        self.assertTrue(len(dataset) > 100)
        self.assertEqual(colors.shape, (B, L, 3, H, W))
        self.assertEqual(depths.shape, (B, L, 1, H, W))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)

        old_dataset_len = len(dataset)
        dataset, loader = TestICL.init_ICL(
            batch_size=B,
            start=L,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            trajectories=("living_room_traj0_frei_png",),
        )
        self.assertTrue(old_dataset_len - len(dataset) == 1)

        dataset, loader = TestICL.init_ICL(
            batch_size=B,
            end=int((old_dataset_len * L) - 2),
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            trajectories=("living_room_traj0_frei_png",),
        )
        self.assertTrue(old_dataset_len - len(dataset) == 1)

    @pytest.mark.skipif(not Path(ICL_ROOT).exists(), reason=ICL_NOT_FOUND)
    def test_extractions(self):
        B = 8
        L = 2
        D = 9
        S = 5
        H = 480
        W = 640
        channels_first = False
        dataset1, loader1 = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            dilation=D,
            stride=S,
            height=H,
            width=W,
            channels_first=channels_first,
            trajectories=("living_room_traj0_frei_png",),
        )
        colors1, depths1, intrinsics1, poses1, transforms1, names1 = next(iter(loader1))

        D = 4
        L = 3
        dataset2, loader2 = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            dilation=D,
            stride=S,
            height=H,
            width=W,
            channels_first=channels_first,
            trajectories=("living_room_traj0_frei_png",),
        )
        colors2, depths2, intrinsics2, poses2, transforms2, names2 = next(iter(loader2))

        assert_allclose(colors1[:, 0], colors2[:, 0])
        assert_allclose(depths1[:, 0], depths2[:, 0])
        assert_allclose(poses1[:, 0], poses2[:, 0])
        assert_allclose(transforms1[:, 0], transforms2[:, 0])

        assert_allclose(colors1[:, 1], colors2[:, 2])
        assert_allclose(depths1[:, 1], depths2[:, 2])
        assert_allclose(poses1[:, 1], poses2[:, 2])
        assert_allclose(transforms1[:, 1], transforms2[:, 1] @ transforms2[:, 2])

    @pytest.mark.skipif(not Path(ICL_ROOT).exists(), reason=ICL_NOT_FOUND)
    def test_dataset_values(self):
        B = 2
        L = 4
        H = 240
        W = 320
        channels_first = False
        normalize_color = False
        _, loader = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            normalize_color=normalize_color,
            trajectories=("living_room_traj3_frei_png",),
            shuffle=False,
        )
        colors, depths, intrinsics, poses, transforms, names = next(iter(loader))

        # test color rgb range
        self.assertTrue(colors.max().item() <= 255)
        self.assertTrue(colors.max().item() > 100)
        self.assertTrue(colors.min().item() >= 0)

        normalize_color = True
        _, loader = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            normalize_color=normalize_color,
            trajectories=("living_room_traj3_frei_png",),
            shuffle=False,
        )
        colors, depths, intrinsics, poses, transforms, names = next(iter(loader))

        # test color rgb range
        self.assertTrue(colors.max().item() <= 1)
        self.assertTrue(colors.min().item() >= 0)
        # test depth range
        self.assertTrue(depths.min().item() >= 0)

    @pytest.mark.skipif(not Path(ICL_ROOT).exists(), reason=ICL_NOT_FOUND)
    def test_intrinsics_extrinsics(self):
        B = 1
        L = 4
        org_H = 480
        org_W = 640
        _, loader = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            height=org_H,
            width=org_W,
            trajectories=("living_room_traj3_frei_png",),
            shuffle=False,
            return_depth=False,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=False,
        )
        _, org_intrinsics, org_poses, org_transforms = next(iter(loader))

        H = 240
        W = 320
        _, loader = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            trajectories=("living_room_traj3_frei_png",),
            shuffle=False,
            return_depth=False,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=False,
        )
        _, intrinsics, poses, transforms = next(iter(loader))
        self.assertEqual(org_intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(org_poses.shape, (B, L, 4, 4))
        self.assertEqual(org_transforms.shape, (B, L, 4, 4))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))

        # test intrinsics
        h_ratio = H / org_H
        w_ratio = W / org_W
        self.assertEqual(intrinsics[0, 0][0, 0] / org_intrinsics[0, 0][0, 0], w_ratio)
        self.assertEqual(intrinsics[0, 0][0, 2] / org_intrinsics[0, 0][0, 2], w_ratio)
        self.assertEqual(intrinsics[0, 0][1, 1] / org_intrinsics[0, 0][1, 1], h_ratio)
        self.assertEqual(intrinsics[0, 0][1, 2] / org_intrinsics[0, 0][1, 2], h_ratio)

        # test extrinsics
        self.assertTrue(abs(poses - org_poses).sum() < 1)
        self.assertTrue(abs(transforms - org_transforms).sum() < 1)

        first_pose = poses[0][0]
        for i in range(1, L):
            recovered_pose = first_pose.mm(transforms[0][i])
            self.assertTrue(abs(poses[0][i] - recovered_pose).sum() < 1)

    @pytest.mark.skipif(not Path(ICL_ROOT).exists(), reason=ICL_NOT_FOUND)
    def test_partial_outputs(self):
        B = 2
        L = 4
        H = 240
        W = 320
        channels_first = False

        # all outputs
        _, loader = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            trajectories=("living_room_traj2_frei_png",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=True,
        )
        colors, depths, intrinsics, poses, transforms, names = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)

        # no depth
        depths = None
        _, loader = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            trajectories=("living_room_traj2_frei_png",),
            return_depth=False,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=True,
        )
        colors, intrinsics, poses, transforms, names = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)

        # no intrinsics
        intrinsics = None
        _, loader = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            trajectories=("living_room_traj2_frei_png",),
            return_depth=True,
            return_intrinsics=False,
            return_pose=True,
            return_transform=True,
            return_names=True,
        )
        colors, depths, poses, transforms, names = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)

        # no pose
        poses = None
        _, loader = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            trajectories=("living_room_traj2_frei_png",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=False,
            return_transform=True,
            return_names=True,
        )
        colors, depths, intrinsics, transforms, names = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)

        # no transform
        transforms = None
        _, loader = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            trajectories=("living_room_traj2_frei_png",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=True,
            return_transform=False,
            return_names=True,
        )
        colors, depths, intrinsics, poses, names = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)

        # no names
        names = None
        _, loader = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            trajectories=("living_room_traj2_frei_png",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=False,
        )
        colors, depths, intrinsics, poses, transforms = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))

        # no poses and transforms (don't load poses)
        dataset, loader = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            trajectories=("living_room_traj2_frei_png",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=False,
            return_transform=False,
            return_names=True,
        )
        colors, depths, intrinsics, names = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(len(dataset.posemetas), 0)

    @pytest.mark.skipif(not Path(ICL_ROOT).exists(), reason=ICL_NOT_FOUND)
    def test_type_errors(self):
        with self.assertRaises(TypeError):
            _, _ = TestICL.init_ICL(seqlen=0.5)

        with self.assertRaises(TypeError):
            _, _ = TestICL.init_ICL(stride=0.5)

        with self.assertRaises(TypeError):
            _, _ = TestICL.init_ICL(dilation=0.5)

        with self.assertRaises(TypeError):
            _, _ = TestICL.init_ICL(start=0.5)

        with self.assertRaises(TypeError):
            _, _ = TestICL.init_ICL(end=0.5)

        with self.assertRaises(TypeError):
            _, _ = TestICL.init_ICL(end=0.5)

        with self.assertRaises(TypeError):
            _, _ = TestICL.init_ICL(trajectories=["living_room_traj2_frei_png"])

    @pytest.mark.skipif(not Path(ICL_ROOT).exists(), reason=ICL_NOT_FOUND)
    def test_value_errors(self):
        with self.assertRaises(ValueError):
            _, _ = TestICL.init_ICL(seqlen=-5)

        with self.assertRaises(ValueError):
            _, _ = TestICL.init_ICL(dilation=-5)

        with self.assertRaises(ValueError):
            _, _ = TestICL.init_ICL(stride=-5)

        with self.assertRaises(ValueError):
            _, _ = TestICL.init_ICL(start=-5)

        with self.assertRaises(ValueError):
            _, _ = TestICL.init_ICL(start=5, end=3)

        with self.assertRaises(ValueError):
            _, _ = TestICL.init_ICL(start=5, end=5)

        with self.assertRaises(ValueError):
            _, _ = TestICL.init_ICL(trajectories="trajfile")

        with self.assertRaises(ValueError):
            _, _ = TestICL.init_ICL(trajectories=("living_room_trajB_frei_png",))

        with self.assertRaises(ValueError):
            _, _ = TestICL.init_ICL(trajectories=("randomname_traj0_frei_png",))

        with self.assertRaises(ValueError):
            _, _ = TestICL.init_ICL(basedir=os.path.dirname(os.path.normpath(ICL_ROOT)))

        with pytest.warns(UserWarning):
            _, _ = TestICL.init_ICL(end=50000)

    @pytest.mark.skipif(not Path(ICL_ROOT).exists(), reason=ICL_NOT_FOUND)
    def test_transforms(self):
        L = 4
        B = 1
        H = 240
        W = 320
        channels_first = False
        dataset, loader = TestICL.init_ICL(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            trajectories=None,
        )
        colors, depths, intrinsics, poses, transforms, names = next(iter(loader))

        assert_allclose(poses[0][0], torch.eye(4))
        assert_allclose(poses[0][0], transforms[0][0])
        assert_allclose(poses[0][1], transforms[0][0].mm(transforms[0][1]))
        assert_allclose(
            poses[0][2],
            transforms[0][0].mm(transforms[0][1].mm(transforms[0][2])),
        )
        assert_allclose(
            poses[0][3],
            transforms[0][0].mm(
                transforms[0][1].mm(transforms[0][2].mm(transforms[0][3]))
            ),
        )


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger("config")
    logger.setLevel(logging.DEBUG)
    unittest.main()
