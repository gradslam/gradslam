import os
import logging
import unittest
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.testing import assert_allclose
from torch.utils.data import DataLoader

from gradslam.datasets.tum import TUM

TUM_ROOT = "G:/Datasets/TUM"

# Tests below can only be run if a TUM dataset is available
TUM_NOT_FOUND = "TUM dataset not found at default location: {}".format(TUM_ROOT)


class TestTUM(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)

    @staticmethod
    def init_TUM(
        basedir=TUM_ROOT,
        sequences=None,
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
        return_timestamps=True,
        batch_size=4,
        shuffle=False,
    ):
        """
        Initializes TUM dataset
        """

        dataset = TUM(
            basedir=basedir,
            sequences=sequences,
            seqlen=seqlen,
            dilation=dilation,
            stride=stride,
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
            return_timestamps=return_timestamps,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        return dataset, loader

    @pytest.mark.skipif(not Path(TUM_ROOT).exists(), reason=TUM_NOT_FOUND)
    def test_simple_shapes(self):
        B = 8
        L = 6
        H = 480
        W = 640
        channels_first = False
        dataset, loader = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            sequences=(
                "rgbd_dataset_freiburg1_rpy",
                "rgbd_dataset_freiburg1_xyz",
            ),
        )
        colors, depths, intrinsics, poses, transforms, names, timestamps = next(
            iter(loader)
        )

        # test types
        self.assertTrue(torch.is_tensor(colors))
        self.assertTrue(torch.is_tensor(depths))
        self.assertTrue(torch.is_tensor(intrinsics))
        self.assertTrue(torch.is_tensor(poses))
        self.assertTrue(torch.is_tensor(transforms))
        self.assertTrue(isinstance(names, tuple))
        self.assertTrue(isinstance(timestamps, tuple))

        # test shapes
        self.assertTrue(len(dataset) > 100)
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(len(timestamps), B)

        L = 12
        channels_first = True
        dataset, loader = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            sequences=("rgbd_dataset_freiburg1_xyz",),
        )
        colors, depths, intrinsics, poses, transforms, names, timestamps = next(
            iter(loader)
        )

        # test shapes
        self.assertTrue(len(dataset) > 50)
        self.assertEqual(colors.shape, (B, L, 3, H, W))
        self.assertEqual(depths.shape, (B, L, 1, H, W))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(len(timestamps), B)

        old_dataset_len = len(dataset)
        dataset, loader = TestTUM.init_TUM(
            batch_size=B,
            start=L + L // 2,  # len(dataset) is determined by #matches in TUM
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            sequences=("rgbd_dataset_freiburg1_xyz",),
        )
        self.assertTrue(old_dataset_len - len(dataset) == 1)

        dataset, loader = TestTUM.init_TUM(
            batch_size=B,
            end=int((old_dataset_len * L) - 2),
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            sequences=("rgbd_dataset_freiburg1_xyz",),
        )
        self.assertTrue(old_dataset_len - len(dataset) == 1)

    @pytest.mark.skipif(not Path(TUM_ROOT).exists(), reason=TUM_NOT_FOUND)
    def test_extractions(self):
        B = 8
        L = 2
        D = 9
        S = 5
        H = 480
        W = 640
        channels_first = False
        dataset1, loader1 = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            dilation=D,
            stride=S,
            height=H,
            width=W,
            channels_first=channels_first,
            sequences=("rgbd_dataset_freiburg1_xyz",),
        )
        colors1, depths1, intrinsics1, poses1, transforms1, names1, timestamps1 = next(
            iter(loader1)
        )

        D = 4
        L = 3
        dataset2, loader2 = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            dilation=D,
            stride=S,
            height=H,
            width=W,
            channels_first=channels_first,
            sequences=("rgbd_dataset_freiburg1_xyz",),
        )
        colors2, depths2, intrinsics2, poses2, transforms2, names2, timestamps2 = next(
            iter(loader2)
        )

        assert_allclose(colors1[:, 0], colors2[:, 0])
        assert_allclose(depths1[:, 0], depths2[:, 0])
        assert_allclose(poses1[:, 0], poses2[:, 0])
        assert_allclose(transforms1[:, 0], transforms2[:, 0])

        assert_allclose(colors1[:, 1], colors2[:, 2])
        assert_allclose(depths1[:, 1], depths2[:, 2])
        assert_allclose(poses1[:, 1], poses2[:, 2])
        assert_allclose(transforms1[:, 1], transforms2[:, 1] @ transforms2[:, 2])

    @pytest.mark.skipif(not Path(TUM_ROOT).exists(), reason=TUM_NOT_FOUND)
    def test_dataset_values(self):
        B = 2
        L = 4
        H = 240
        W = 320
        channels_first = False
        normalize_color = False
        _, loader = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            normalize_color=normalize_color,
            sequences=("rgbd_dataset_freiburg1_xyz",),
            shuffle=False,
        )
        colors, depths, intrinsics, poses, transforms, names, timestamps = next(
            iter(loader)
        )

        # test color rgb range
        self.assertTrue(colors.max().item() <= 255)
        self.assertTrue(colors.max().item() > 100)
        self.assertTrue(colors.min().item() >= 0)

        normalize_color = True
        _, loader = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            normalize_color=normalize_color,
            sequences=("rgbd_dataset_freiburg1_xyz",),
            shuffle=False,
        )
        colors, depths, intrinsics, poses, transforms, names, timestamps = next(
            iter(loader)
        )

        # test color rgb range
        self.assertTrue(colors.max().item() <= 1)
        self.assertTrue(colors.min().item() >= 0)
        # test depth range
        self.assertTrue(depths.min().item() >= 0)

    @pytest.mark.skipif(not Path(TUM_ROOT).exists(), reason=TUM_NOT_FOUND)
    def test_intrinsics_extrinsics(self):
        B = 1
        L = 4
        org_H = 480
        org_W = 640
        _, loader = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            height=org_H,
            width=org_W,
            sequences=("rgbd_dataset_freiburg1_xyz",),
            shuffle=False,
            return_depth=False,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=False,
            return_timestamps=False,
        )
        _, org_intrinsics, org_poses, org_transforms = next(iter(loader))

        H = 240
        W = 320
        _, loader = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            sequences=("rgbd_dataset_freiburg1_xyz",),
            shuffle=False,
            return_depth=False,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=False,
            return_timestamps=False,
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

    @pytest.mark.skipif(not Path(TUM_ROOT).exists(), reason=TUM_NOT_FOUND)
    def test_partial_outputs(self):
        B = 2
        L = 4
        H = 240
        W = 320
        channels_first = False

        # all outputs
        _, loader = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            sequences=("rgbd_dataset_freiburg1_xyz",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=True,
            return_timestamps=True,
        )
        colors, depths, intrinsics, poses, transforms, names, timestamps = next(
            iter(loader)
        )
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(len(timestamps), B)

        # no depth
        depths = None
        _, loader = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            sequences=("rgbd_dataset_freiburg1_xyz",),
            return_depth=False,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=True,
            return_timestamps=True,
        )
        colors, intrinsics, poses, transforms, names, timestamps = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(len(timestamps), B)

        # no intrinsics
        intrinsics = None
        _, loader = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            sequences=("rgbd_dataset_freiburg1_xyz",),
            return_depth=True,
            return_intrinsics=False,
            return_pose=True,
            return_transform=True,
            return_names=True,
            return_timestamps=True,
        )
        colors, depths, poses, transforms, names, timestamps = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(len(timestamps), B)

        # no pose
        poses = None
        _, loader = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            sequences=("rgbd_dataset_freiburg1_xyz",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=False,
            return_transform=True,
            return_names=True,
            return_timestamps=True,
        )
        colors, depths, intrinsics, transforms, names, timestamps = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(len(timestamps), B)

        # no transform
        transforms = None
        _, loader = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            sequences=("rgbd_dataset_freiburg1_xyz",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=True,
            return_transform=False,
            return_names=True,
            return_timestamps=True,
        )
        colors, depths, intrinsics, poses, names, timestamps = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(len(timestamps), B)

        # no names
        names = None
        _, loader = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            sequences=("rgbd_dataset_freiburg1_xyz",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=False,
            return_timestamps=True,
        )
        colors, depths, intrinsics, poses, transforms, timestamps = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(timestamps), B)

        # no timestamps
        _, loader = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            sequences=("rgbd_dataset_freiburg1_xyz",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=True,
            return_timestamps=False,
        )
        colors, depths, intrinsics, poses, transforms, names = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)

        # no poses and transforms (don't load poses)
        dataset, loader = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            sequences=("rgbd_dataset_freiburg1_xyz",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=False,
            return_transform=False,
            return_names=True,
            return_timestamps=True,
        )
        colors, depths, intrinsics, names, timestamps = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(len(timestamps), B)
        self.assertEqual(len(dataset.poses), 0)

    @pytest.mark.skipif(not Path(TUM_ROOT).exists(), reason=TUM_NOT_FOUND)
    def test_type_errors(self):
        with self.assertRaises(TypeError):
            _, _ = TestTUM.init_TUM(seqlen=0.5)

        with self.assertRaises(TypeError):
            _, _ = TestTUM.init_TUM(dilation=0.5)

        with self.assertRaises(TypeError):
            _, _ = TestTUM.init_TUM(stride=0.5)

        with self.assertRaises(TypeError):
            _, _ = TestTUM.init_TUM(start=0.5)

        with self.assertRaises(TypeError):
            _, _ = TestTUM.init_TUM(end=0.5)

        with self.assertRaises(TypeError):
            _, _ = TestTUM.init_TUM(end=0.5)

        with self.assertRaises(TypeError):
            _, _ = TestTUM.init_TUM(sequences=["rgbd_dataset_freiburg1_xyz"])

    @pytest.mark.skipif(not Path(TUM_ROOT).exists(), reason=TUM_NOT_FOUND)
    def test_value_errors(self):
        with self.assertRaises(ValueError):
            _, _ = TestTUM.init_TUM(seqlen=-5)

        with self.assertRaises(ValueError):
            _, _ = TestTUM.init_TUM(dilation=-5)

        with self.assertRaises(ValueError):
            _, _ = TestTUM.init_TUM(stride=-5)

        with self.assertRaises(ValueError):
            _, _ = TestTUM.init_TUM(start=-5)

        with self.assertRaises(ValueError):
            _, _ = TestTUM.init_TUM(start=5, end=3)

        with self.assertRaises(ValueError):
            _, _ = TestTUM.init_TUM(start=5, end=5)

        with self.assertRaises(ValueError):
            _, _ = TestTUM.init_TUM(sequences="trajfile")

        with self.assertRaises(ValueError):
            _, _ = TestTUM.init_TUM(basedir=os.path.dirname(os.path.normpath(TUM_ROOT)))

        with pytest.warns(UserWarning):
            _, _ = TestTUM.init_TUM(end=50000)

    @pytest.mark.skipif(not Path(TUM_ROOT).exists(), reason=TUM_NOT_FOUND)
    def test_transforms(self):
        L = 4
        B = 1
        H = 240
        W = 320
        channels_first = False
        dataset, loader = TestTUM.init_TUM(
            batch_size=B,
            seqlen=L,
            height=H,
            width=W,
            channels_first=channels_first,
            sequences=None,
        )
        colors, depths, intrinsics, poses, transforms, names, timestamps = next(
            iter(loader)
        )

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
