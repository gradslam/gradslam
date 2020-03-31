import logging
import unittest
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.testing import assert_allclose
from torch.utils.data import DataLoader

import gradslam.datasets.datautils as datautils
from gradslam.datasets.scannet import Scannet

SCANNET_ROOT = "/Users/Soroosh/Downloads/data/ScanNet-gradSLAM/extractions/scans"
SCANNET_META_ROOT = (
    "/Users/Soroosh/Downloads/data/ScanNet-gradSLAM/extractions/sequence_associations"
)

# Tests below can only be run if a Scannet dataset is available
SCANNET_NOT_FOUND = "Scannet scans not found at default location: {}".format(
    SCANNET_ROOT
)
SCANNET_META_NOT_FOUND = "Scannet metadata not found at default location: {}".format(
    SCANNET_META_ROOT
)


class TestScannet(unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)

    @staticmethod
    def init_scannet(
        scenes=None,
        start=0,
        end=-1,
        height=480,
        width=640,
        seg_classes="scannet20",
        channels_first=False,
        normalize_color=False,
        return_depth=True,
        return_intrinsics=True,
        return_pose=True,
        return_transform=True,
        return_names=True,
        return_labels=True,
        batch_size=4,
        shuffle=True,
    ):
        """
        Function to initialize Scannet dataset

        Args:
            start (int): Index of the frame from which to start for every sequence. Default: 0
            end (int): Index of the frame at which to end for every sequence. Default: -1
            h (int): Spatial height for the frames. Default: 480
            w (int): Spatial width for the frames. Default: 640
            seg_classes (str): The palette of classes that the network should learn. Either `"nyu40"` or `"scannet20"`.
                Default: `"scannet20"`
            channels_first (bool): If True, will use channels first representation :math:`(B, L, C, H, W)` for images
            `(batchsize, sequencelength, width, height, channels)`. If False, will use channels last representation
            :math:`(B, L, H, W, C)`. Default: False
            normalize_color (bool): Normalize color to range :math:`[0 1]` or leave it at range :math:`[0 255]`.
                Default: False
            return_depth (bool): Determines whether to return depths. Default: True
            return_intrinsics (bool): Determines whether to return intrinsics. Default: True
            return_pose (bool): Determines whether to return poses. Default: True
            return_transform (bool): Determines whether to return transforms w.r.t. initial pose being transformed to be
                identity. Default: True
            return_names (bool): Determines whether to return sequence names. Default: True
            return_labels (bool): Determines whether to return segmentation labels. Default: True

        """
        dataset = Scannet(
            basedir=SCANNET_ROOT,
            seqmetadir=SCANNET_META_ROOT,
            scenes=scenes,
            start=start,
            end=end,
            height=height,
            width=width,
            seg_classes=seg_classes,
            channels_first=channels_first,
            normalize_color=normalize_color,
            return_depth=return_depth,
            return_intrinsics=return_intrinsics,
            return_pose=return_pose,
            return_transform=return_transform,
            return_names=return_names,
            return_labels=return_labels,
        )
        loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,)
        return dataset, loader

    @pytest.mark.skipif(not Path(SCANNET_ROOT).exists(), reason=SCANNET_NOT_FOUND)
    @pytest.mark.skipif(
        not Path(SCANNET_META_ROOT).exists(), reason=SCANNET_META_NOT_FOUND
    )
    def test_simple_shapes(self):
        start = 0
        B = 8
        L = 6
        H = 480
        W = 640
        channels_first = False
        _, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=start + L,
            height=H,
            width=W,
            channels_first=channels_first,
            scenes=("scene0005_00", "scene0010_00",),
        )
        colors, depths, intrinsics, poses, transforms, names, labels = next(
            iter(loader)
        )

        # test types
        self.assertTrue(torch.is_tensor(colors))
        self.assertTrue(torch.is_tensor(depths))
        self.assertTrue(torch.is_tensor(intrinsics))
        self.assertTrue(torch.is_tensor(poses))
        self.assertTrue(torch.is_tensor(transforms))
        self.assertTrue(isinstance(names, tuple))
        self.assertTrue(torch.is_tensor(labels))

        # test shapes
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(labels.shape, (B, L, H, W, 1))

        start = 5
        channels_first = True
        _, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=start + L,
            height=H,
            width=W,
            channels_first=channels_first,
            scenes=("scene0005_00", "scene0010_00",),
        )
        colors, depths, intrinsics, poses, transforms, names, labels = next(
            iter(loader)
        )

        # test shapes
        self.assertEqual(colors.shape, (B, L, 3, H, W))
        self.assertEqual(depths.shape, (B, L, 1, H, W))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(labels.shape, (B, L, H, W, 1))

    @pytest.mark.skipif(not Path(SCANNET_ROOT).exists(), reason=SCANNET_NOT_FOUND)
    @pytest.mark.skipif(
        not Path(SCANNET_META_ROOT).exists(), reason=SCANNET_META_NOT_FOUND
    )
    def test_full_dataset_shapes(self):
        start = 0
        B = 4
        H = 240
        W = 320
        channels_first = False
        dataset, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=-1,
            height=H,
            width=W,
            channels_first=channels_first,
            scenes=None,
        )
        colors, depths, intrinsics, poses, transforms, names, labels = next(
            iter(loader)
        )

        # test shapes
        self.assertTrue(dataset.__len__() > 100)
        self.assertEqual(colors.shape, (B, 16, H, W, 3))
        self.assertEqual(depths.shape, (B, 16, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, 16, 4, 4))
        self.assertEqual(transforms.shape, (B, 16, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(labels.shape, (B, 16, H, W, 1))

        start = 1
        channels_first = True
        dataset, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=-1,
            height=H,
            width=W,
            channels_first=channels_first,
            scenes=None,
        )
        colors, depths, intrinsics, poses, transforms, names, labels = next(
            iter(loader)
        )

        # test shapes
        self.assertTrue(dataset.__len__() > 100)
        self.assertEqual(colors.shape, (B, 15, 3, H, W))
        self.assertEqual(depths.shape, (B, 15, 1, H, W))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, 15, 4, 4))
        self.assertEqual(transforms.shape, (B, 15, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(labels.shape, (B, 15, H, W, 1))

    @pytest.mark.skipif(not Path(SCANNET_ROOT).exists(), reason=SCANNET_NOT_FOUND)
    @pytest.mark.skipif(
        not Path(SCANNET_META_ROOT).exists(), reason=SCANNET_META_NOT_FOUND
    )
    def test_dataset_values(self):
        start = 10
        B = 2
        L = 4
        H = 240
        W = 320
        channels_first = False
        normalize_color = False
        seg_classes = "scannet20"
        _, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=start + L,
            height=H,
            width=W,
            normalize_color=normalize_color,
            seg_classes=seg_classes,
            scenes=("scene0010_00",),
            shuffle=False,
        )
        colors, depths, intrinsics, poses, transforms, names, labels = next(
            iter(loader)
        )

        # test color rgb range
        self.assertTrue(colors.max().item() <= 255)
        self.assertTrue(colors.max().item() > 100)
        self.assertTrue(colors.min().item() >= 0)
        prev_labels = labels

        normalize_color = True
        seg_classes = "nyu40"
        _, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=start + L,
            height=H,
            width=W,
            normalize_color=normalize_color,
            seg_classes=seg_classes,
            scenes=("scene0010_00",),
            shuffle=False,
        )
        colors, depths, intrinsics, poses, transforms, names, labels = next(
            iter(loader)
        )

        # test color rgb range
        self.assertTrue(colors.max().item() <= 1)
        self.assertTrue(colors.min().item() >= 0)
        # test depth range
        self.assertTrue(depths.min().item() >= 0)
        # test labels difference
        self.assertTrue((prev_labels - labels).abs().mean() > 10)

    @pytest.mark.skipif(not Path(SCANNET_ROOT).exists(), reason=SCANNET_NOT_FOUND)
    @pytest.mark.skipif(
        not Path(SCANNET_META_ROOT).exists(), reason=SCANNET_META_NOT_FOUND
    )
    def test_intrinsics_extrinsics(self):
        start = 3
        B = 1
        L = 4
        org_H = 480
        org_W = 640
        _, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=start + L,
            height=org_H,
            width=org_W,
            scenes=("scene0010_00",),
            shuffle=False,
            return_depth=False,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=False,
            return_labels=False,
        )
        _, org_intrinsics, org_poses, org_transforms = next(iter(loader))

        H = 240
        W = 320
        _, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=start + L,
            height=H,
            width=W,
            scenes=("scene0010_00",),
            shuffle=False,
            return_depth=False,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=False,
            return_labels=False,
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

        # synthetic test intrinsics scaling
        syn_intrinsics = np.array(
            [[10, 0, 5, 0], [0, 4, 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        syn_out = datautils.scale_intrinsics(syn_intrinsics, w_ratio=0.2, h_ratio=0.5)
        gt_out = np.array([[2, 0, 1, 0], [0, 2, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.assertTrue(np.sum(abs(syn_out - gt_out)) < 0.1)
        syn_intrinsics_recovered = datautils.scale_intrinsics(
            syn_out, w_ratio=5.0, h_ratio=2.0
        )
        self.assertTrue(np.sum(abs(syn_intrinsics - syn_intrinsics_recovered)) < 0.1)

        # test extrinsics
        self.assertTrue(abs(poses - org_poses).sum() < 1)
        self.assertTrue(abs(transforms - org_transforms).sum() < 1)

        first_pose = poses[0][0]
        for i in range(1, L):
            recovered_pose = first_pose.mm(transforms[0][i])
            self.assertTrue(abs(poses[0][i] - recovered_pose).sum() < 1)

    @pytest.mark.skipif(not Path(SCANNET_ROOT).exists(), reason=SCANNET_NOT_FOUND)
    @pytest.mark.skipif(
        not Path(SCANNET_META_ROOT).exists(), reason=SCANNET_META_NOT_FOUND
    )
    def test_partial_outputs(self):
        start = 3
        B = 2
        L = 4
        H = 240
        W = 320
        channels_first = False

        # all outputs
        _, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=start + L,
            height=H,
            width=W,
            channels_first=channels_first,
            scenes=("scene0010_00",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=True,
            return_labels=True,
        )
        colors, depths, intrinsics, poses, transforms, names, labels = next(
            iter(loader)
        )
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(labels.shape, (B, L, H, W, 1))

        # no depth
        depths = None
        _, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=start + L,
            height=H,
            width=W,
            channels_first=channels_first,
            scenes=("scene0010_00",),
            return_depth=False,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=True,
            return_labels=True,
        )
        colors, intrinsics, poses, transforms, names, labels = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(labels.shape, (B, L, H, W, 1))

        # no intrinsics
        intrinsics = None
        _, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=start + L,
            height=H,
            width=W,
            channels_first=channels_first,
            scenes=("scene0010_00",),
            return_depth=True,
            return_intrinsics=False,
            return_pose=True,
            return_transform=True,
            return_names=True,
            return_labels=True,
        )
        colors, depths, poses, transforms, names, labels = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(labels.shape, (B, L, H, W, 1))

        # no pose
        poses = None
        _, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=start + L,
            height=H,
            width=W,
            channels_first=channels_first,
            scenes=("scene0010_00",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=False,
            return_transform=True,
            return_names=True,
            return_labels=True,
        )
        colors, depths, intrinsics, transforms, names, labels = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(labels.shape, (B, L, H, W, 1))

        # no transform
        transforms = None
        _, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=start + L,
            height=H,
            width=W,
            channels_first=channels_first,
            scenes=("scene0010_00",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=True,
            return_transform=False,
            return_names=True,
            return_labels=True,
        )
        colors, depths, intrinsics, poses, names, labels = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)
        self.assertEqual(labels.shape, (B, L, H, W, 1))

        # no names
        names = None
        _, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=start + L,
            height=H,
            width=W,
            channels_first=channels_first,
            scenes=("scene0010_00",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=False,
            return_labels=True,
        )
        colors, depths, intrinsics, poses, transforms, labels = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(labels.shape, (B, L, H, W, 1))

        # no labels
        labels = None
        _, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=start + L,
            height=H,
            width=W,
            channels_first=channels_first,
            scenes=("scene0010_00",),
            return_depth=True,
            return_intrinsics=True,
            return_pose=True,
            return_transform=True,
            return_names=True,
            return_labels=False,
        )
        colors, depths, intrinsics, poses, transforms, names = next(iter(loader))
        self.assertEqual(colors.shape, (B, L, H, W, 3))
        self.assertEqual(depths.shape, (B, L, H, W, 1))
        self.assertEqual(intrinsics.shape, (B, 1, 4, 4))
        self.assertEqual(poses.shape, (B, L, 4, 4))
        self.assertEqual(transforms.shape, (B, L, 4, 4))
        self.assertEqual(len(names), B)

    @pytest.mark.skipif(not Path(SCANNET_ROOT).exists(), reason=SCANNET_NOT_FOUND)
    @pytest.mark.skipif(
        not Path(SCANNET_META_ROOT).exists(), reason=SCANNET_META_NOT_FOUND
    )
    def test_create_label_image(self):
        start = 3
        B = 2
        L = 4
        H = 240
        W = 320
        # just test if any errors when creating label images
        seg_classes = "nyu40"
        dataset, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=start + L,
            height=H,
            width=W,
            seg_classes=seg_classes,
            scenes=("scene0010_00",),
            shuffle=False,
        )
        _, _, _, _, _, _, labels = next(iter(loader))
        label_sample = labels[0].contiguous().detach().cpu().numpy().astype(np.uint8)
        encoding_list = dataset.color_encoding
        encoding_list = [v for _, v in encoding_list.items()]
        label_image = datautils.create_label_image(
            label_sample[0].squeeze(), encoding_list
        )

        seg_classes = "scannet20"
        dataset, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=start + L,
            height=H,
            width=W,
            seg_classes=seg_classes,
            scenes=("scene0010_00",),
            shuffle=False,
        )
        _, _, _, _, _, _, labels = next(iter(loader))
        label_sample = labels[0].contiguous().detach().cpu().numpy().astype(np.uint8)
        encoding_list = dataset.color_encoding
        encoding_list = [v for _, v in encoding_list.items()]
        label_image = datautils.create_label_image(
            label_sample[0].squeeze(), encoding_list
        )

    @pytest.mark.skipif(not Path(SCANNET_ROOT).exists(), reason=SCANNET_NOT_FOUND)
    @pytest.mark.skipif(
        not Path(SCANNET_META_ROOT).exists(), reason=SCANNET_META_NOT_FOUND
    )
    def test_bad_inputs(self):
        start = 0
        end = 20
        channels_first = False
        with self.assertRaises(ValueError):
            _, _ = TestScannet.init_scannet(
                start=start, end=end, scenes=("scene0010_00",)
            )

        start = -2
        end = 3
        with self.assertRaises(ValueError):
            _, _ = TestScannet.init_scannet(
                start=start, end=end, scenes=("scene0010_00",)
            )

        start = 5
        end = 3
        with self.assertRaises(ValueError):
            _, _ = TestScannet.init_scannet(
                start=start, end=end, scenes=("scene0010_00",)
            )

        start = 0
        end = 3
        scenes = "bad_file_name"
        with self.assertRaises(ValueError):
            _, _ = TestScannet.init_scannet(start=start, end=end, scenes=scenes)

        start = 0
        end = 3
        scenes = ["scene0009_00", "scene0010_00"]
        with self.assertRaises(TypeError):
            _, _ = TestScannet.init_scannet(start=start, end=end, scenes=scenes)

    def test_channels_first(self):
        a = np.random.randn(480, 640, 3)
        self.assertEqual(datautils.channels_first(a).shape, (3, 480, 640))

        a = np.random.randn(12, 5, 480, 640, 3)
        self.assertEqual(datautils.channels_first(a).shape, (12, 5, 3, 480, 640))

    def test_transforms(self):
        start = 0
        end = 4
        B = 1
        H = 240
        W = 320
        channels_first = False
        dataset, loader = TestScannet.init_scannet(
            batch_size=B,
            start=start,
            end=end,
            height=H,
            width=W,
            channels_first=channels_first,
            scenes=None,
        )
        colors, depths, intrinsics, poses, transforms, names, labels = next(
            iter(loader)
        )

        assert_allclose(poses[0][0], torch.eye(4))
        assert_allclose(poses[0][0], transforms[0][0])
        assert_allclose(poses[0][1], transforms[0][1].mm(transforms[0][0]))
        assert_allclose(
            poses[0][2],
            transforms[0][2].mm(transforms[0][1].mm(transforms[0][0])),
            rtol=0.001,
            atol=1e-04,
        )
        assert_allclose(
            poses[0][3],
            transforms[0][3].mm(
                transforms[0][2].mm(transforms[0][1].mm(transforms[0][0]))
            ),
            rtol=0.001,
            atol=1e-04,
        )

    def test_scale_intrinsics(self):
        intrinsics0 = torch.Tensor(
            [
                [577.87, 0.0, 319.5, 0.0],
                [0.0, 577.87, 239.5, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        intrinsics1 = torch.Tensor(
            [
                [377.87, 0.0, 219.5, 0.0],
                [0.0, 377.87, 139.5, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        intrinsics = torch.stack([intrinsics0, intrinsics1], 0)
        assert_allclose(
            datautils.scale_intrinsics(intrinsics, 2, 2)[0],
            datautils.scale_intrinsics(intrinsics0, 2, 2),
        )
        assert_allclose(
            datautils.scale_intrinsics(intrinsics, 2, 2)[1],
            datautils.scale_intrinsics(intrinsics1, 2, 2),
        )


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger("config")
    logger.setLevel(logging.DEBUG)
    unittest.main()
