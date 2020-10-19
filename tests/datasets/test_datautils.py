import logging
import unittest

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose as np_assert_allclose
from torch.testing import assert_allclose

from gradslam.datasets.datautils import *


class TestDataUtils(unittest.TestCase):
    def setup(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)

    def test_normalize_image(self):
        self.setup()
        N = 4
        L = 6
        H = 240
        W = 320
        C = 3
        imsize = (N, L, H, W, C)

        # numpy
        img = np.random.uniform(0, 255, imsize).astype(np.uint8)
        norm_img = normalize_image(img)
        assert norm_img.dtype == np.float
        assert np.max(norm_img) < 1.000000001
        assert np.min(norm_img) > -1e-10
        assert np.sum(norm_img > 0.5) > 100
        assert np.sum(norm_img < 0.5) > 100

        # torch
        img = (torch.rand(*imsize) * 255).to(torch.uint8)
        norm_img = normalize_image(img)
        assert norm_img.dtype == torch.float
        assert torch.max(norm_img) < 1.000000001
        assert torch.min(norm_img) > -1e-10
        assert torch.sum(norm_img > 0.5) > 100
        assert torch.sum(norm_img < 0.5) > 100

        # type errors
        with pytest.raises(TypeError):
            normalize_image([0, 125, 255])

    def test_channels_first(self):
        self.setup()
        N = 4
        L = 6
        H = 240
        W = 320
        C = 3
        imsize1 = (H, W, C)
        imsize2 = (N, L, H, W, C)

        # numpy
        hwc_img = np.random.uniform(0, 255, imsize1).astype(np.uint8)
        chw_img = channels_first(hwc_img)
        assert hwc_img.dtype == chw_img.dtype
        assert hwc_img.shape == (H, W, C)
        assert chw_img.shape == (C, H, W)

        hwc_img = np.random.uniform(0, 255, imsize2).astype(np.uint8)
        chw_img = channels_first(hwc_img)
        assert hwc_img.dtype == chw_img.dtype
        assert hwc_img.shape == (N, L, H, W, C)
        assert chw_img.shape == (N, L, C, H, W)

        # torch
        hwc_img = (torch.rand(*imsize1) * 255).to(torch.uint8)
        chw_img = channels_first(hwc_img)
        assert hwc_img.dtype == chw_img.dtype
        assert hwc_img.shape == (H, W, C)
        assert chw_img.shape == (C, H, W)

        hwc_img = (torch.rand(*imsize2) * 255).to(torch.uint8)
        chw_img = channels_first(hwc_img)
        assert hwc_img.dtype == chw_img.dtype
        assert hwc_img.shape == (N, L, H, W, C)
        assert chw_img.shape == (N, L, C, H, W)

        # type errors
        with pytest.raises(TypeError):
            channels_first([0, 125, 255])

        # value errors
        with pytest.raises(ValueError):
            img = np.random.uniform(0, 255, (5, 10)).astype(np.uint8)
            channels_first(img)

        #  warnings
        with pytest.warns(UserWarning):
            hwc_img = np.random.uniform(0, 255, (2, 10, 3)).astype(np.uint8)
            chw_img = channels_first(hwc_img)
        assert hwc_img.dtype == chw_img.dtype
        assert hwc_img.shape == (2, 10, 3)
        assert chw_img.shape == (3, 2, 10)

    def test_scale_intrinsics(self):
        intrinsics0 = np.array(
            [
                [577.87, 0.0, 319.5, 0.0],
                [0.0, 577.87, 239.5, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        intrinsics1 = np.array(
            [
                [377.87, 0.0, 219.5, 0.0],
                [0.0, 377.87, 139.5, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        intrinsics = np.stack([intrinsics0, intrinsics1], 0)

        # numpy
        # tst1
        np.testing.assert_allclose(
            scale_intrinsics(intrinsics, 2, 2)[0],
            scale_intrinsics(intrinsics0, 2, 2),
        )
        np.testing.assert_allclose(
            scale_intrinsics(intrinsics, 2, 2)[1],
            scale_intrinsics(intrinsics1, 2, 2),
        )
        np.testing.assert_allclose(
            scale_intrinsics(intrinsics[:, :3, :3], 2, 2)[0],
            scale_intrinsics(intrinsics0[:3, :3], 2, 2),
        )
        np.testing.assert_allclose(
            scale_intrinsics(intrinsics[:, :3, :3], 2, 2)[1],
            scale_intrinsics(intrinsics1[:3, :3], 2, 2),
        )
        # tst2
        syn_intrinsics = np.array(
            [[10, 0, 5, 0], [0, 4, 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        syn_out = scale_intrinsics(syn_intrinsics, w_ratio=0.2, h_ratio=0.5)
        gt_out = np.array([[2, 0, 1, 0], [0, 2, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.assertTrue(np.sum(abs(syn_out - gt_out)) < 0.1)
        syn_intrinsics_recovered = scale_intrinsics(syn_out, w_ratio=5.0, h_ratio=2.0)
        self.assertTrue(np.sum(abs(syn_intrinsics - syn_intrinsics_recovered)) < 0.1)

        # torch
        # tst1
        intrinsics = torch.tensor(intrinsics)
        intrinsics0 = torch.tensor(intrinsics0)
        intrinsics1 = torch.tensor(intrinsics1)
        assert_allclose(
            scale_intrinsics(intrinsics[:, :3, :3], 2, 2)[0],
            scale_intrinsics(intrinsics0[:3, :3], 2, 2),
        )
        assert_allclose(
            scale_intrinsics(intrinsics[:, :3, :3], 2, 2)[1],
            scale_intrinsics(intrinsics1[:3, :3], 2, 2),
        )
        # tst2
        syn_intrinsics = torch.tensor(syn_intrinsics)
        syn_out = scale_intrinsics(syn_intrinsics, w_ratio=0.2, h_ratio=0.5)
        gt_out = torch.tensor(gt_out)
        self.assertTrue((syn_out - gt_out).abs().sum() < 0.1)
        syn_intrinsics_recovered = scale_intrinsics(syn_out, w_ratio=5.0, h_ratio=2.0)
        self.assertTrue((syn_intrinsics - syn_intrinsics_recovered).abs().sum() < 0.1)

        # type errors
        with pytest.raises(TypeError):
            scale_intrinsics("abc", 2, 2)

        # value errors
        with pytest.raises(ValueError):
            corrupted_intrinsics = torch.rand(5, 10, 4, 3)
            scale_intrinsics(corrupted_intrinsics, 2, 2)

        #  warnings
        with pytest.warns(UserWarning):
            corrupted_intrinsics = torch.rand(5, 10, 4, 4)
            scale_intrinsics(corrupted_intrinsics, 2, 2)

    def test_pointquaternion_to_homogeneous(self):
        def single_ptquat2transform(pq):
            tx, ty, tz = pq[:3]
            X, Y, Z, W = pq[3:7]

            sqw = W * W
            sqx = X * X
            sqy = Y * Y
            sqz = Z * Z

            invs = 1 / (sqx + sqy + sqz + sqw)
            m00 = (sqx - sqy - sqz + sqw) * invs
            m11 = (-sqx + sqy - sqz + sqw) * invs
            m22 = (-sqx - sqy + sqz + sqw) * invs

            tmp1 = X * Y
            tmp2 = Z * W
            m10 = 2.0 * (tmp1 + tmp2) * invs
            m01 = 2.0 * (tmp1 - tmp2) * invs

            tmp1 = X * Z
            tmp2 = Y * W
            m20 = 2.0 * (tmp1 - tmp2) * invs
            m02 = 2.0 * (tmp1 + tmp2) * invs
            tmp1 = Y * Z
            tmp2 = X * W
            m21 = 2.0 * (tmp1 + tmp2) * invs
            m12 = 2.0 * (tmp1 - tmp2) * invs

            m03 = tx
            m13 = ty
            m23 = tz

            m30 = 0
            m31 = 0
            m32 = 0
            m33 = 1

            transform = np.array(
                (
                    (m00, m01, m02, m03),
                    (m10, m11, m12, m13),
                    (m20, m21, m22, m23),
                    (m30, m31, m32, m33),
                )
            )
            if torch.is_tensor(pq):
                return torch.tensor(transform).float()
            else:
                return transform

        # 0-4 from livingRoom0.gt.freiburg
        pq0 = np.array((0, 0, -2.25, 0, 0, 0, 1)).astype(np.float32)
        pq1 = np.array(
            (
                0.000466347,
                0.00895357,
                -2.24935,
                -0.00101358,
                0.00052453,
                -0.000231475,
                0.999999,
            )
        ).astype(np.float32)
        pq2 = np.array(
            (
                -0.000154972,
                -0.000102997,
                -2.25066,
                -0.00465149,
                0.000380752,
                0.000400181,
                0.999989,
            )
        ).astype(np.float32)
        pq3 = np.array(
            (
                0.000739813,
                -0.00242257,
                -2.25076,
                -0.0040878,
                8.72709e-05,
                0.000833608,
                0.999991,
            )
        ).astype(np.float32)
        pq = np.stack([pq0, pq1, pq2, pq3], 0)
        syn_pq = np.array(
            [0, 0, 0, 0, 0, 0.7071, 0.7071]
        )  # 90 degree rotation about the z axis
        syn_gt = np.array(
            (
                (0.0, -1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
        ).astype(
            np.float32
        )  # 90 degree rotation about the z axis

        # numpy
        # tst1
        out = pointquaternion_to_homogeneous(syn_pq)
        out_gt = single_ptquat2transform(syn_pq)
        np.testing.assert_allclose(out, syn_gt)
        np.testing.assert_allclose(out, out_gt)

        # tst2
        out = pointquaternion_to_homogeneous(pq)
        out_list = [pointquaternion_to_homogeneous(pq[i]) for i in range(len(pq))]
        gt_list = [single_ptquat2transform(pq[i]) for i in range(len(pq))]
        for i in range(len(pq)):
            np.testing.assert_allclose(out[i], out_list[i], atol=0, rtol=1e-5)
            np.testing.assert_allclose(out[i], gt_list[i], atol=0, rtol=1e-5)

        # torch
        # tst1
        syn_pq = torch.tensor(syn_pq)
        syn_gt = torch.tensor(syn_gt)
        out = pointquaternion_to_homogeneous(syn_pq)
        out_gt = single_ptquat2transform(syn_pq)
        assert_allclose(out, syn_gt)
        assert_allclose(out, out_gt)

        # tst2
        pq = torch.tensor(pq)
        out = pointquaternion_to_homogeneous(pq)
        out_list = [pointquaternion_to_homogeneous(pq[i]) for i in range(len(pq))]
        gt_list = [single_ptquat2transform(pq[i]) for i in range(len(pq))]
        for i in range(len(pq)):
            assert_allclose(out[i], out_list[i], atol=0, rtol=1e-5)
            assert_allclose(out[i], gt_list[i], atol=0, rtol=1e-5)

        # type errors
        with pytest.raises(TypeError):
            pointquaternion_to_homogeneous("abc")
        with pytest.raises(TypeError):
            pointquaternion_to_homogeneous(pq, "abc")

        # value errors
        with pytest.raises(ValueError):
            pointquaternion_to_homogeneous(np.ones((5, 8)))

    def test_transforms(self):
        poses = """-0.999762 0.000000 -0.021799 1.370500
0.000000 1.000000 0.000000 1.517390
0.021799 0.000000 -0.999762 1.449630
</br>
-0.999742 0.004366 -0.022272 1.367762
0.004157 0.999947 0.009404 1.519902
0.022312 0.009309 -0.999707 1.449820
</br>
-0.999656 0.004837 -0.025762 1.366492
0.004688 0.999972 0.005846 1.513329
0.025790 0.005723 -0.999651 1.453128
</br>
-0.998973 0.011128 -0.043917 1.359956
0.010923 0.999928 0.004896 1.516617
0.043968 0.004411 -0.999023 1.452431
</br>
-0.992071 0.047180 -0.116485 1.349768
0.046773 0.998886 0.006226 1.521626
0.116649 0.000729 -0.993173 1.444061
</br>
-0.980396 0.089427 -0.175573 1.244141
0.087771 0.995992 0.017191 1.526088
0.176407 0.001444 -0.984316 1.457656
</br>
-0.954118 0.130523 -0.269486 1.223184
0.128578 0.991386 0.024936 1.532083
0.270419 -0.010858 -0.962681 1.461294
</br>
-0.916433 0.180477 -0.357180 1.186825
0.176638 0.983308 0.043642 1.537364
0.359094 -0.023096 -0.933015 1.480111
"""

        poses = poses.split("</br>")
        for i, p in enumerate(poses):
            p = [list(map(float, l.split(" "))) for l in p.split("\n") if len(l) != 0]
            p.append([0.0, 0.0, 0.0, 1.0])
            pose = np.array(p, dtype=np.float32)
            if i == 0:
                p0_inv = np.linalg.inv(pose).copy()
            poses[i] = p0_inv @ pose

        # list of np.ndarray poses
        transforms = poses_to_transforms(poses)
        for i, t in enumerate(transforms):
            if i == 0:
                t_agg = t.copy()
            else:
                t_agg = np.matmul(t_agg, t)
            np_assert_allclose(poses[i], t_agg, rtol=1e-05, atol=1e-6)

        # np.ndarray poses
        poses = np.array(poses, dtype=np.float32)
        transforms = poses_to_transforms(poses)
        for i, t in enumerate(transforms):
            if i == 0:
                t_agg = t.copy()
            else:
                t_agg = np.matmul(t_agg, t)
            np_assert_allclose(poses[i], t_agg, rtol=1e-05, atol=1e-6)

    # def test_create_label_image(self):
    #     pass


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger("config")
    logger.setLevel(logging.DEBUG)
    unittest.main()
