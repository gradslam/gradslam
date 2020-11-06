import logging
import unittest

import numpy as np
import open3d as o3d
import pytest
import torch

from tests.common_testing import TestCaseMixin
from gradslam.structures import Pointclouds, structutils


# Copied Pytorch3d testcase (and modified)
class TestPointclouds(TestCaseMixin, unittest.TestCase):
    def setUp(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)

    @staticmethod
    def init_pointclouds(
        num_pointclouds: int = 10,
        max_n: int = 100,
        point_attributes: bool = False,
        requires_grad: bool = False,
        lists_to_tensors: bool = False,
        device: str = "cpu",
    ):
        """
        Function to generate a Pointclouds object of N pointclouds with
        random numbers of points.

        Args:
            num_pontclouds: Number of pointclouds to generate.
            max_n: Max number of points per pointclouds.
            point_attributes: Determines whether the generated points should have all of normals/colors/features
            lists_to_tensors: Determines whether the generated pointclouds should be
                              constructed from lists (=False) or
                              a tensor (=True) of points.

        Returns:
            Pointclouds object.
        """
        device = torch.device(device)

        points_list = []
        faces_list = []
        normals_list = None
        colors_list = None
        features_list = None

        # Randomly generate numbers of points in each pointclouds.
        if lists_to_tensors:
            # If we define points with tensors, n has to be the same for each pointclouds in the batch.
            n = torch.randint(3, high=max_n, size=(1,), dtype=torch.int32)
            n = n.repeat(num_pointclouds)
        else:
            # For lists of points, we can sample different n per pointclouds.
            n = torch.randint(3, high=max_n, size=(num_pointclouds,), dtype=torch.int32)

        # Generate the actual points
        points_list = [
            torch.rand((n[i], 3), dtype=torch.float32, device=device)
            for i in range(num_pointclouds)
        ]

        if point_attributes:
            normals_list = [
                torch.rand((n[i], 3), dtype=torch.float32, device=device)
                for i in range(num_pointclouds)
            ]
            colors_list = [
                torch.rand((n[i], 3), dtype=torch.float32, device=device)
                for i in range(num_pointclouds)
            ]
            features_list = [
                torch.rand((n[i], 3), dtype=torch.float32, device=device)
                for i in range(num_pointclouds)
            ]

        if requires_grad:
            points_list = [p.requires_grad_() for p in points_list]
            if point_attributes:
                normals_list = [n.requires_grad_() for n in normals_list]
                colors_list = [c.requires_grad_() for c in colors_list]
                features_list = [f.requires_grad_() for f in features_list]

        if lists_to_tensors:
            points_list = torch.stack(points_list)
            if point_attributes:
                normals_list = torch.stack(normals_list)
                colors_list = torch.stack(colors_list)
                features_list = torch.stack(features_list)

        return Pointclouds(
            points=points_list,
            normals=normals_list,
            colors=colors_list,
            features=features_list,
        )

    @staticmethod
    def init_simple_pointclouds(device: str = "cpu"):
        """
        Returns a Pointclouds data structure of simple pointclouds examples.

        Returns:
            Pointclouds object.
        """
        device = torch.device(device)

        points = [
            torch.tensor(
                [[0.1, 0.3, 0.5], [0.5, 0.2, 0.1], [0.6, 0.8, 0.7]],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [[0.1, 0.3, 0.3], [0.6, 0.7, 0.8], [0.2, 0.3, 0.4], [0.1, 0.5, 0.3]],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [
                    [0.7, 0.3, 0.6],
                    [0.2, 0.4, 0.8],
                    [0.9, 0.5, 0.2],
                    [0.2, 0.3, 0.4],
                    [0.9, 0.3, 0.8],
                ],
                dtype=torch.float32,
                device=device,
            ),
        ]
        return Pointclouds(points=points)

    def assertAllClose(self, tensor_list1, tensor_list2):
        self.assertEqual(len(tensor_list1), len(tensor_list2))
        for b in range(len(tensor_list1)):
            self.assertClose(tensor_list1[b], tensor_list2[b])

    def test_simple(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        pointclouds = TestPointclouds.init_simple_pointclouds(device)

        self.assertFalse(pointclouds.has_normals)
        self.assertFalse(pointclouds.has_colors)
        self.assertFalse(pointclouds.has_features)
        self.assertClose(
            pointclouds.num_points_per_pointcloud.cpu(), torch.tensor([3, 4, 5])
        )

    def test_basic_ops(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        points = [
            torch.tensor(
                [[0.1, 0.3, 0.5], [0.5, 0.2, 0.1], [0.6, 0.8, 0.7]],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [[0.1, 0.3, 0.3], [0.6, 0.7, 0.8], [0.2, 0.3, 0.4], [0.1, 0.5, 0.3]],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [
                    [0.7, 0.3, 0.6],
                    [0.2, 0.4, 0.8],
                    [0.9, 0.5, 0.2],
                    [0.2, 0.3, 0.4],
                    [0.9, 0.3, 0.8],
                ],
                dtype=torch.float32,
                device=device,
            ),
        ]
        points_padded = structutils.list_to_padded(points, (5, 3), pad_value=0.0)

        # inplace scalar offset
        pointclouds = TestPointclouds.init_simple_pointclouds(device)
        pointclouds.offset_(5)
        offsetted_points_list = [p + 5 for p in points]
        offsetted_points_padded = structutils.list_to_padded(
            offsetted_points_list, (5, 3), pad_value=0.0
        )
        self.assertAllClose(pointclouds.points_list, offsetted_points_list)
        self.assertClose(pointclouds.points_padded, offsetted_points_padded)

        # + scalar
        pointclouds = TestPointclouds.init_simple_pointclouds(device)
        res = pointclouds + 5
        self.assertAllClose(res.points_list, offsetted_points_list)
        self.assertClose(res.points_padded, offsetted_points_padded)

        # + 1-dim tensor
        a = torch.Tensor([2, 5, 7]).unsqueeze(0).unsqueeze(0).to(device)
        res = pointclouds + a
        offsetted_points_list = [p + a.squeeze() for p in points]
        offsetted_points_padded = structutils.list_to_padded(
            offsetted_points_list, (5, 3), pad_value=0.0
        )
        self.assertAllClose(res.points_list, offsetted_points_list)
        self.assertClose(res.points_padded, offsetted_points_padded)

        # + 3-dim tensor
        a = (
            torch.Tensor([[0, 1, -4], [-2, 10, -0.2], [1, 5, 3]])
            .unsqueeze(1)
            .to(device)
        )
        res = pointclouds + a
        offsetted_points_list = [p + a[b] for b, p in enumerate(points)]
        offsetted_points_padded = structutils.list_to_padded(
            offsetted_points_list, (5, 3), pad_value=0.0
        )
        self.assertAllClose(res.points_list, offsetted_points_list)
        self.assertClose(res.points_padded, offsetted_points_padded)

        # inplace scalar scale test
        pointclouds.scale_(5)
        scaled_points_list = [p * 5 for p in points]
        scaled_points_padded = structutils.list_to_padded(
            scaled_points_list, (5, 3), pad_value=0.0
        )
        self.assertAllClose(pointclouds.points_list, scaled_points_list)
        self.assertClose(pointclouds.points_padded, scaled_points_padded)
        self.assertSeparate(res.points_padded, pointclouds.points_padded)

        # * 1-dim tensor
        pointclouds = TestPointclouds.init_simple_pointclouds(device)
        a = torch.Tensor([2, 5, 7]).unsqueeze(0).unsqueeze(0).to(device)
        res = pointclouds * a
        scaled_points_list = [p * a.squeeze() for p in points]
        scaled_points_padded = structutils.list_to_padded(
            scaled_points_list, (5, 3), pad_value=0.0
        )
        self.assertAllClose(res.points_list, scaled_points_list)
        self.assertClose(res.points_padded, scaled_points_padded)

        # - 1-dim tensor
        a = torch.Tensor([2, 5, 7]).unsqueeze(0).unsqueeze(0).to(device)
        res = pointclouds - a
        sub_points_list = [p - a.squeeze() for p in points]
        sub_points_padded = structutils.list_to_padded(
            sub_points_list, (5, 3), pad_value=0.0
        )
        self.assertAllClose(res.points_list, sub_points_list)
        self.assertClose(res.points_padded, sub_points_padded)
        res = (pointclouds * -1) + a
        sub_points_list = [a.squeeze() - p for p in points]
        sub_points_padded = structutils.list_to_padded(
            sub_points_list, (5, 3), pad_value=0.0
        )
        self.assertAllClose(res.points_list, sub_points_list)
        self.assertClose(res.points_padded, sub_points_padded)
        self.assertSeparate(res.points_padded, pointclouds.points_padded)

        # / 1-dim tensor
        a = torch.Tensor([2, 5, 7]).unsqueeze(0).unsqueeze(0).to(device)
        res = pointclouds / a
        div_points_list = [p / a.squeeze() for p in points]
        div_points_padded = structutils.list_to_padded(
            div_points_list, (5, 3), pad_value=0.0
        )
        self.assertAllClose(res.points_list, div_points_list)
        self.assertClose(res.points_padded, div_points_padded)
        self.assertSeparate(res.points_padded, pointclouds.points_padded)

    def test_geometry_linalg_ops(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        points = [
            torch.tensor(
                [[0.1, 0.3, 0.5], [0.5, 0.2, 0.1], [0.6, 0.8, 0.7]],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [[0.1, 0.3, 0.3], [0.6, 0.7, 0.8], [0.2, 0.3, 0.4], [0.1, 0.5, 0.3]],
                dtype=torch.float32,
                device=device,
            ),
            torch.tensor(
                [
                    [0.7, 0.3, 0.6],
                    [0.2, 0.4, 0.8],
                    [0.9, 0.5, 0.2],
                    [0.2, 0.3, 0.4],
                    [0.9, 0.3, 0.8],
                ],
                dtype=torch.float32,
                device=device,
            ),
        ]
        # rotate_ and transform_ testing
        transform = torch.Tensor(
            [
                [-0.802837, 0.056561, -0.593509, 2.583219],
                [0.596192, 0.071654, -0.799638, 4.008804],
                [-0.002701, -0.995825, -0.091248, 1.439254],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).to(device)
        mat = transform[:3, :3]
        tvec = transform[:3, 3]
        intrinsics0 = torch.Tensor(
            [
                [577.87, 0.0, 319.5, 0.0],
                [0.0, 577.87, 239.5, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).to(device)
        intrinsics1 = torch.Tensor(
            [
                [377.87, 0.0, 219.5, 0.0],
                [0.0, 377.87, 139.5, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).to(device)
        intrinsics2 = torch.Tensor(
            [
                [677.87, 0.0, 419.5, 0.0],
                [0.0, 677.87, 339.5, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).to(device)
        batch_intrinsics = torch.stack([intrinsics0, intrinsics1, intrinsics2], 0)

        # pretransform rotate_
        pre_mult_points_list = [torch.mm(mat, p.t()).t() for p in points]
        pre_mult_points_padded = structutils.list_to_padded(
            pre_mult_points_list, (5, 3), pad_value=0.0
        )
        pointclouds = TestPointclouds.init_simple_pointclouds(device)
        pointclouds.rotate_(mat)
        self.assertClose(pointclouds.points_padded, pre_mult_points_padded)
        self.assertAllClose(pointclouds.points_list, pre_mult_points_list)

        # mat @ Pointclouds
        normals = [p * 2 for p in points]
        pre_mult_normals_list = [torch.mm(mat, n.t()).t() for n in normals]
        pre_mult_normals_padded = structutils.list_to_padded(
            pre_mult_normals_list, (5, 3), pad_value=0.0
        )
        pointclouds = Pointclouds(points=points, normals=normals)
        # import pdb; pdb.set_trace();
        pointclouds = pointclouds.rotate(mat)
        self.assertClose(pointclouds.points_padded, pre_mult_points_padded)
        self.assertAllClose(pointclouds.points_list, pre_mult_points_list)
        self.assertClose(pointclouds.normals_padded, pre_mult_normals_padded)
        self.assertAllClose(pointclouds.normals_list, pre_mult_normals_list)

        # posttransform rotate_
        post_mult_points_list = [torch.mm(p, mat) for p in points]
        post_mult_points_padded = structutils.list_to_padded(
            post_mult_points_list, (5, 3), pad_value=0.0
        )
        pointclouds = TestPointclouds.init_simple_pointclouds(device)
        pointclouds.rotate_(mat, pre_multiplication=False)
        self.assertClose(pointclouds.points_padded, post_mult_points_padded)
        self.assertAllClose(pointclouds.points_list, post_mult_points_list)

        # Pointclouds @ mat
        pointclouds = TestPointclouds.init_simple_pointclouds(device)
        pointclouds = pointclouds @ mat
        self.assertClose(pointclouds.points_padded, post_mult_points_padded)
        self.assertAllClose(pointclouds.points_list, post_mult_points_list)

        # batch rotate_
        mat_batch = [mat * i for i in range(1, 4)]
        pre_batch_mult_points_list = [
            torch.mm(mat_batch[b], p.t()).t() for b, p in enumerate(points)
        ]
        pre_batch_mult_points_padded = structutils.list_to_padded(
            pre_batch_mult_points_list, (5, 3), pad_value=0.0
        )
        pointclouds = TestPointclouds.init_simple_pointclouds(device)
        mat_batch_tensor = torch.stack(mat_batch, 0)
        pointclouds.rotate_(mat_batch_tensor)
        self.assertClose(pointclouds.points_padded, pre_batch_mult_points_padded)
        self.assertAllClose(pointclouds.points_list, pre_batch_mult_points_list)

        # batch_mat @ Pointclouds
        pointclouds = TestPointclouds.init_simple_pointclouds(device)
        pointclouds = pointclouds.rotate(mat_batch_tensor)
        self.assertClose(pointclouds.points_padded, pre_batch_mult_points_padded)
        self.assertAllClose(pointclouds.points_list, pre_batch_mult_points_list)

        # transform_
        transform_points_list = [p + tvec for p in pre_mult_points_list]
        transform_points_padded = structutils.list_to_padded(
            transform_points_list, (5, 3), pad_value=0.0
        )
        pointclouds = TestPointclouds.init_simple_pointclouds(device)
        pointclouds.transform_(transform)
        self.assertClose(pointclouds.points_padded, transform_points_padded)
        self.assertAllClose(pointclouds.points_list, transform_points_list)

        # transform @ Pointclouds
        pointclouds = TestPointclouds.init_simple_pointclouds(device)
        pointclouds = pointclouds.transform(transform)
        self.assertClose(pointclouds.points_padded, transform_points_padded)
        self.assertAllClose(pointclouds.points_list, transform_points_list)

        # batch transform_
        transform_batch = [transform * i for i in range(1, 4)]
        batch_transform_points_list = [
            p + (tvec * (b + 1)) for b, p in enumerate(pre_batch_mult_points_list)
        ]
        batch_transform_points_padded = structutils.list_to_padded(
            batch_transform_points_list, (5, 3), pad_value=0.0
        )
        pointclouds = TestPointclouds.init_simple_pointclouds(device)
        transform_batch_tensor = torch.stack(transform_batch, 0)
        pointclouds.transform_(transform_batch_tensor)
        self.assertClose(pointclouds.points_padded, batch_transform_points_padded)
        self.assertAllClose(pointclouds.points_list, batch_transform_points_list)

        # batch_transform @ Pointclouds
        pointclouds = TestPointclouds.init_simple_pointclouds(device)
        pointclouds = pointclouds.transform(transform_batch_tensor)
        self.assertClose(pointclouds.points_padded, batch_transform_points_padded)
        self.assertAllClose(pointclouds.points_list, batch_transform_points_list)

        # pinhole_projection_
        pointclouds0 = TestPointclouds.init_simple_pointclouds(device)
        pointclouds1 = TestPointclouds.init_simple_pointclouds(device)
        pointclouds2 = TestPointclouds.init_simple_pointclouds(device)
        pointclouds0.pinhole_projection_(intrinsics0)
        pointclouds1.pinhole_projection_(intrinsics1)
        pointclouds2.pinhole_projection_(intrinsics2)
        self.assertEqual(pointclouds0.points_padded.shape, (3, 5, 3))
        self.assertEqual(len(pointclouds0.points_list), 3)
        points_list = pointclouds0.points_list
        points_padded = pointclouds0.points_padded
        points_per_pointcloud = pointclouds0.num_points_per_pointcloud
        for b in range(3):
            n = points_list[b].shape[0]
            self.assertClose(points_padded[b, :n, :], points_list[b])
            if points_padded.shape[1] > n:
                self.assertTrue(points_padded[b, n:, :].eq(0).all())
            self.assertEqual(points_per_pointcloud[b], n)

        # batch pinhole_projection_
        pointclouds = TestPointclouds.init_simple_pointclouds(device)
        pointclouds.pinhole_projection_(batch_intrinsics)
        self.assertEqual(pointclouds.points_padded.shape, (3, 5, 3))
        self.assertClose(pointclouds.points_padded[0], pointclouds0.points_padded[0])
        self.assertClose(pointclouds.points_padded[1], pointclouds1.points_padded[1])
        self.assertClose(pointclouds.points_padded[2], pointclouds2.points_padded[2])

    def test_bad_ops_inputs(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        pointclouds = TestPointclouds.init_simple_pointclouds(device)

        # +
        with self.assertRaises(RuntimeError):
            out = pointclouds + torch.rand(1, 1, 4).to(device)

        with self.assertRaises(NotImplementedError):
            out = pointclouds + pointclouds

        # -
        with self.assertRaises(RuntimeError):
            out = pointclouds - torch.rand(1, 1, 4).to(device)

        with self.assertRaises(NotImplementedError):
            out = pointclouds - (1, 2, 3)

        with self.assertRaises(NotImplementedError):
            out = pointclouds - pointclouds

        # *
        with self.assertRaises(RuntimeError):
            out = pointclouds * torch.rand(1, 1, 4).to(device)

        with self.assertRaises(NotImplementedError):
            out = pointclouds * pointclouds

        # /
        with self.assertRaises(RuntimeError):
            out = pointclouds / torch.rand(1, 1, 4).to(device)

        with self.assertRaises(NotImplementedError):
            out = pointclouds / (1, 2, 3)

        # @
        with self.assertRaises(ValueError):
            out = pointclouds @ torch.rand(1, 1, 4).to(device)

        with self.assertRaises(NotImplementedError):
            out = pointclouds @ (1, 2, 3)

        # rotate_
        with self.assertRaises(ValueError):
            out = pointclouds.rotate_(torch.rand(5, 2, 4))

    def test_simple_random_pointclouds(self):
        # Define the test pointclouds object either as a list or tensor of points.
        for lists_to_tensors in (False, True):
            B = 10
            pointclouds = TestPointclouds.init_pointclouds(
                B, 100, lists_to_tensors=lists_to_tensors
            )
            points_list = pointclouds.points_list

            # Check batch calculations.
            points_padded = pointclouds.points_padded
            points_per_pointcloud = pointclouds.num_points_per_pointcloud
            for b in range(B):
                n = points_list[b].shape[0]
                self.assertClose(points_padded[b, :n, :], points_list[b])
                if points_padded.shape[1] > n:
                    self.assertTrue(points_padded[b, n:, :].eq(0).all())
                self.assertEqual(points_per_pointcloud[b], n)

        # test pointclouds with attributes (normals/colors/features)
        for lists_to_tensors in (False, True):
            B = 10
            pointclouds = TestPointclouds.init_pointclouds(
                B, 100, point_attributes=True, lists_to_tensors=lists_to_tensors
            )

            points_list = pointclouds.points_list
            normals_list = pointclouds.normals_list
            colors_list = pointclouds.colors_list
            features_list = pointclouds.features_list

            # Check batch calculations.
            points_padded = pointclouds.points_padded
            normals_padded = pointclouds.normals_padded
            colors_padded = pointclouds.colors_padded
            features_padded = pointclouds.features_padded

            points_per_pointcloud = pointclouds.num_points_per_pointcloud
            for b in range(B):
                n = points_list[b].shape[0]
                self.assertClose(points_padded[b, :n, :], points_list[b])
                self.assertClose(normals_padded[b, :n, :], normals_list[b])
                self.assertClose(colors_padded[b, :n, :], colors_list[b])
                self.assertClose(features_padded[b, :n, :], features_list[b])
                if points_padded.shape[1] > n:
                    self.assertTrue(points_padded[b, n:, :].eq(0).all())
                    self.assertTrue(normals_padded[b, n:, :].eq(0).all())
                    self.assertTrue(colors_padded[b, n:, :].eq(0).all())
                    self.assertTrue(features_padded[b, n:, :].eq(0).all())
                self.assertEqual(points_per_pointcloud[b], n)

    def test_has_attributes(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        B = 10

        for lists_to_tensors in (False, True):
            # pointcloud without attributes
            pointclouds = TestPointclouds.init_pointclouds(
                B,
                100,
                point_attributes=False,
                lists_to_tensors=lists_to_tensors,
                device=device,
            )
            self.assertFalse(pointclouds.has_normals)
            self.assertFalse(pointclouds.has_colors)
            self.assertFalse(pointclouds.has_features)

            # pointcloud with attributes
            pointclouds = TestPointclouds.init_pointclouds(
                B,
                100,
                point_attributes=True,
                lists_to_tensors=lists_to_tensors,
                device=device,
            )
            self.assertTrue(pointclouds.has_normals)
            self.assertTrue(pointclouds.has_colors)
            self.assertTrue(pointclouds.has_features)

    def test_invalid_inputs(self):
        # test points
        points = []
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points)

        points = torch.Tensor()
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points)

        points = [torch.Tensor()]
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points)

        points = [torch.rand(1, 4)]
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points)

        points = torch.rand(1, 1, 4)
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points)

        # test normals
        points = [torch.rand(1, 3)]
        normals = torch.rand(1, 1, 3)
        with self.assertRaises(TypeError):
            pointclouds = Pointclouds(points=points, normals=normals)

        points = [torch.rand(1, 3), torch.rand(4, 3)]
        normals = [torch.rand(1, 3)]
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, normals=normals)

        points = [torch.rand(1, 3), torch.rand(4, 3)]
        normals = [torch.rand(1, 3), torch.rand(1, 3), torch.rand(4, 3)]
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, normals=normals)

        points = [torch.rand(1, 3), torch.rand(4, 3)]
        normals = [torch.rand(1, 3), torch.rand(5, 3)]
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, normals=normals)

        points = [torch.rand(1, 3), torch.rand(4, 3)]
        normals = [torch.rand(1, 3), torch.rand(4, 4)]
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, normals=normals)

        points = torch.rand(4, 1, 3)
        normals = torch.rand(5, 1, 3)
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, normals=normals)

        points = torch.rand(4, 1, 3)
        normals = torch.rand(4, 2, 3)
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, normals=normals)

        points = torch.rand(4, 1, 3)
        normals = torch.rand(4, 1, 4)
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, normals=normals)

        # test colors
        points = [torch.rand(1, 3)]
        colors = torch.rand(1, 1, 3)
        with self.assertRaises(TypeError):
            pointclouds = Pointclouds(points=points, colors=colors)

        points = [torch.rand(1, 3), torch.rand(4, 3)]
        colors = [torch.rand(1, 3)]
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, colors=colors)

        points = [torch.rand(1, 3), torch.rand(4, 3)]
        colors = [torch.rand(1, 3), torch.rand(1, 3), torch.rand(4, 3)]
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, colors=colors)

        points = [torch.rand(1, 3), torch.rand(4, 3)]
        colors = [torch.rand(1, 3), torch.rand(5, 3)]
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, colors=colors)

        points = [torch.rand(1, 3), torch.rand(4, 3)]
        colors = [torch.rand(1, 3), torch.rand(4, 4)]
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, colors=colors)

        points = torch.rand(4, 1, 3)
        colors = torch.rand(5, 1, 3)
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, colors=colors)

        points = torch.rand(4, 1, 3)
        colors = torch.rand(4, 2, 3)
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, colors=colors)

        points = torch.rand(4, 1, 3)
        colors = torch.rand(4, 1, 4)
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, colors=colors)

        # test features
        points = [torch.rand(1, 3)]
        features = torch.rand(1, 1, 3)
        with self.assertRaises(TypeError):
            pointclouds = Pointclouds(points=points, features=features)

        points = [torch.rand(1, 3), torch.rand(4, 3)]
        features = [torch.rand(1, 3)]
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, features=features)

        points = [torch.rand(1, 3), torch.rand(4, 3)]
        features = [torch.rand(1, 3), torch.rand(1, 3), torch.rand(4, 3)]
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, features=features)

        points = [torch.rand(1, 3), torch.rand(4, 3)]
        features = [torch.rand(1, 3), torch.rand(5, 3)]
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, features=features)

        points = [torch.rand(1, 3), torch.rand(4, 3)]
        features = [torch.rand(1, 3), torch.rand(4, 4)]
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, features=features)

        points = torch.rand(4, 1, 3)
        features = torch.rand(5, 1, 3)
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, features=features)

        points = torch.rand(4, 1, 3)
        features = torch.rand(4, 2, 3)
        with self.assertRaises(ValueError):
            pointclouds = Pointclouds(points=points, features=features)

        points = torch.rand(4, 1, 3)
        features = torch.rand(4, 1, 4)
        pointclouds = Pointclouds(points=points, features=features)

    def test_clone(self):
        B = 10
        pointclouds = TestPointclouds.init_pointclouds(B, 100, point_attributes=True)
        for force in [0, 1]:
            if force:
                # force pointclouds to have computed attributes
                pointclouds.points_padded
                pointclouds.normals_padded
                pointclouds.colors_padded
                pointclouds.features_padded

            new_pointclouds = pointclouds.clone()

            # Modify tensors in both pointclouds.
            new_pointclouds._points_list[0] *= 5
            pointclouds._num_points_per_pointcloud *= 2

            # Check cloned and original Pointclouds objects do not share tensors.
            for b in range(B):
                if b == 0:
                    self.assertFalse(
                        torch.allclose(
                            new_pointclouds._points_list[b], pointclouds._points_list[b]
                        )
                    )
                else:
                    self.assertTrue(
                        torch.allclose(
                            new_pointclouds._points_list[b], pointclouds._points_list[b]
                        )
                    )

                self.assertTrue(
                    torch.allclose(
                        new_pointclouds._normals_list[b], pointclouds._normals_list[b]
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        new_pointclouds._colors_list[b], pointclouds._colors_list[b]
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        new_pointclouds._features_list[b], pointclouds._features_list[b]
                    )
                )

            self.assertFalse(
                torch.allclose(
                    pointclouds.num_points_per_pointcloud,
                    new_pointclouds.num_points_per_pointcloud,
                )
            )

            self.assertSeparate(
                new_pointclouds.points_padded, pointclouds.points_padded
            )
            self.assertSeparate(
                new_pointclouds.normals_padded, pointclouds.normals_padded
            )
            self.assertSeparate(
                new_pointclouds.colors_padded, pointclouds.colors_padded
            )
            self.assertSeparate(
                new_pointclouds.features_padded, pointclouds.features_padded
            )

            self.assertAllSeparate(
                [*new_pointclouds.points_list, *pointclouds.points_list]
            )
            self.assertAllSeparate(
                [*new_pointclouds.normals_list, *pointclouds.normals_list]
            )
            self.assertAllSeparate(
                [*new_pointclouds.colors_list, *pointclouds.colors_list]
            )
            self.assertAllSeparate(
                [*new_pointclouds.features_list, *pointclouds.features_list]
            )

    def test_detach(self):
        B = 10
        pointclouds = TestPointclouds.init_pointclouds(
            B, 100, point_attributes=True, requires_grad=True
        )
        for force in [0, 1]:
            if force:
                # force pointclouds to have computed attributes
                pointclouds.points_padded
                pointclouds.normals_padded
                pointclouds.colors_padded
                pointclouds.features_padded

            new_pointclouds = pointclouds.detach()
            for b in range(B):
                self.assertTrue(
                    torch.allclose(
                        new_pointclouds._points_list[b], pointclouds._points_list[b]
                    )
                )

                self.assertTrue(pointclouds.points_list[b].requires_grad)
                self.assertTrue(pointclouds.normals_list[b].requires_grad)
                self.assertTrue(pointclouds.colors_list[b].requires_grad)
                self.assertTrue(pointclouds.features_list[b].requires_grad)

                self.assertFalse(new_pointclouds.points_list[b].requires_grad)
                self.assertFalse(new_pointclouds.normals_list[b].requires_grad)
                self.assertFalse(new_pointclouds.colors_list[b].requires_grad)
                self.assertFalse(new_pointclouds.features_list[b].requires_grad)

            self.assertTrue(pointclouds.points_padded[b].requires_grad)
            self.assertTrue(pointclouds.normals_padded[b].requires_grad)
            self.assertTrue(pointclouds.colors_padded[b].requires_grad)
            self.assertTrue(pointclouds.features_padded[b].requires_grad)

            self.assertFalse(new_pointclouds.points_padded[b].requires_grad)
            self.assertFalse(new_pointclouds.normals_padded[b].requires_grad)
            self.assertFalse(new_pointclouds.colors_padded[b].requires_grad)
            self.assertFalse(new_pointclouds.features_padded[b].requires_grad)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda not available")
    def test_to(self):
        for lists_to_tensors in (True, False):
            pointclouds = TestPointclouds.init_pointclouds(
                10,
                100,
                point_attributes=True,
                lists_to_tensors=lists_to_tensors,
                device=torch.device("cpu"),
            )
            device = torch.device("cuda:0")

            new_pointclouds = pointclouds.to(device)
            self.assertTrue(new_pointclouds.device == device)
            self.assertTrue(new_pointclouds.points_padded.device == device)
            self.assertTrue(new_pointclouds.normals_padded.device == device)
            self.assertTrue(pointclouds.device == torch.device("cpu"))
            self.assertTrue(pointclouds.points_padded.device == torch.device("cpu"))
            self.assertTrue(pointclouds.normals_padded.device == torch.device("cpu"))

            self.assertTrue(new_pointclouds.cpu().device == torch.device("cpu"))
            self.assertTrue(
                new_pointclouds.cpu().points_padded.device == torch.device("cpu")
            )
            self.assertTrue(pointclouds.cuda().device == torch.device("cuda:0"))
            self.assertTrue(
                pointclouds.cuda().points_padded.device == torch.device("cuda:0")
            )
            self.assertTrue(
                pointclouds.cuda().points_padded.device
                == pointclouds.points_padded.cuda().device
            )

    def test_getitem(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        B = 10

        pointclouds = TestPointclouds.init_pointclouds(
            5, 100, point_attributes=True, device=torch.device("cpu")
        )

        def check_equal(selected, indices):
            for selectedIdx, index in enumerate(indices):
                self.assertClose(
                    selected.points_list[selectedIdx],
                    pointclouds.points_list[index],
                )
                self.assertClose(
                    selected.normals_list[selectedIdx],
                    pointclouds.normals_list[index],
                )
                self.assertClose(
                    selected.colors_list[selectedIdx],
                    pointclouds.colors_list[index],
                )
                self.assertClose(
                    selected.features_list[selectedIdx],
                    pointclouds.features_list[index],
                )

        # int index
        index = 1
        pointcloud_selected = pointclouds[index]
        self.assertTrue(len(pointcloud_selected) == 1)
        check_equal(pointcloud_selected, [index])

        # list index
        index = [1, 2]
        pointcloud_selected = pointclouds[index]
        self.assertTrue(len(pointcloud_selected) == len(index))
        check_equal(pointcloud_selected, index)

        # slice index
        index = slice(0, 2, 1)
        pointcloud_selected = pointclouds[index]
        check_equal(pointcloud_selected, [0, 1])

        # bool tensor
        index = torch.tensor([1, 0, 1], dtype=torch.bool, device=device)
        pointcloud_selected = pointclouds[index]
        self.assertTrue(len(pointcloud_selected) == index.sum())
        check_equal(pointcloud_selected, [0, 2])

        # int tensor
        index = torch.tensor([1, 2], dtype=torch.int64, device=device)
        pointcloud_selected = pointclouds[index]
        self.assertTrue(len(pointcloud_selected) == index.numel())
        check_equal(pointcloud_selected, index.tolist())

        # invalid index
        index = torch.tensor([1, 0, 1], dtype=torch.float32, device=device)
        with self.assertRaises(IndexError):
            pointcloud_selected = pointclouds[index]
        index = 1.2
        with self.assertRaises(IndexError):
            pointcloud_selected = pointclouds[index]

    def test_append(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        B = 10

        # pointcloud without attributes
        pointclouds1 = TestPointclouds.init_pointclouds(
            B,
            100,
            point_attributes=True,
            lists_to_tensors=False,
            device=device,
        )
        pointclouds2 = TestPointclouds.init_pointclouds(
            B,
            100,
            point_attributes=True,
            lists_to_tensors=False,
            device=device,
        )
        num1 = pointclouds1.num_points_per_pointcloud
        num2 = pointclouds2.num_points_per_pointcloud
        num3 = num1 + num2
        pointclouds3 = pointclouds1.clone()
        pointclouds3.append_points(pointclouds2)

        for b in range(B):
            N1_b = pointclouds1.num_points_per_pointcloud[b]
            N2_b = pointclouds2.num_points_per_pointcloud[b]

            # Points
            self.assertTrue(
                torch.allclose(
                    pointclouds1.points_list[b], pointclouds3.points_list[b][:N1_b]
                )
            )
            self.assertTrue(
                torch.allclose(
                    pointclouds2.points_list[b], pointclouds3.points_list[b][N1_b:]
                )
            )
            # import pdb; pdb.set_trace();
            self.assertTrue(
                torch.allclose(
                    pointclouds1.points_padded[b][:N1_b],
                    pointclouds3.points_padded[b][:N1_b],
                )
            )
            self.assertTrue(
                torch.allclose(
                    pointclouds2.points_padded[b][:N2_b],
                    pointclouds3.points_padded[b][N1_b : N1_b + N2_b],
                )
            )

            # Features
            self.assertTrue(
                torch.allclose(
                    pointclouds1.features_list[b], pointclouds3.features_list[b][:N1_b]
                )
            )
            self.assertTrue(
                torch.allclose(
                    pointclouds2.features_list[b], pointclouds3.features_list[b][N1_b:]
                )
            )
            self.assertTrue(
                torch.allclose(
                    pointclouds1.features_padded[b][:N1_b],
                    pointclouds3.features_padded[b][:N1_b],
                )
            )
            self.assertTrue(
                torch.allclose(
                    pointclouds2.features_padded[b][:N2_b],
                    pointclouds3.features_padded[b][N1_b : N1_b + N2_b],
                )
            )

            # nonpad mask
            self.assertTrue(
                torch.allclose(
                    pointclouds1.nonpad_mask[b][:N1_b].float(),
                    pointclouds3.nonpad_mask[b][:N1_b].float(),
                )
            )
            self.assertTrue(
                torch.allclose(
                    pointclouds2.nonpad_mask[b][:N2_b].float(),
                    pointclouds3.nonpad_mask[b][N1_b : N1_b + N2_b].float(),
                )
            )

        self.assertClose(num3, pointclouds3.num_points_per_pointcloud)

    def test_o3d(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # pointcloud without attributes
        B = 10
        pointclouds = TestPointclouds.init_pointclouds(
            B,
            100,
            point_attributes=False,
            lists_to_tensors=False,
            device=device,
        )
        o3d_single = pointclouds.open3d(2)
        assert isinstance(o3d_single, o3d.geometry.PointCloud)

        # pointcloud with attributes
        B = 10
        pointclouds = TestPointclouds.init_pointclouds(
            B,
            100,
            point_attributes=True,
            lists_to_tensors=False,
            device=device,
        )
        o3d_single = pointclouds.open3d(2)
        assert isinstance(o3d_single, o3d.geometry.PointCloud)
        colors = np.asarray(o3d_single.colors)
        assert np.max(colors) < 1.01

    def test_propertysetters(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        B = 10
        pointclouds = TestPointclouds.init_pointclouds(
            B,
            100,
            point_attributes=True,
            lists_to_tensors=False,
            device=device,
        )
        points_list = [p * 2 for p in pointclouds.points_list]
        normals_list = [n * 2 for n in pointclouds.normals_list]
        colors_list = [c * 2 for c in pointclouds.colors_list]
        features_list = [f * 2 for f in pointclouds.features_list]
        points_padded = pointclouds.points_padded.clone()
        normals_padded = pointclouds.normals_padded.clone()
        colors_padded = pointclouds.colors_padded.clone()
        features_padded = pointclouds.features_padded.clone()
        pointclouds1 = Pointclouds(pointclouds.points_list, device=device)
        num = pointclouds.num_points_per_pointcloud
        pointclouds2 = Pointclouds(
            points_list, normals_list, colors_list, features_list
        )
        pointclouds3 = TestPointclouds.init_pointclouds(
            B,
            100,
            point_attributes=True,
            lists_to_tensors=False,
            device=device,
        )

        # set points_list
        pointclouds = pointclouds1.clone()
        pointclouds.points_list = pointclouds2.points_list
        assert pointclouds._points_padded is None
        for b in range(B):
            self.assertTrue(
                torch.allclose(
                    pointclouds.points_list[b],
                    pointclouds2.points_list[b],
                )
            )
        self.assertTrue(
            torch.allclose(
                pointclouds.points_padded,
                pointclouds2.points_padded,
            )
        )
        with self.assertRaises(ValueError):
            pointclouds.points_list = pointclouds3.points_list

        # set normals_list
        pointclouds = pointclouds1.clone()
        pointclouds.normals_list = pointclouds2.normals_list
        assert pointclouds._normals_padded is None
        for b in range(B):
            self.assertTrue(
                torch.allclose(
                    pointclouds.normals_list[b],
                    pointclouds2.normals_list[b],
                )
            )
        self.assertTrue(
            torch.allclose(
                pointclouds.normals_padded,
                pointclouds2.normals_padded,
            )
        )
        with self.assertRaises(ValueError):
            pointclouds.normals_list = pointclouds3.normals_list

        # set colors_list
        pointclouds = pointclouds1.clone()
        pointclouds.colors_list = pointclouds2.colors_list
        assert pointclouds._colors_padded is None
        for b in range(B):
            self.assertTrue(
                torch.allclose(
                    pointclouds.colors_list[b],
                    pointclouds2.colors_list[b],
                )
            )
        self.assertTrue(
            torch.allclose(
                pointclouds.colors_padded,
                pointclouds2.colors_padded,
            )
        )
        with self.assertRaises(ValueError):
            pointclouds.colors_list = pointclouds3.colors_list

        # set features_list
        pointclouds = pointclouds1.clone()
        pointclouds.features_list = pointclouds2.features_list
        assert pointclouds._features_padded is None
        for b in range(B):
            self.assertTrue(
                torch.allclose(
                    pointclouds.features_list[b],
                    pointclouds2.features_list[b],
                )
            )
        self.assertTrue(
            torch.allclose(
                pointclouds.features_padded,
                pointclouds2.features_padded,
            )
        )
        with self.assertRaises(ValueError):
            pointclouds.features_list = pointclouds3.features_list

        # set points_padded
        pointclouds = pointclouds1.clone()
        pointclouds.points_padded = pointclouds2.points_padded
        assert pointclouds._points_list is None
        for b in range(B):
            self.assertTrue(
                torch.allclose(
                    pointclouds.points_list[b],
                    pointclouds2.points_list[b],
                )
            )
        self.assertTrue(
            torch.allclose(
                pointclouds.points_padded,
                pointclouds2.points_padded,
            )
        )
        with self.assertRaises(ValueError):
            pointclouds.points_padded = torch.ones_like(points_padded)

        # set normals_padded
        pointclouds = pointclouds1.clone()
        pointclouds.normals_padded = pointclouds2.normals_padded
        assert pointclouds._normals_list is None
        for b in range(B):
            self.assertTrue(
                torch.allclose(
                    pointclouds.normals_list[b],
                    pointclouds2.normals_list[b],
                )
            )
        self.assertTrue(
            torch.allclose(
                pointclouds.normals_padded,
                pointclouds2.normals_padded,
            )
        )
        with self.assertRaises(ValueError):
            pointclouds.normals_padded = torch.ones_like(normals_padded)

        # set colors_padded
        pointclouds = pointclouds1.clone()
        pointclouds.colors_padded = pointclouds2.colors_padded
        assert pointclouds._colors_list is None
        for b in range(B):
            self.assertTrue(
                torch.allclose(
                    pointclouds.colors_list[b],
                    pointclouds2.colors_list[b],
                )
            )
        self.assertTrue(
            torch.allclose(
                pointclouds.colors_padded,
                pointclouds2.colors_padded,
            )
        )
        with self.assertRaises(ValueError):
            pointclouds.colors_padded = torch.ones_like(colors_padded)

        # set features_padded
        pointclouds = pointclouds1.clone()
        pointclouds.features_padded = pointclouds2.features_padded
        assert pointclouds._features_list is None
        for b in range(B):
            self.assertTrue(
                torch.allclose(
                    pointclouds.features_list[b],
                    pointclouds2.features_list[b],
                )
            )
        self.assertTrue(
            torch.allclose(
                pointclouds.features_padded,
                pointclouds2.features_padded,
            )
        )
        with self.assertRaises(ValueError):
            pointclouds.features_padded = torch.ones_like(features_padded)

    def test_emptypointclouds(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        pointclouds = Pointclouds(device=device)

        assert pointclouds.points_list is None
        assert pointclouds.points_padded is None

        # indexing
        with self.assertRaises(IndexError):
            pointclouds[0]

        # ops
        pointclouds2 = pointclouds + 5
        assert pointclouds2.points_list is None
        assert pointclouds2.points_padded is None
        assert not pointclouds2.has_points

        pointclouds2 = pointclouds * 5
        assert pointclouds2.points_list is None
        assert pointclouds2.points_padded is None
        assert not pointclouds2.has_points

        pointclouds2 = pointclouds / 5
        assert pointclouds2.points_list is None
        assert pointclouds2.points_padded is None
        assert not pointclouds2.has_points

        transform = torch.Tensor(
            [
                [-0.802837, 0.056561, -0.593509, 2.583219],
                [0.596192, 0.071654, -0.799638, 4.008804],
                [-0.002701, -0.995825, -0.091248, 1.439254],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).to(device)
        rmat = transform[:3, :3]
        tvec = transform[:3, 3]
        intrinsics0 = torch.Tensor(
            [
                [577.87, 0.0, 319.5, 0.0],
                [0.0, 577.87, 239.5, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).to(device)
        pointclouds2 = pointclouds.rotate(rmat)
        assert pointclouds2.points_list is None
        assert pointclouds2.points_padded is None
        assert not pointclouds2.has_points

        pointclouds2 = pointclouds.offset_(tvec)
        assert pointclouds2.points_list is None
        assert pointclouds2.points_padded is None
        assert not pointclouds2.has_points

        pointclouds2 = pointclouds.transform(transform)
        assert pointclouds2.points_list is None
        assert pointclouds2.points_padded is None
        assert not pointclouds2.has_points

        pointclouds2 = pointclouds.pinhole_projection(intrinsics0)
        assert pointclouds2.points_list is None
        assert pointclouds2.points_padded is None
        assert not pointclouds2.has_points

        # properties
        assert len(pointclouds) == 0
        assert pointclouds.points_list is None
        assert pointclouds.normals_list is None
        assert pointclouds.colors_list is None
        assert pointclouds.features_list is None
        assert pointclouds.points_padded is None
        assert pointclouds.normals_padded is None
        assert pointclouds.colors_padded is None
        assert pointclouds.features_padded is None
        assert pointclouds.nonpad_mask is None
        assert pointclouds.num_points_per_pointcloud == torch.tensor([0], device=device)

        # property setters

        # clone/detach/to
        pointclouds2 = pointclouds.clone()
        assert pointclouds2.points_list is None
        assert pointclouds2.normals_list is None
        assert pointclouds2.colors_list is None
        assert pointclouds2.features_list is None
        assert pointclouds2.points_padded is None
        assert pointclouds2.normals_padded is None
        assert pointclouds2.colors_padded is None
        assert pointclouds2.features_padded is None
        assert pointclouds2.nonpad_mask is None
        assert pointclouds2.device == pointclouds.device

        pointclouds2 = pointclouds.detach()
        assert pointclouds2.points_list is None
        assert pointclouds2.normals_list is None
        assert pointclouds2.colors_list is None
        assert pointclouds2.features_list is None
        assert pointclouds2.points_padded is None
        assert pointclouds2.normals_padded is None
        assert pointclouds2.colors_padded is None
        assert pointclouds2.features_padded is None
        assert pointclouds2.nonpad_mask is None
        assert pointclouds2.device == pointclouds.device

        assert pointclouds.to("cpu").device == torch.device("cpu")

        # append points
        B = 10
        pointclouds1 = TestPointclouds.init_pointclouds(
            B,
            100,
            point_attributes=True,
            lists_to_tensors=False,
            device=device,
        )
        num1 = pointclouds1.num_points_per_pointcloud
        pointclouds2 = pointclouds.clone()
        pointclouds2.append_points(pointclouds1)
        pointclouds3 = pointclouds1.clone().append_points(pointclouds.clone())

        for b in range(B):
            N1_b = pointclouds1.num_points_per_pointcloud[b]
            N2_b = pointclouds2.num_points_per_pointcloud[b]
            assert N1_b == N2_b

            # points_list
            self.assertTrue(
                torch.allclose(pointclouds1.points_list[b], pointclouds2.points_list[b])
            )
            self.assertTrue(
                torch.allclose(pointclouds1.points_list[b], pointclouds3.points_list[b])
            )

            # features_list
            self.assertTrue(
                torch.allclose(
                    pointclouds1.features_list[b], pointclouds2.features_list[b]
                )
            )
            self.assertTrue(
                torch.allclose(
                    pointclouds1.features_list[b], pointclouds3.features_list[b]
                )
            )

        # points_padded
        self.assertTrue(
            torch.allclose(
                pointclouds1.points_padded,
                pointclouds2.points_padded,
            )
        )
        self.assertTrue(
            torch.allclose(
                pointclouds1.points_padded,
                pointclouds3.points_padded,
            )
        )

        # features_padded
        self.assertTrue(
            torch.allclose(
                pointclouds1.features_padded[b],
                pointclouds2.features_padded[b],
            )
        )
        self.assertTrue(
            torch.allclose(
                pointclouds1.features_padded,
                pointclouds3.features_padded,
            )
        )

        # nonpad mask
        self.assertTrue(
            torch.allclose(
                pointclouds1.nonpad_mask.float(),
                pointclouds2.nonpad_mask.float(),
            )
        )
        self.assertTrue(
            torch.allclose(
                pointclouds1.nonpad_mask.float(),
                pointclouds3.nonpad_mask.float(),
            )
        )


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger("config")
    logger.setLevel(logging.DEBUG)
    unittest.main()
