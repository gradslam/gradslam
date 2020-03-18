import logging
import unittest

import numpy as np
import pytest
import torch

from gradslam import Pointclouds
from gradslam.structures import structutils
from tests.common_testing import TestCaseMixin


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

    def test_simple(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        pointclouds = TestPointclouds.init_simple_pointclouds(device)

        self.assertFalse(pointclouds.has_normals())
        self.assertFalse(pointclouds.has_colors())
        self.assertFalse(pointclouds.has_features())
        self.assertClose(
            pointclouds.num_points_per_pointcloud().cpu(), torch.tensor([3, 4, 5])
        )

    def test_simple_random_pointclouds(self):
        # Define the test pointclouds object either as a list or tensor of points.
        for lists_to_tensors in (False, True):
            B = 10
            pointclouds = TestPointclouds.init_pointclouds(
                B, 100, lists_to_tensors=lists_to_tensors
            )
            points_list = pointclouds.points_list()

            # Check batch calculations.
            points_padded = pointclouds.points_padded()
            points_per_pointcloud = pointclouds.num_points_per_pointcloud()
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

            points_list = pointclouds.points_list()
            normals_list = pointclouds.normals_list()
            colors_list = pointclouds.colors_list()
            features_list = pointclouds.features_list()

            # Check batch calculations.
            points_padded = pointclouds.points_padded()
            normals_padded = pointclouds.normals_padded()
            colors_padded = pointclouds.colors_padded()
            features_padded = pointclouds.features_padded()

            points_per_pointcloud = pointclouds.num_points_per_pointcloud()
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
            self.assertFalse(pointclouds.has_normals())
            self.assertFalse(pointclouds.has_colors())
            self.assertFalse(pointclouds.has_features())

            # pointcloud with attributes
            pointclouds = TestPointclouds.init_pointclouds(
                B,
                100,
                point_attributes=True,
                lists_to_tensors=lists_to_tensors,
                device=device,
            )
            self.assertTrue(pointclouds.has_normals())
            self.assertTrue(pointclouds.has_colors())
            self.assertTrue(pointclouds.has_features())

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
                pointclouds.points_padded()
                pointclouds.normals_padded()
                pointclouds.colors_padded()
                pointclouds.features_padded()

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
                    pointclouds.num_points_per_pointcloud(),
                    new_pointclouds.num_points_per_pointcloud(),
                )
            )

            self.assertSeparate(
                new_pointclouds.points_padded(), pointclouds.points_padded()
            )
            self.assertSeparate(
                new_pointclouds.normals_padded(), pointclouds.normals_padded()
            )
            self.assertSeparate(
                new_pointclouds.colors_padded(), pointclouds.colors_padded()
            )
            self.assertSeparate(
                new_pointclouds.features_padded(), pointclouds.features_padded()
            )

            self.assertAllSeparate(
                [*new_pointclouds.points_list(), *pointclouds.points_list()]
            )
            self.assertAllSeparate(
                [*new_pointclouds.normals_list(), *pointclouds.normals_list()]
            )
            self.assertAllSeparate(
                [*new_pointclouds.colors_list(), *pointclouds.colors_list()]
            )
            self.assertAllSeparate(
                [*new_pointclouds.features_list(), *pointclouds.features_list()]
            )

    def test_detach(self):
        B = 10
        pointclouds = TestPointclouds.init_pointclouds(
            B, 100, point_attributes=True, requires_grad=True
        )
        for force in [0, 1]:
            if force:
                # force pointclouds to have computed attributes
                pointclouds.points_padded()
                pointclouds.normals_padded()
                pointclouds.colors_padded()
                pointclouds.features_padded()

            new_pointclouds = pointclouds.detach()
            for b in range(B):
                self.assertTrue(
                    torch.allclose(
                        new_pointclouds._points_list[b], pointclouds._points_list[b]
                    )
                )

                self.assertTrue(pointclouds.points_list()[b].requires_grad)
                self.assertTrue(pointclouds.normals_list()[b].requires_grad)
                self.assertTrue(pointclouds.colors_list()[b].requires_grad)
                self.assertTrue(pointclouds.features_list()[b].requires_grad)

                self.assertFalse(new_pointclouds.points_list()[b].requires_grad)
                self.assertFalse(new_pointclouds.normals_list()[b].requires_grad)
                self.assertFalse(new_pointclouds.colors_list()[b].requires_grad)
                self.assertFalse(new_pointclouds.features_list()[b].requires_grad)

            self.assertTrue(pointclouds.points_padded()[b].requires_grad)
            self.assertTrue(pointclouds.normals_padded()[b].requires_grad)
            self.assertTrue(pointclouds.colors_padded()[b].requires_grad)
            self.assertTrue(pointclouds.features_padded()[b].requires_grad)

            self.assertFalse(new_pointclouds.points_padded()[b].requires_grad)
            self.assertFalse(new_pointclouds.normals_padded()[b].requires_grad)
            self.assertFalse(new_pointclouds.colors_padded()[b].requires_grad)
            self.assertFalse(new_pointclouds.features_padded()[b].requires_grad)

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
            self.assertTrue(new_pointclouds.points_padded().device == device)
            self.assertTrue(new_pointclouds.normals_padded().device == device)
            self.assertTrue(pointclouds.device == torch.device("cpu"))
            self.assertTrue(pointclouds.points_padded().device == torch.device("cpu"))
            self.assertTrue(pointclouds.normals_padded().device == torch.device("cpu"))

            self.assertTrue(new_pointclouds.cpu().device == torch.device("cpu"))
            self.assertTrue(
                new_pointclouds.cpu().points_padded().device == torch.device("cpu")
            )
            self.assertTrue(pointclouds.cuda().device == torch.device("cuda:0"))
            self.assertTrue(
                pointclouds.cuda().points_padded().device == torch.device("cuda:0")
            )
            self.assertTrue(
                pointclouds.cuda().points_padded().device
                == pointclouds.points_padded().cuda().device
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
                    selected.points_list()[selectedIdx],
                    pointclouds.points_list()[index],
                )
                self.assertClose(
                    selected.normals_list()[selectedIdx],
                    pointclouds.normals_list()[index],
                )
                self.assertClose(
                    selected.colors_list()[selectedIdx],
                    pointclouds.colors_list()[index],
                )
                self.assertClose(
                    selected.features_list()[selectedIdx],
                    pointclouds.features_list()[index],
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


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger("config")
    logger.setLevel(logging.DEBUG)
    unittest.main()
