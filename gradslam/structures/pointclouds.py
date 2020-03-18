from typing import List, Union

import torch

from . import structutils


__all__ = ["Pointclouds"]


class Pointclouds(object):
    r"""Holds a batch of pointclouds (with varying numbers of points), enabling conversion between 2 representations:

    - List: Store points of each pointcloud of shape :math:`(N_b, 3)` in a list of length :math:`B`.
    - Padded: Store all points in a :math:`(B, max(N_b), 3)` tensor with zero padding as required.

    Args:
        points (torch.Tensor or list of torch.Tensor): :math:`(X, Y, Z)` coordinates of each point.
        normals (torch.Tensor or list of torch.Tensor): Normals of each point. Default: None
        colors (torch.Tensor or list of torch.Tensor): RGB color of each point. Default: None
        features (torch.Tensor or list of torch.Tensor): C features of each point. Default: None

    Shape:
        - points: Can either be a list of tensors of shape :math:`(N, 3)` or a padded tensor of shape
        :math:`(B, N, 3)`.
        - normals: Can either be a list of tensors of shape :math:`(N, 3)` or a padded tensor of shape
        :math:`(B, N, 3)`.
        - colors: Can either be a list of tensors of shape :math:`(N, 3)` or a padded tensor of shape
        :math:`(B, N, 3)`.
        - features: Can either be a list of tensors of shape :math:`(N, C)` or a padded tensor of shape
        :math:`(B, N, C)`.

    Examples::

        >>> points_list = [torch.rand(1, 3), torch.rand(4, 3)]
        >>> pcs1 = gradslam.Pointclouds(points_list)
        >>> print(pcs1.points_padded().shape)
        torch.Size([2, 4, 3])
        >>> print(len(pcs1.points_list()))
        2
        >>> pcs2 = gradslam.Pointclouds(torch.rand((2, 4, 3)))
        >>> print(pcs2.points_padded().shape)
        torch.Size([2, 4, 3])
    """

    _INTERNAL_TENSORS = [
        "_points_padded",
        "_normals_padded",
        "_colors_padded",
        "_features_padded",
        "_num_points_per_pointcloud",
    ]

    def __init__(
        self,
        points: Union[List[torch.Tensor], torch.Tensor, None],
        normals: Union[List[torch.Tensor], torch.Tensor, None] = None,
        colors: Union[List[torch.Tensor], torch.Tensor, None] = None,
        features: Union[List[torch.Tensor], torch.Tensor, None] = None,
    ):
        # input types: list or tensor
        if not (isinstance(points, list) or torch.is_tensor(points)):
            msg = "Expected points to be of type list or tensor; got %r"
            raise TypeError(msg % type(points))
        if not (normals is None or isinstance(normals, type(points))):
            msg = "Expected normals to be of type %r; got %r"
            raise TypeError(msg % (type(points), type(normals)))
        if not (colors is None or isinstance(colors, type(points))):
            msg = "Expected colors to be of type %r; got %r"
            raise TypeError(msg % (type(points), type(colors)))
        if not (features is None or isinstance(features, type(points))):
            msg = "Expected features to be of type %r; got %r"
            raise TypeError(msg % (type(points), type(features)))

        if len(points) == 0:
            raise ValueError("len(points) (= 0) should be > 0")

        self.device = points[0].device

        self._points_list = None
        self._normals_list = None
        self._colors_list = None
        self._features_list = None

        self._points_padded = None
        self._normals_padded = None
        self._colors_padded = None
        self._features_padded = None

        self._has_normals = None
        self._has_colors = None
        self._has_features = None

        self._num_points_per_pointcloud = None

        self.equisized = False

        if isinstance(points, list):
            self._points_list = points
            points_shape_per_pointcloud = [p.shape for p in points]
            num_points_per_pointcloud = [x[0] for x in points_shape_per_pointcloud]

            # points shape check
            if 0 in num_points_per_pointcloud:
                raise ValueError("cannot have empty tensors in list of points")
            if set([x[-1] for x in points_shape_per_pointcloud]) != set([3]):
                raise ValueError("last dim of points should have shape 3 (X, Y, Z)")

            # attributes shape check
            if not (
                normals is None
                or [n.shape for n in normals] == points_shape_per_pointcloud
            ):
                raise ValueError(
                    "normals tensors should have same shape as points tensors, but didn't"
                )
            if not (
                colors is None
                or [c.shape for c in colors] == points_shape_per_pointcloud
            ):
                raise ValueError(
                    "colors tensors should have same shape as points tensors, but didn't"
                )
            if not (
                features is None
                or [len(f) for f in features] == num_points_per_pointcloud
            ):
                raise ValueError(
                    "number of features per pointcloud has to be equal to number of points"
                )
            if not (features is None or len(set([f.shape[-1] for f in features])) == 1):
                raise ValueError("number of features per pointcloud has to be the same")

            self._normals_list = (
                None if normals is None else [n.to(self.device) for n in normals]
            )
            self._colors_list = (
                None if colors is None else [c.to(self.device) for c in colors]
            )
            self._features_list = (
                None if features is None else [f.to(self.device) for f in features]
            )

            self._B = len(self._points_list)
            self._num_points_per_pointcloud = torch.tensor(
                num_points_per_pointcloud, device=self.device
            )
            self._N = self._num_points_per_pointcloud.max()
            self.equisized = len(self._num_points_per_pointcloud.unique()) == 1

        elif torch.is_tensor(points):
            # check points shape (B, N, 3)
            if points.size(2) != 3:
                raise ValueError("points tensor has incorrect dimensions.")
            if points.shape[-1] != 3:
                msg = (
                    "last dim of points should have shape 3 (X, Y, Z) but had shape %r"
                )
                raise ValueError(msg % (points.shape[-1]))

            # check attribute shapes match points shape
            if not (normals is None or normals.shape == points.shape):
                msg = "normals tensor should have same shape as points tensor, but didn't: %r != %r"
                raise ValueError(msg % (normals.shape, points.shape))
            if not (colors is None or colors.shape == points.shape):
                msg = "colors tensor should have same shape as points tensor, but didn't: %r != %r"
                raise ValueError(msg % (colors.shape, points.shape))
            if not (features is None or features.shape[:-1] == points.shape[:-1]):
                msg = "first 2 dims of features tensor and points tensor should have same shape, but didn't: %r != %r"
                raise ValueError(msg % (features.shape[:-1], points.shape[:-1]))

            self._points_padded = points
            self._normals_padded = None if normals is None else normals.to(self.device)
            self._colors_padded = None if colors is None else colors.to(self.device)
            self._features_padded = (
                None if features is None else features.to(self.device)
            )
            self._B = self._points_padded.shape[0]
            self._N = self._points_padded.shape[1]
            self._num_points_per_pointcloud = torch.tensor(
                [self._N for _ in range(self._B)], device=self.device
            )
            self.equisized = True

        else:
            raise ValueError(
                "points must be either a list or a tensor with shape (batch_size, N, 3) where N is either the maximum \
                    number of points."
            )

    def __len__(self):
        return self._B

    def __getitem__(self, index):
        r"""
        Args:
            index (int or slice or list of int): Specifying the index of the mesh to retrieve.
            Can be an int, slice, list of ints or a boolean tensor.

        Returns:
            gradslam.Pointclouds: selected pointclouds. The pointclouds tensors are not cloned.
        """
        if isinstance(index, (int, slice)):
            points = self.points_list()[index]
            normals = self.normals_list()[index] if self.has_normals() else None
            colors = self.colors_list()[index] if self.has_colors() else None
            features = self.features_list()[index] if self.has_features() else None
        elif isinstance(index, list):
            points = [self.points_list()[i] for i in index]
            normals = (
                [self.normals_list()[i] for i in index] if self.has_normals() else None
            )
            colors = (
                [self.colors_list()[i] for i in index] if self.has_colors() else None
            )
            features = (
                [self.features_list()[i] for i in index]
                if self.has_features()
                else None
            )
        elif isinstance(index, torch.Tensor):
            if index.dim() != 1 or index.dtype.is_floating_point:
                raise IndexError(index)
            if index.dtype == torch.bool:
                index = index.nonzero()
                index = index.squeeze(1) if index.numel() > 0 else index
                index = index.tolist()
            points = [self.points_list()[i] for i in index]
            normals = (
                [self.normals_list()[i] for i in index] if self.has_normals() else None
            )
            colors = (
                [self.colors_list()[i] for i in index] if self.has_colors() else None
            )
            features = (
                [self.features_list()[i] for i in index]
                if self.has_features()
                else None
            )
        else:
            raise IndexError(index)

        if isinstance(points, list):
            return Pointclouds(
                points=points, normals=normals, colors=colors, features=features
            )
        elif torch.is_tensor(points):
            points = [points]
            normals = None if normals is None else [normals]
            colors = None if colors is None else [colors]
            features = None if features is None else [features]
            return Pointclouds(
                points=points, normals=normals, colors=colors, features=features
            )
        else:
            raise ValueError("points not defined correctly")

    def has_normals(self):
        r"""Determines whether pointclouds have normals or not

        Returns:
            bool
        """
        if self._has_normals is None:
            self._has_normals = (
                self._normals_list is not None or self._normals_padded is not None
            )
        return self._has_normals

    def has_colors(self):
        r"""Determines whether pointclouds have colors or not

        Returns:
            bool
        """
        if self._has_colors is None:
            self._has_colors = (
                self._colors_list is not None or self._colors_padded is not None
            )
        return self._has_colors

    def has_features(self):
        r"""Determines whether pointclouds have features or not

        Returns:
            bool
        """
        if self._has_features is None:
            self._has_features = (
                self._features_list is not None or self._features_padded is not None
            )
        return self._has_features

    def points_list(self):
        r"""Get the list representation of the points.

        Returns:
            list of torch.Tensor: list of :math:`B` tensors of points of shape :math:`(N_b, 3)`.
        """
        if self._points_list is None:
            assert (
                self._points_padded is not None
            ), "points_padded is required to compute points_list."
            self._points_list = [
                p[0] for p in self._points_padded.split([1] * self._B, 0)
            ]
        return self._points_list

    def normals_list(self):
        r"""Get the list representation of the point normals.

        Returns:
            list of torch.Tensor: list of :math:`B` tensors of normals of shape :math:`(N_b, 3)`.
        """
        if self._normals_list is None and self._normals_padded is not None:
            self._normals_list = [
                n[0] for n in self._normals_padded.split([1] * self._B, 0)
            ]
        return self._normals_list

    def colors_list(self):
        r"""Get the list representation of the point colors.

        Returns:
            list of torch.Tensor: list of :math:`B` tensors of colors of shape :math:`(N_b, 3)`.
        """
        if self._colors_list is None and self._colors_padded is not None:
            self._colors_list = [
                c[0] for c in self._colors_padded.split([1] * self._B, 0)
            ]
        return self._colors_list

    def features_list(self):
        r"""Get the list representation of the point features.

        Returns:
            list of torch.Tensor: list of :math:`B` tensors of features of shape :math:`(N_b, 3)`.
        """
        if self._features_list is None and self._features_padded is not None:
            self._features_list = [
                f[0] for f in self._features_padded.split([1] * self._B, 0)
            ]
        return self._features_list

    def points_padded(self):
        r"""Get the padded representation of the points.

        Returns:
            torch.Tensor: tensor representation of points with zero padding as required

        Shape:
            - Output: :math:`(B, max(N_b), 3)`
        """
        self._compute_padded()
        return self._points_padded

    def normals_padded(self):
        r"""Get the padded representation of the normals.

        Returns:
            torch.Tensor: tensor representation of normals with zero padding as required

        Shape:
            - Output: :math:`(B, max(N_b), 3)`
        """
        self._compute_padded()
        return self._normals_padded

    def colors_padded(self):
        r"""Get the padded representation of the colors.

        Returns:
            torch.Tensor: tensor representation of colors with zero padding as required

        Shape:
            - Output: :math:`(B, max(N_b), 3)`
        """
        self._compute_padded()
        return self._colors_padded

    def features_padded(self):
        r"""Get the padded representation of the features.

        Returns:
            torch.Tensor: tensor representation of features with zero padding as required

        Shape:
            - Output: :math:`(B, max(N_b), C)`
        """
        self._compute_padded()
        return self._features_padded

    def num_points_per_pointcloud(self):
        r"""Return a 1D tensor with length equal to the number of pointclouds giving the number of points in each
        pointcloud.

        Returns:
            torch.Tensor: 1D tensor of sizes

        Shapes:
            - Output: tensor of shape :math:`(B)`.
        """
        self._compute_padded()
        return self._num_points_per_pointcloud

    def _compute_padded(self, refresh: bool = False):
        r"""Computes the padded version of pointclouds.

        Args:
            refresh (bool): If True, will recompute padded representation even if it already exists
        """
        if not (refresh or self._points_padded is None):
            return

        self._points_padded = structutils.list_to_padded(
            self._points_list, (self._N, 3), pad_value=0.0, equisized=self.equisized,
        )
        self._normals_padded = (
            None
            if self._normals_list is None
            else structutils.list_to_padded(
                self._normals_list,
                (self._N, 3),
                pad_value=0.0,
                equisized=self.equisized,
            )
        )
        self._colors_padded = (
            None
            if self._colors_list is None
            else structutils.list_to_padded(
                self._colors_list,
                (self._N, 3),
                pad_value=0.0,
                equisized=self.equisized,
            )
        )
        self._features_padded = (
            None
            if self._features_list is None
            else structutils.list_to_padded(
                self._features_list,
                (self._N, 3),
                pad_value=0.0,
                equisized=self.equisized,
            )
        )

    def clone(self):
        r"""Deep copy of Pointclouds object. All internal tensors are cloned individually.

        Returns:
            gradslam.Pointclouds: cloned gradslam.Pointclouds object
        """
        if self._points_list is not None:
            new_points = [p.clone() for p in self.points_list()]
            new_normals = (
                [n.clone() for n in self.normals_list()] if self.has_normals() else None
            )
            new_colors = (
                [c.clone() for c in self.colors_list()] if self.has_colors() else None
            )
            new_features = (
                [f.clone() for f in self.features_list()]
                if self.has_features()
                else None
            )
        elif self._points_padded is not None:
            new_points = self._points_padded.clone()
            new_normals = self._normals_padded.clone()
            new_colors = self._colors_padded.clone()
            new_features = self._features_padded.clone()
        other = Pointclouds(
            points=new_points,
            normals=new_normals,
            colors=new_colors,
            features=new_features,
        )
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
        return other

    def detach(self):
        r"""Detach Pointclouds object. All internal tensors are detached individually.

        Returns:
            gradslam.Pointclouds: detached gradslam.Pointclouds object
        """
        other = self.clone()
        if other._B > 0:
            other._points_list = [p.detach() for p in other._points_list]
            if other._normals_list is not None:
                other._normals_list = [n.detach() for n in other._normals_list]
            if other._colors_list is not None:
                other._colors_list = [c.detach() for c in other._colors_list]
            if other._features_list is not None:
                other._features_list = [f.detach() for f in other._features_list]
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.detach())
        return other

    def to(self, device, copy: bool = False):
        r"""Match functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the returned tensor is a copy of self with the
        desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device, then self is returned.

        Args:
            device (str): Device id for the new tensor.
            copy (bool): Boolean indicator whether or not to clone self. Default False.

        Returns:
            gradslam.Pointclouds
        """
        if not copy and self.device == device:
            return self

        other = self.clone()
        if self.device != device:
            # hack to know which gpu is used when device("cuda")
            other.device = torch.Tensor().to(device).device
            if other._points_list is not None:
                other._points_list = [p.to(device) for p in other._points_list]
                if other._normals_list is not None:
                    other._normals_list = [n.to(device) for n in other._normals_list]
                if other._colors_list is not None:
                    other._colors_list = [c.to(device) for c in other._colors_list]
                if other._features_list is not None:
                    other._features_list = [f.to(device) for f in other._features_list]
            for k in self._INTERNAL_TENSORS:
                v = getattr(self, k)
                if torch.is_tensor(v):
                    setattr(other, k, v.to(device))
        return other

    def cpu(self):
        r"""Match functionality of torch.Tensor.cpu()

        Returns:
            gradslam.Pointclouds
        """
        return self.to(torch.device("cpu"))

    def cuda(self):
        r"""Match functionality of torch.Tensor.cuda()

        Returns:
            gradslam.Pointclouds
        """
        return self.to(torch.device("cuda"))
