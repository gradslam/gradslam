import os
import warnings
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from ..geometry import projutils
from . import structutils

__all__ = ["Pointclouds"]


class Pointclouds(object):
    r"""Batch of pointclouds (with varying numbers of points), enabling conversion between 2 representations:

    - List: Store points of each pointcloud of shape :math:`(N_b, 3)` in a list of length :math:`B`.
    - Padded: Store all points in a :math:`(B, max(N_b), 3)` tensor with zero padding as required.

    Args:
        points (torch.Tensor or list of torch.Tensor or None): :math:`(X, Y, Z)` coordinates of each point.
            Default: None
        normals (torch.Tensor or list of torch.Tensor or None): Normals :math:`(N_x, N_y, N_z)` of each point.
            Default: None
        colors (torch.Tensor or list of torch.Tensor or None): :math:`(R, G, B)` color of each point.
            Default: None
        features (torch.Tensor or list of torch.Tensor or None): :math:`C` features of each point.
            Default: None
        device (torch.device or str or None): The desired device of internal tensors. If None, sets device to be
            same as `points` device. Default: None

    Shape:
        - points: Can either be a list of tensors of shape :math:`(N_b, 3)` or a padded tensor of shape
          :math:`(B, N, 3)`.
        - normals: Can either be a list of tensors of shape :math:`(N_b, 3)` or a padded tensor of shape
          :math:`(B, N, 3)`.
        - colors: Can either be a list of tensors of shape :math:`(N_b, 3)` or a padded tensor of shape
          :math:`(B, N, 3)`.
        - features: Can either be a list of tensors of shape :math:`(N_b, C)` or a padded tensor of shape
          :math:`(B, N, C)`.

    Examples::

        >>> points_list = [torch.rand(1, 3), torch.rand(4, 3)]
        >>> pcs1 = gradslam.Pointclouds(points_list)
        >>> print(pcs1.points_padded.shape)
        torch.Size([2, 4, 3])
        >>> print(len(pcs1.points_list))
        2
        >>> pcs2 = gradslam.Pointclouds(torch.rand((2, 4, 3)))
        >>> print(pcs2.points_padded.shape)
        torch.Size([2, 4, 3])
    """

    _INTERNAL_TENSORS = [
        "_points_padded",
        "_normals_padded",
        "_colors_padded",
        "_features_padded",
        "_embeddings_padded",  # KM
        "_confidences_padded",
        "_nonpad_mask",
        "_num_points_per_pointcloud",
    ]

    def __init__(
        self,
        points: Union[List[torch.Tensor], torch.Tensor, None] = None,
        normals: Union[List[torch.Tensor], torch.Tensor, None] = None,
        colors: Union[List[torch.Tensor], torch.Tensor, None] = None,
        features: Union[List[torch.Tensor], torch.Tensor, None] = None,
        device: Union[torch.device, str, None] = None,
        embeddings: Union[List[torch.Tensor], torch.Tensor, None] = None,  # KM
        confidences: Union[List[torch.Tensor], torch.Tensor, None] = None,
    ):
        super().__init__()

        # input types: list or tensor or None
        if not (points is None or isinstance(points, list) or torch.is_tensor(points)):
            msg = "Expected points to be of type list or tensor or None; got %r"
            raise TypeError(msg % type(points))
        if not (normals is None or isinstance(normals, type(points))):
            msg = "Expected normals to be of same type as points (%r); got %r"
            raise TypeError(msg % (type(points), type(normals)))
        if not (colors is None or isinstance(colors, type(points))):
            msg = "Expected colors to be of same type as points (%r); got %r"
            raise TypeError(msg % (type(points), type(colors)))
        if not (features is None or isinstance(features, type(points))):
            msg = "Expected features to be of same type as points (%r); got %r"
            raise TypeError(msg % (type(points), type(features)))

        if points is not None and len(points) == 0:
            raise ValueError("len(points) (= 0) should be > 0")

        self._points_list = None
        self._normals_list = None
        self._colors_list = None
        self._features_list = None
        self._embeddings_list = None  # KM
        self._confidences_list = None

        self._points_padded = None
        self._normals_padded = None
        self._colors_padded = None
        # _features_padded actually serves as the map_counts
        self._features_padded = None

        self._embeddings_padded = None  # KM
        # _confidences_padded serves the map_counts for per-point emebddings
        self._confidences_padded = None

        self._nonpad_mask = None

        self._has_points = None
        self._has_normals = None
        self._has_colors = None
        self._has_features = None
        self._has_embeddings = None  # KM
        self._has_confidences = None

        self._num_points_per_pointcloud = None

        self.equisized = False

        if isinstance(points, list):
            # points shape check
            points_shape_per_pointcloud = [p.shape for p in points]
            if any([p.ndim != 2 for p in points]):
                raise ValueError("ndim of all tensors in points list should be 2")
            if any([x[-1] != 3 for x in points_shape_per_pointcloud]):
                raise ValueError(
                    "last dim of all tensors in points should have shape 3 (X, Y, Z)"
                )

            self.device = (
                torch.Tensor().to(device).device
                if device is not None
                else points[0].device
            )
            self._points_list = [p.to(self.device) for p in points]
            num_points_per_pointcloud = [x[0] for x in points_shape_per_pointcloud]

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
            if not (features is None or all([f.ndim == 2 for f in features])):
                raise ValueError("ndim of all tensors in features list should be 2")
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

            # KM
            self._embeddings_list = (
                None if embeddings is None else [e.to(self.device) for e in embeddings]
            )

            self._confidences_list = (
                None
                if confidences is None
                else [c.to(self.device) for c in confidences]
            )

            self._B = len(self._points_list)
            self._num_points_per_pointcloud = torch.tensor(
                num_points_per_pointcloud, device=self.device
            )
            self._N = self._num_points_per_pointcloud.max().item()
            self.equisized = len(self._num_points_per_pointcloud.unique()) == 1

        elif torch.is_tensor(points):
            self.device = (
                torch.Tensor().to(device).device
                if device is not None
                else points.device
            )
            # check points shape (B, N, 3)
            if points.ndim != 3:
                msg = "points should have ndim=3, but had ndim={}".format(points.ndim)
                raise ValueError(msg)
            if points.shape[-1] != 3:
                msg = (
                    "last dim of points should have shape 3 (X, Y, Z) but had shape %r"
                )
                raise ValueError(msg % (points.shape[-1]))
            if points.shape[0] == 0:
                msg = "Batch size of 0 not supported yet. Got input points shape {}.".format(
                    points.shape
                )
                raise ValueError(msg)

            # check attribute shapes match points shape
            if not (normals is None or normals.shape == points.shape):
                msg = "normals tensor should have same shape as points tensor, but didn't: %r != %r"
                raise ValueError(msg % (normals.shape, points.shape))
            if not (colors is None or colors.shape == points.shape):
                msg = "colors tensor should have same shape as points tensor, but didn't: %r != %r"
                raise ValueError(msg % (colors.shape, points.shape))
            if not (features is None or features.ndim == 3):
                msg = "features should have ndim=3, but had ndim={}".format(
                    features.ndim
                )
                raise ValueError(msg)
            if not (features is None or features.shape[:-1] == points.shape[:-1]):
                msg = "first 2 dims of features tensor and points tensor should have same shape, but didn't: %r != %r"
                raise ValueError(msg % (features.shape[:-1], points.shape[:-1]))

            self._points_padded = points.to(self.device)
            self._normals_padded = None if normals is None else normals.to(self.device)
            self._colors_padded = None if colors is None else colors.to(self.device)
            self._features_padded = (
                None if features is None else features.to(self.device)
            )
            self._embeddings_padded = (
                None if embeddings is None else embeddings.to(self.device)
            )  # KM
            self._confidences_padded = (
                None if confidences is None else confidences.to(self.device)
            )
            self._B = self._points_padded.shape[0]
            self._N = self._points_padded.shape[1]
            self._num_points_per_pointcloud = torch.tensor(
                [self._N for _ in range(self._B)], device=self.device
            )
            self.equisized = True

        elif points is None:
            self.device = (
                torch.Tensor().to(device).device
                if device is not None
                else torch.device("cpu")
            )
            self._B = 0
            self._N = 0
            self._num_points_per_pointcloud = torch.tensor([0], device=self.device)
            self.equisized = None

        else:
            raise ValueError(
                "points must either be None, a list, or a tensor with shape (batch_size, N, 3) where N is \
                    the maximum number of points."
            )

    @classmethod
    def load_pointcloud_from_h5(
        cls,
        saved_path: str,
    ):
        r"""Loads pointcloud from disk.

        Args:
            saved_path (str): Path to load the pointcloud from.
        """
        import h5py

        # if saved_path doesn't end with a folder names "pointclouds", append it
        if not Path(saved_path).name == "pointclouds":
            saved_path = f"{saved_path}/pointclouds"

        # now check if the folder exists
        if not Path(saved_path).exists():
            raise FileNotFoundError(f"{saved_path} does not exist")

        # Check if the h5 files exist in the folder
        points_path = f"{saved_path}/pc_points.h5"
        colors_path = f"{saved_path}/pc_colors.h5"
        features_path = f"{saved_path}/pc_features.h5"
        embeddings_path = f"{saved_path}/pc_embeddings.h5"
        confidences_path = f"{saved_path}/pc_confidences.h5"

        load_points, load_colors, load_features, load_embeddings, load_confidences = (
            True,
            True,
            True,
            True,
            True,
        )

        if not os.path.exists(points_path):
            raise FileNotFoundError(f"{points_path} does not exist")
            load_points = False
        if not os.path.exists(colors_path):
            warnings.warn(f"Could not find pointcloud colors: {colors_path}. Skipping")
            load_colors = False
        if not os.path.exists(features_path):
            warnings.warn(
                f"Could not find pointcloud features: {features_path}. Skipping"
            )
            load_features = False
        if not os.path.exists(embeddings_path):
            warnings.warn(
                f"Could not find pointcloud embeddings: {embeddings_path}. Skipping"
            )
            load_embeddings = False
        if not os.path.exists(confidences_path):
            warnings.warn(
                f"Could not find pointcloud confidences: {confidences_path}. Skipping"
            )
            load_confidences = False

        pc_points, pc_colors, pc_features, pc_embeddings, pc_confidences = (
            None,
            None,
            None,
            None,
            None,
        )

        # Load the tensors from the h5 files using h5py
        if load_points:
            with h5py.File(points_path, "r") as f:
                pc_points = torch.from_numpy(f["pc_points"][:])
        if load_colors:
            with h5py.File(colors_path, "r") as f:
                pc_colors = torch.from_numpy(f["pc_colors"][:])
        if load_features:
            with h5py.File(features_path, "r") as f:
                pc_features = torch.from_numpy(f["pc_features"][:])
        if load_embeddings:
            with h5py.File(embeddings_path, "r") as f:
                pc_embeddings = torch.from_numpy(f["pc_embeddings"][:])
        if load_confidences:
            with h5py.File(confidences_path, "r") as f:
                pc_confidences = torch.from_numpy(f["pc_confidences"][:])

        # add a dimension to pc_points, colors, features, embeddings
        pc_points = pc_points.unsqueeze(0)
        if pc_colors is not None:
            pc_colors = pc_colors.unsqueeze(0)
        if pc_features is not None:
            pc_features = pc_features.unsqueeze(0)
        if pc_embeddings is not None:
            pc_embeddings = pc_embeddings.unsqueeze(0)
        if pc_confidences is not None:
            pc_confidences = pc_confidences.unsqueeze(0)

        return cls(
            points=pc_points,
            colors=pc_colors,
            features=pc_features,
            embeddings=pc_embeddings,
            confidences=pc_confidences,
        )

    def __len__(self):
        return self._B

    def __getitem__(self, index):
        r"""
        Args:
            index (int or slice or list of int or torch.Tensor): Specifying the index of the pointclouds to retrieve.
                Can be an int, slice, list of ints or a boolean tensor.

        Returns:
            gradslam.Pointclouds: Selected pointclouds. The pointclouds tensors are not cloned.
        """
        if not self.has_points:
            raise IndexError("Cannot index empty pointclouds object")
        if isinstance(index, (int, slice)):
            points = self.points_list[index]
            normals = self.normals_list[index] if self.has_normals else None
            colors = self.colors_list[index] if self.has_colors else None
            features = self.features_list[index] if self.has_features else None
            embeddings = (
                self.embeddings_list[index] if self.has_embeddings else None
            )  # KM
            confidences = self.confidences_list[index] if self.has_confidences else None
        elif isinstance(index, list):
            points = [self.points_list[i] for i in index]
            normals = (
                [self.normals_list[i] for i in index] if self.has_normals else None
            )
            colors = [self.colors_list[i] for i in index] if self.has_colors else None
            features = (
                [self.features_list[i] for i in index] if self.has_features else None
            )
            # KM
            embeddings = (
                [self.embeddings_list[i] for i in index]
                if self.has_embeddings
                else None
            )
            confidences = (
                [self.confidences_list[i] for i in index]
                if self.has_confidences
                else None
            )
        elif isinstance(index, torch.Tensor):
            if index.dim() != 1 or index.dtype.is_floating_point:
                raise IndexError(index)
            if index.dtype == torch.bool:
                index = index.nonzero()
                index = index.squeeze(1) if index.numel() > 0 else index
                index = index.tolist()
            points = [self.points_list[i] for i in index]
            normals = (
                [self.normals_list[i] for i in index] if self.has_normals else None
            )
            colors = [self.colors_list[i] for i in index] if self.has_colors else None
            features = (
                [self.features_list[i] for i in index] if self.has_features else None
            )
            # KM
            embeddings = (
                [self.embeddings_list[i] for i in index]
                if self.has_embeddings
                else None
            )
        else:
            raise IndexError(index)

        if isinstance(points, list):
            return Pointclouds(
                points=points,
                normals=normals,
                colors=colors,
                features=features,
                embeddings=embeddings,
                confidences=confidences,
            )
        elif torch.is_tensor(points):
            points = [points]
            normals = None if normals is None else [normals]
            colors = None if colors is None else [colors]
            features = None if features is None else [features]
            embeddings = None if embeddings else [embeddings]  # KM
            confidences = None if confidences is None else [confidences]
            return Pointclouds(
                points=points,
                normals=normals,
                colors=colors,
                features=features,
                embeddings=embeddings,
                confidences=confidences,
            )
        else:
            raise ValueError("points not defined correctly")

    def __add__(self, other):
        r"""Out-of-place implementation of `Pointclouds.offset_`"""
        try:
            return self.clone().offset_(other)
        except TypeError:
            raise NotImplementedError(
                "Pointclouds + {} currently not implemented.".format(type(other))
            )

    def __sub__(self, other):
        r"""Subtracts `other` from all Pointclouds' points (`Pointclouds` - `other`).

        Args:
            other (torch.Tensor or float or int): Value(s) to subtract from all points.

        returns:
            gradslam.Pointclouds: Subtracted Pointclouds
        """
        try:
            return self.clone().offset_(other * -1)
        except TypeError:
            raise NotImplementedError(
                "Pointclouds - {} currently not implemented.".format(type(other))
            )

    def __mul__(self, other):
        r"""Out-of-place implementation of `Pointclouds.scale_`"""
        try:
            return self.clone().scale_(other)
        except TypeError:
            raise NotImplementedError(
                "Pointclouds * {} currently not implemented.".format(type(other))
            )

    def __truediv__(self, other):
        r"""Divides all Pointclouds' points by `other`.

        Args:
            other (torch.Tensor or float or int): Value(s) to divide all points by.

        Returns:
            self

        Shape:
            - other: Any. Must be compatible with :math:`(B, N, 3)`.
        """
        try:
            return self.__mul__(1.0 / other)
        except TypeError:
            raise NotImplementedError(
                "Pointclouds / {} currently not implemented.".format(type(other))
            )

    def __matmul__(self, other):
        r"""Post-multiplication :math:`SE(3)` transformation or :math:`SO(3)` rotation of Pointclouds' points and
        normals.

        Args:
            other (torch.Tensor): Either :math:`SE(3)` transformation or :math:`SO(3)` rotation

        Returns:
            self

        Shape:
            - other: Either :math:`SE(3)` transformation of shape :math:`(4, 4)` or :math:`(B, 4, 4)`, or :math:`SO(3)`
                rotation of shape :math:`(3, 3)` or :math:`(B, 3, 3)`
        """
        if not torch.is_tensor(other):
            raise NotImplementedError(
                "Pointclouds @ {} currently not implemented.".format(type(other))
            )

        if not (
            (other.ndim == 2 or other.ndim == 3)
            and (other.shape[-2:] == (3, 3) or other.shape[-2:] == (4, 4))
        ):
            msg = "Unsupported shape for Pointclouds @ operand: {}\n".format(
                other.shape
            )
            msg += "Use tensor of shape (3, 3) or (B, 3, 3) for rotations, or (4, 4) or (B, 4, 4) for transformations"
            raise ValueError(msg)

        if other.shape[-2:] == (3, 3):
            return self.clone().rotate_(other, pre_multiplication=False)
        if other.shape[-2:] == (4, 4):
            return self.clone().transform_(other, pre_multiplication=False)

    def rotate(self, rmat: torch.Tensor, *, pre_multiplication=True):
        r"""Out-of-place implementation of `Pointclouds.rotate_`"""
        return self.clone().rotate_(rmat, pre_multiplication=pre_multiplication)

    def transform(self, transform: torch.Tensor, *, pre_multiplication=True):
        r"""Out-of-place implementation of `Pointclouds.transform_`"""
        return self.clone().transform_(transform, pre_multiplication=pre_multiplication)

    def pinhole_projection(self, intrinsics: torch.Tensor):
        r"""Out-of-place implementation of `Pointclouds.pinhole_projection_`"""
        return self.clone().pinhole_projection_(intrinsics)

    def offset_(self, offset: Union[torch.Tensor, float, int]):
        r"""Adds :math:`offset` to all Pointclouds' points. In place operation.

        Args:
            offset (torch.Tensor or float or int): Value(s) to add to all points.

        Returns:
            self

        Shape:
            - offset: Any. Must be compatible with :math:`(B, N, 3)`.
        """
        if not (
            torch.is_tensor(offset)
            or isinstance(offset, float)
            or isinstance(offset, int)
        ):
            raise TypeError(
                "Operand should be tensor, float or int but was %r instead"
                % type(offset)
            )
        if not self.has_points:
            return self
        # update padded representation
        self._points_padded = self.points_padded + (
            offset * self.nonpad_mask.to(self.points_padded.dtype).unsqueeze(-1)
        )

        # update list representation when inferred
        self._points_list = None

        return self

    def scale_(self, scale: Union[torch.Tensor, float, int]):
        r"""Scales all Pointclouds' points by `scale`. In place operation.

        Args:
            scale (torch.Tensor or float or int): Value(s) to scale all points by.

        Returns:
            self

        Shape:
            - scale: Any. Must be compatible with :math:`(B, N, 3)`.
        """
        if not (
            torch.is_tensor(scale) or isinstance(scale, float) or isinstance(scale, int)
        ):
            raise TypeError(
                "Operand should be tensor, float or int but was %r instead"
                % type(scale)
            )
        if not self.has_points:
            return self

        # update padded representation
        self._points_padded = (
            self.points_padded
            * scale
            * self.nonpad_mask.to(self.points_padded.dtype).unsqueeze(-1)
        )

        # update list representation when inferred
        self._points_list = None

        return self

    def rotate_(self, rmat: torch.Tensor, *, pre_multiplication=True):
        r"""Applies batch or single :math:`SO(3)` rotation to all Pointclouds' points and normals. In place operation.

        Args:
            rmat (torch.Tensor): Either batch or single :math:`SO(3)` rotation matrix
            pre_multiplication (torch.Tensor): If True, will pre-multiply the rotation. Otherwise will
                post-multiply the rotation. Default: True

        Returns:
            self

        Shape:
            - rmat: :math:`(3, 3)` or :math:`(B, 3, 3)`
        """
        if not torch.is_tensor(rmat):
            raise TypeError(
                "Rotation matrix should be tensor, but was %r instead" % type(rmat)
            )

        if not ((rmat.ndim == 2 or rmat.ndim == 3) and rmat.shape[-2:] == (3, 3)):
            raise ValueError(
                "Rotation matrix should be of shape (3, 3) or (B, 3, 3), but was {} instead.".format(
                    rmat.shape
                )
            )

        if rmat.ndim == 3 and rmat.shape[0] != self._B:
            raise ValueError(
                "Rotation matrix batch size ({}) != Pointclouds batch size ({})".format(
                    rmat.shape[0], self._B
                )
            )
        if not self.has_points:
            return self

        if pre_multiplication:
            rmat = rmat.transpose(-1, -2)

        # update padded representation
        if rmat.ndim == 2:
            self._points_padded = torch.einsum("bij,jk->bik", self.points_padded, rmat)
            self._normals_padded = (
                None
                if self.normals_padded is None
                else torch.einsum("bij,jk->bik", self.normals_padded, rmat)
            )
        elif rmat.ndim == 3:
            self._points_padded = torch.einsum("bij,bjk->bik", self.points_padded, rmat)
            self._normals_padded = (
                None
                if self.normals_padded is None
                else torch.einsum("bij,bjk->bik", self.normals_padded, rmat)
            )

        # force update of list representation
        self._points_list = None
        self._normals_list = None

        return self

    def transform_(self, transform: torch.Tensor, *, pre_multiplication=True):
        r"""Applies batch or single :math:`SE(3)` transformation to all Pointclouds' points and normals. In place
        operation.

        Args:
            transform (torch.Tensor): Either batch or single :math:`SE(3)` transformation tensor
            pre_multiplication (torch.Tensor): If True, will pre-multiply the transformation. Otherwise will
                post-multiply the transformation. Default: True

        Returns:
            self

        Shape:
            - transform: :math:`(4, 4)` or :math:`(B, 4, 4)`
        """
        if not torch.is_tensor(transform):
            raise TypeError(
                "transform should be tensor, but was %r instead" % type(transform)
            )

        if not (
            (transform.ndim == 2 or transform.ndim == 3)
            and transform.shape[-2:] == (4, 4)
        ):
            raise ValueError(
                "transform should be of shape (4, 4) or (B, 4, 4), but was {} instead.".format(
                    transform.shape
                )
            )

        if transform.ndim == 3 and transform.shape[0] != self._B:
            raise ValueError(
                "transform batch size ({}) != Pointclouds batch size ({})".format(
                    transform.shape[0], self._B
                )
            )
        if not self.has_points:
            return self

        # rotation and translation matrix
        rmat = transform[..., :3, :3]
        tvec = transform[..., :3, 3]

        # expand dims to ensure correct broadcasting of offset
        while tvec.ndim < self.points_padded.ndim:
            tvec = tvec.unsqueeze(-2)

        return self.rotate_(rmat, pre_multiplication=pre_multiplication).offset_(tvec)

    def pinhole_projection_(self, intrinsics: torch.Tensor):
        r"""Projects Pointclouds' points onto :math:`z=1` plane using intrinsics of a pinhole camera. In place
        operation.

        Args:
            intrinsics (torch.Tensor): Either batch or single intrinsics matrix

        Returns:
            self

        Shape:
            - intrinsics: :math:`(4, 4)` or :math:`(B, 4, 4)`
        """
        if not torch.is_tensor(intrinsics):
            raise TypeError(
                "intrinsics should be tensor, but was {} instead".format(
                    type(intrinsics)
                )
            )

        if not (
            (intrinsics.ndim == 2 or intrinsics.ndim == 3)
            and intrinsics.shape[-2:] == (4, 4)
        ):
            msg = "intrinsics should be of shape (4, 4) or (B, 4, 4), but was {} instead.".format(
                intrinsics.shape
            )
            raise ValueError(msg)
        if not self.has_points:
            return self

        projected_2d = projutils.project_points(self.points_padded, intrinsics)
        self._points_padded = projutils.homogenize_points(
            projected_2d
        ) * self.nonpad_mask.to(projected_2d.dtype).unsqueeze(-1)

        # force update of list representation
        self._points_list = None

        return self

    @property
    def has_points(self):
        r"""Determines whether pointclouds have points or not

        Returns:
            bool
        """
        if self._has_points is None:
            self._has_points = (
                self._points_list is not None or self._points_padded is not None
            )
        return self._has_points

    @property
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

    @property
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

    @property
    def has_features(self):
        r"""Determines whether pointclouds have features or not

        Returns:
            bool
        """
        # print("In has_features()")
        if self._has_features is None:
            self._has_features = (
                self._features_list is not None or self._features_padded is not None
            )
        return self._has_features

    # KM
    @property
    def has_embeddings(self):
        r"""Determines whether pointclouds have embeddings or not

        Returns:
            bool
        """
        # print("In has_embeddings")
        if self._has_embeddings is None:
            self._has_embeddings = (
                self._embeddings_list is not None or self._embeddings_padded is not None
            )
        return self._has_embeddings

    @property
    def has_confidences(self):
        r"""Determines whether pointclouds have confidences or not

        Returns:
            bool
        """
        if self._has_confidences is None:
            self._has_confidences = (
                self._confidences_list is not None
                or self._confidences_padded is not None
            )
        return self._has_confidences

    @property
    def num_features(self):
        r"""Determines number of features in pointclouds

        Returns:
            int
        """
        if not self.has_features:
            return 0
        if self._features_padded is not None:
            return self._features_padded.shape[-1]
        if self._features_list is not None:
            return self._features_list[0].shape[-1]

    # KM
    @property
    def num_embeddings(self):
        r"""Determines number of dims in embeddings

        Returns:
            int
        """
        if not self.has_embeddings:
            return 0
        if self._embeddings_padded is not None:
            return self._embeddings_padded.shape[-1]
        if self._embeddings_list is not None:
            return self._embeddings_list[0].shape[-1]

    @property
    def num_confidences(self):
        r"""Determines number of confidences in pointclouds

        Returns:
            int
        """
        if not self.has_confidences:
            return 0
        if self._confidences_padded is not None:
            return self._confidences_padded.shape[-1]
        if self._confidences_list is not None:
            return self._confidences_list[0].shape[-1]

    @property
    def points_list(self):
        r"""Gets the list representation of the points.

        Returns:
            list of torch.Tensor: list of :math:`B` tensors of points of shape :math:`(N_b, 3)`.
        """
        if self._points_list is None and self._points_padded is not None:
            self._points_list = [
                p[0, : self._num_points_per_pointcloud[b]]
                for b, p in enumerate(self._points_padded.split([1] * self._B, 0))
            ]
        return self._points_list

    @property
    def normals_list(self):
        r"""Gets the list representation of the point normals.

        Returns:
            list of torch.Tensor: list of :math:`B` tensors of normals of shape :math:`(N_b, 3)`.
        """
        if self._normals_list is None and self._normals_padded is not None:
            self._normals_list = [
                n[0, : self._num_points_per_pointcloud[b]]
                for b, n in enumerate(self._normals_padded.split([1] * self._B, 0))
            ]
        return self._normals_list

    @property
    def colors_list(self):
        r"""Gets the list representation of the point colors.

        Returns:
            list of torch.Tensor: list of :math:`B` tensors of colors of shape :math:`(N_b, 3)`.
        """
        if self._colors_list is None and self._colors_padded is not None:
            self._colors_list = [
                c[0, : self._num_points_per_pointcloud[b]]
                for b, c in enumerate(self._colors_padded.split([1] * self._B, 0))
            ]
        return self._colors_list

    @property
    def features_list(self):
        r"""Gets the list representation of the point features.

        Returns:
            list of torch.Tensor: list of :math:`B` tensors of features of shape :math:`(N_b, 3)`.
        """
        if self._features_list is None and self._features_padded is not None:
            self._features_list = [
                f[0, : self._num_points_per_pointcloud[b]]
                for b, f in enumerate(self._features_padded.split([1] * self._B, 0))
            ]
        return self._features_list

    # KM
    @property
    def embeddings_list(self):
        if self._embeddings_list is None and self._embeddings_padded is not None:
            self._embeddings_list = [
                f[0, : self._num_points_per_pointcloud[b]]
                for b, f in enumerate(self._embeddings_padded.split([1] * self._B, 0))
            ]
        return self._embeddings_list

    @property
    def confidences_list(self):
        if self._confidences_list is None and self._confidences_padded is not None:
            self._confidences_list = [
                f[0, : self._num_points_per_pointcloud[b]]
                for b, f in enumerate(self._confidences_padded.split([1] * self._B, 0))
            ]
        return self._confidences_list

    @property
    def points_padded(self):
        r"""Gets the padded representation of the points.

        Returns:
            torch.Tensor: tensor representation of points with zero padding as required

        Shape:
            - Output: :math:`(B, max(N_b), 3)`
        """
        self._compute_padded()
        return self._points_padded

    @property
    def normals_padded(self):
        r"""Gets the padded representation of the normals.

        Returns:
            torch.Tensor: tensor representation of normals with zero padding as required

        Shape:
            - Output: :math:`(B, max(N_b), 3)`
        """
        self._compute_padded()
        return self._normals_padded

    @property
    def colors_padded(self):
        r"""Gets the padded representation of the colors.

        Returns:
            torch.Tensor: tensor representation of colors with zero padding as required

        Shape:
            - Output: :math:`(B, max(N_b), 3)`
        """
        self._compute_padded()
        return self._colors_padded

    @property
    def features_padded(self):
        r"""Gets the padded representation of the features.

        Returns:
            torch.Tensor: tensor representation of features with zero padding as required

        Shape:
            - Output: :math:`(B, max(N_b), C)`
        """
        self._compute_padded()
        return self._features_padded

    # KM
    @property
    def embeddings_padded(self):
        self._compute_padded()
        return self._embeddings_padded

    @property
    def confidences_padded(self):
        self._compute_padded()
        return self._confidences_padded

    @property
    def nonpad_mask(self):
        r"""Returns tensor of `bool` values which are True wherever points exist and False wherever there is padding.

        Returns:
            torch.Tensor: 2d `bool` mask

        Shape:
            - Output: :math:`(B, N)`
        """
        if self._nonpad_mask is None and self.has_points:
            self._nonpad_mask = torch.ones(
                (self._B, self._N), dtype=torch.bool, device=self.device
            )
            if self.equisized:
                self._nonpad_mask[:, self._num_points_per_pointcloud[0] :] = 0
            else:
                for b in range(self._B):
                    self._nonpad_mask[b, self._num_points_per_pointcloud[b] :] = 0
        return self._nonpad_mask

    @property
    def num_points_per_pointcloud(self):
        r"""Returns a 1D tensor with length equal to the number of pointclouds giving the number of points in each
        pointcloud.

        Returns:
            torch.Tensor: 1D tensor of sizes

        Shape:
            - Output: tensor of shape :math:`(B)`.
        """
        return self._num_points_per_pointcloud

    @points_list.setter
    def points_list(self, value: List[torch.Tensor]):
        r"""Updates `points_list` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change.

        Args:
            value (list of torch.Tensor): list of :math:`B` tensors of points of shape :math:`(N_b, 3)`.
                Shape of tensors in `value` and `pointclouds.points_list` must match.

        """
        self._assert_set_list(value)
        self._points_list = [v.clone().to(self.device) for v in value]
        self._points_padded = None

    @normals_list.setter
    def normals_list(self, value: List[torch.Tensor]):
        r"""Updates `normals_list` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change.

        Args:
            value (list of torch.Tensor): list of :math:`B` tensors of points of shape :math:`(N_b, 3)`.
                Shape of tensors in `value` and `pointclouds.points_list` must match.

        """
        self._assert_set_list(value)
        self._normals_list = [v.clone().to(self.device) for v in value]
        self._noramls_padded = None

    @colors_list.setter
    def colors_list(self, value: List[torch.Tensor]):
        r"""Updates `colors_list` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change.

        Args:
            value (list of torch.Tensor): list of :math:`B` tensors of points of shape :math:`(N_b, 3)`.
                Shape of tensors in `value` and `pointclouds.points_list` must match.

        """
        self._assert_set_list(value)
        self._colors_list = [v.clone().to(self.device) for v in value]
        self._noramls_padded = None  # TODO: fix typo / bug

    @features_list.setter
    def features_list(self, value: List[torch.Tensor]):
        r"""Updates `features_list` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change.

        Args:
            value (list of torch.Tensor): list of :math:`B` tensors of points of shape :math:`(N_b, C)`.
                Shape of tensors in `value` and `pointclouds.points_list` must match.

        """
        self._assert_set_list(value, first_dim_only=True)
        self._features_list = [v.clone().to(self.device) for v in value]
        self._noramls_padded = None  # TODO: fix typo / bug

    # KM
    @embeddings_list.setter
    def embeddings_list(self, value: List[torch.Tensor]):
        self._assert_set_list(value, first_dim_only=True)
        self._embeddings_list = [v.clone().to(self.device) for v in value]

    @confidences_list.setter
    def confidences_list(self, value: List[torch.Tensor]):
        self._assert_set_list(value, first_dim_only=True)
        self._confidences_list = [v.clone().to(self.device) for v in value]

    @points_padded.setter
    def points_padded(self, value: torch.Tensor):
        r"""Updates `points_padded` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change
            (can not change the shape or padding of `points_padded`).

        Args:
            value (torch.Tensor): tensor representation of (zero padded) points with the same shape and number of
                points per pointcloud as `self.points_padded`

        Shape:
            - value: :math:`(B, max(N_b), 3)`
        """
        self._assert_set_padded(value)
        self._points_padded = value.clone().to(self.device)
        self._points_list = None

    @normals_padded.setter
    def normals_padded(self, value: torch.Tensor):
        r"""Updates `normals_padded` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change
            (can not change the shape or padding of `normals_padded`).

        Args:
            value (torch.Tensor): tensor representation of (zero padded) normals with the same shape and number of
                points per pointcloud as `self.points_padded`

        Shape:
            - value: :math:`(B, max(N_b), 3)`
        """
        self._assert_set_padded(value)
        self._normals_padded = value.clone().to(self.device)
        self._normals_list = None

    @colors_padded.setter
    def colors_padded(self, value: torch.Tensor):
        r"""Updates `colors_padded` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change
            (can not change the shape or padding of `colors_padded`).

        Args:
            value (torch.Tensor): tensor representation of (zero padded) colors with the same shape and number of
                points per pointcloud as `self.points_padded`

        Shape:
            - value: :math:`(B, max(N_b), 3)`
        """
        self._assert_set_padded(value)
        self._colors_padded = value.clone().to(self.device)
        self._colors_list = None

    @features_padded.setter
    def features_padded(self, value: torch.Tensor):
        r"""Updates `features_padded` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change
            (can not change the shape or padding of `features_padded`).

        Args:
            value (torch.Tensor): tensor representation of (zero padded) features with the same shape and number of
                points per pointcloud as `self.points_padded`

        Shape:
            - value: :math:`(B, max(N_b), C)`
        """
        self._assert_set_padded(value, first_2_dims_only=True)
        self._features_padded = value.clone().to(self.device)
        self._features_list = None

    # KM
    @embeddings_padded.setter
    def embeddings_padded(self, value: torch.Tensor):
        self._assert_set_padded(value, first_2_dims_only=True)
        self._embeddings_padded = value.clone().to(self.device)
        self._embeddings_list = None

    @confidences_padded.setter
    def confidences_padded(self, value: torch.Tensor):
        r"""Updates `confidences_padded` representation.
        .. note:: The number of pointclouds and the number of points per pointcloud can not change
            (can not change the shape or padding of `confidences_padded`).

        Args:
            value (torch.Tensor): tensor representation of (zero padded) confidences with the same shape and number of
                points per pointcloud as `self.points_padded`

        Shape:
            - value: :math:`(B, max(N_b))`
        """
        self._assert_set_padded(value, first_2_dims_only=True)
        self._confidences_padded = value.clone().to(self.device)
        self._confidences_list = None

    def _compute_padded(self, refresh: bool = False):
        r"""Computes the padded version of pointclouds.

        Args:
            refresh (bool): If True, will recompute padded representation even if it already exists
        """
        if not self.has_points:
            return

        if not (refresh or self._points_padded is None):
            return

        self._points_padded = structutils.list_to_padded(
            self._points_list,
            (self._N, 3),
            pad_value=0.0,
            equisized=self.equisized,
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
                (self._N, self.num_features),
                pad_value=0.0,
                equisized=self.equisized,
            )
        )
        # KM
        self._embeddings_padded = (
            None
            if self._embeddings_list is None
            else structutils.list_to_padded(
                self._embeddings_list,
                (self._N, self.num_embeddings),
                pad_value=0.0,
                equisized=self.equisized,
            )
        )
        self._confidences_padded = (
            None
            if self._confidences_list is None
            else structutils.list_to_padded(
                self._confidences_list,
                (self._N, self.num_confidences),
                pad_value=0.0,
                equisized=self.equisized,
            )
        )

    def clone(self):
        r"""Returns deep copy of Pointclouds object. All internal tensors are cloned individually.

        Returns:
            gradslam.Pointclouds: cloned gradslam.Pointclouds object
        """
        if not self.has_points:
            return Pointclouds(device=self.device)
        elif self._points_list is not None:
            new_points = [p.clone() for p in self.points_list]
            new_normals = (
                None
                if self._normals_list is None
                else [n.clone() for n in self._normals_list]
            )
            new_colors = (
                None
                if self._colors_list is None
                else [c.clone() for c in self._colors_list]
            )
            new_features = (
                None
                if self._features_list is None
                else [f.clone() for f in self._features_list]
            )
            # KM
            new_embeddings = (
                None
                if self._embeddings_list is None
                else [e.clone() for e in self._embeddings_list]
            )
            new_confidences = (
                None
                if self._confidences_list is None
                else [c.clone() for c in self._confidences_list]
            )
        elif self._points_padded is not None:
            new_points = self._points_padded.clone()
            new_normals = (
                None if self._normals_padded is None else self._normals_padded.clone()
            )
            new_colors = (
                None if self._colors_padded is None else self._colors_padded.clone()
            )
            new_features = (
                None if self._features_padded is None else self._features_padded.clone()
            )
            # KM
            new_embeddings = (
                None
                if self._embeddings_padded is None
                else self._embeddings_padded.clone()
            )
            new_confidences = (
                None
                if self._confidences_padded is None
                else self._confidences_padded.clone()
            )

        other = Pointclouds(
            points=new_points,
            normals=new_normals,
            colors=new_colors,
            features=new_features,
            embeddings=new_embeddings,  # KM
            confidences=new_confidences,
        )
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
        return other

    def detach(self):
        r"""Detachs Pointclouds object. All internal tensors are detached individually.

        Returns:
            gradslam.Pointclouds: detached gradslam.Pointclouds object
        """
        other = self.clone()
        if other._points_list is not None:
            other._points_list = [p.detach() for p in other._points_list]
        if other._normals_list is not None:
            other._normals_list = [n.detach() for n in other._normals_list]
        if other._colors_list is not None:
            other._colors_list = [c.detach() for c in other._colors_list]
        if other._features_list is not None:
            other._features_list = [f.detach() for f in other._features_list]
        if other._embeddings_list is not None:  # KM
            other._embeddings_list = [e.detach() for e in other._embeddings_list]  # KM
        if other._confidences_list is not None:
            other._confidences_list = [c.detach() for c in other._confidences_list]
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.detach())
        return other

    def to(self, device: Union[torch.device, str], copy: bool = False):
        r"""Match functionality of torch.Tensor.to(device)
        If copy = True or the self Tensor is on a different device, the returned tensor is a copy of self with the
        desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device, then self is returned.

        Args:
            device (torch.device or str): Device id for the new tensor.
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
            if other._embeddings_list is not None:  # KM
                other._embeddings_list = [
                    e.to(device) for e in other._embeddings_list
                ]  # KM
            if other._confidences_list is not None:
                other._confidences_list = [
                    c.to(device) for c in other._confidences_list
                ]
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

    def append_points(self, pointclouds: "Pointclouds"):
        r"""Appends points, normals, colors and features of a gradslam.Pointclouds object to the current pointclouds.
        Both Pointclouds must have/not have the same attributes. In place operation.

        Args:
            pointclouds (gradslam.Pointclouds): Pointclouds to get appended to self. Must have same batch size as self.

        Returns:
            self
        """
        if not isinstance(pointclouds, type(self)):
            raise TypeError(
                "Append object must be of type gradslam.Pointclouds, but was of type {}.".format(
                    type(pointclouds)
                )
            )
        if not (pointclouds.device == self.device):
            raise ValueError(
                "Device of pointclouds to append and to be appended must match: ({0} != {1})".format(
                    pointclouds.device, self.device
                )
            )

        if not pointclouds.has_points:
            return self

        if not self.has_points:
            if pointclouds.has_points:
                self._points_list = [
                    p.clone().to(self.device) for p in pointclouds.points_list
                ]
                if pointclouds.has_normals:
                    self._normals_list = [
                        n.clone().to(self.device) for n in pointclouds.normals_list
                    ]
                if pointclouds.has_colors:
                    self._colors_list = [
                        c.clone().to(self.device) for c in pointclouds.colors_list
                    ]
                if pointclouds.has_features:
                    self._features_list = [
                        f.clone().to(self.device) for f in pointclouds.features_list
                    ]
                if pointclouds.has_embeddings:  # KM
                    self._embeddings_list = [
                        e.clone().to(self.device)
                        for e in pointclouds._embeddings_list  # KM
                    ]
                if pointclouds.has_confidences:
                    self._confidences_list = [
                        c.clone().to(self.device) for c in pointclouds.confidences_list
                    ]
                self._has_points = pointclouds._has_points
                self._has_normals = pointclouds._has_normals
                self._has_colors = pointclouds._has_colors
                self._has_features = pointclouds._has_features
                self._has_embeddings = pointclouds._has_embeddings  # KM
                self._has_confidences = pointclouds._has_confidences
                self._B = pointclouds._B
                self._N = pointclouds._N
                self.equisized = pointclouds.equisized
                for k in self._INTERNAL_TENSORS:
                    v = getattr(pointclouds, k)
                    if torch.is_tensor(v):
                        setattr(self, k, v.clone())
            return self

        if not (len(pointclouds) == len(self)):
            raise ValueError(
                "Batch size of pointclouds to append and to be appended must match: ({0} != {1})".format(
                    len(pointclouds), len(self)
                )
            )
        if self.has_normals != pointclouds.has_normals:
            raise ValueError(
                "pointclouds to append and to be appended must either both have or not have normals: ({0} != {1})".format(
                    pointclouds.has_normals, self.has_normals
                )
            )
        if self.has_colors != pointclouds.has_colors:
            raise ValueError(
                "pointclouds to append and to be appended must either both have or not have colors: ({0} != {1})".format(
                    pointclouds.has_colors, self.has_colors
                )
            )
        if self.has_features != pointclouds.has_features:
            raise ValueError(
                "pointclouds to append and to be appended must either both have or not have features: ({0} != {1})".format(
                    pointclouds.has_features, self.has_features
                )
            )
        if self.has_embeddings != pointclouds.has_embeddings:  # KM
            raise ValueError(
                "pointclouds to append and to be appended must have the same number of embeddings: ({0} != {1})".format(
                    pointclouds.num_embeddings, self.num_embeddings
                )
            )  # KM
        if self.has_confidences != pointclouds.has_confidences:
            raise ValueError(
                "pointclouds to append and to be appended must either both have or not have confidences: ({0} != {1})".format(
                    pointclouds.has_confidences, self.has_confidences
                )
            )
        self._points_list = [
            torch.cat([self.points_list[b], pointclouds.points_list[b]], 0)
            for b in range(self._B)
        ]
        self._points_padded = None

        if self.has_normals:
            self._normals_list = [
                torch.cat([self.normals_list[b], pointclouds.normals_list[b]], 0)
                for b in range(self._B)
            ]
            self._normals_padded = None

        if self.has_colors:
            self._colors_list = [
                torch.cat([self.colors_list[b], pointclouds.colors_list[b]], 0)
                for b in range(self._B)
            ]
            self._colors_padded = None

        if self.has_features:
            self._features_list = [
                torch.cat([self.features_list[b], pointclouds.features_list[b]], 0)
                for b in range(self._B)
            ]
            self._features_padded = None

        # KM
        if self.has_embeddings:
            self._embeddings_list = [
                torch.cat([self.embeddings_list[b], pointclouds.embeddings_list[b]], 0)
                for b in range(self._B)
            ]
            self._embeddings_padded = None

        if self.has_confidences:
            self._confidences_list = [
                torch.cat(
                    [self.confidences_list[b], pointclouds.confidences_list[b]], 0
                )
                for b in range(self._B)
            ]
            self._confidences_padded = None

        self._num_points_per_pointcloud = (
            self._num_points_per_pointcloud + pointclouds._num_points_per_pointcloud
        )
        self.equisized = len(self._num_points_per_pointcloud.unique()) == 1
        self._N = self._num_points_per_pointcloud.max()
        self._nonpad_mask = None

        return self

    def open3d(
        self,
        index: int,
        include_colors: bool = True,
        max_num_points: Optional[int] = None,
        include_normals: bool = False,
    ):
        r"""Converts `index`-th pointcloud to a `open3d.geometry.Pointcloud` object (e.g. for visualization).

        Args:
            index (int): Index of which pointcloud (from the batch of pointclouds) to convert to
                `open3d.geometry.Pointcloud`.
            include_colors (bool): If True, will include colors in the `o3d.geometry.Pointcloud`
                objects. Default: True
            max_num_points (int): Maximum number of points to include in the returned object. If None,
                will not set a max size (will not downsample). Default: None
            include_normals (bool): If True, will include normal vectors in the `o3d.geometry.Pointcloud`
                objects. Default: False

        Returns:
            pcd (open3d.geometry.Pointcloud): `open3d.geometry.Pointcloud` object from `index`-th pointcloud.
        """
        import open3d as o3d

        if not isinstance(index, int):
            raise TypeError("Index should be int, but was {}.".format(type(index)))

        pcd = o3d.geometry.PointCloud()

        num_points = self.num_points_per_pointcloud[index]
        torch_points = self.points_list[index]
        subsample = max_num_points is not None and max_num_points < num_points
        if subsample:
            perm = torch.randperm(num_points)
            point_inds = perm[:max_num_points]
            torch_points = torch_points[point_inds]
        numpy_points = torch_points.detach().cpu().numpy()
        pcd.points = o3d.utility.Vector3dVector(numpy_points)

        if self.has_colors and include_colors:
            torch_colors = self.colors_list[index]
            if subsample:
                torch_colors = torch_colors[point_inds]
            # if colors > 1, assume 255 range
            if (torch_colors.max() > 1.1).item():
                torch_colors = torch_colors / 255
            torch_colors = torch.clamp(torch_colors, min=0.0, max=1.0)
            numpy_colors = torch_colors.detach().cpu().numpy()
            pcd.colors = o3d.utility.Vector3dVector(numpy_colors)

        if self.has_normals and include_normals:
            torch_normals = self.normals_list[index]
            if subsample:
                torch_normals = torch_normals[point_inds]
            numpy_normals = torch_normals.detach().cpu().numpy()
            pcd.normals = o3d.utility.Vector3dVector(numpy_normals)

        return pcd

    def plotly(
        self,
        index: int,
        include_colors: bool = True,
        max_num_points: Optional[int] = 200000,
        as_figure: bool = True,
        point_size: int = 2,
    ):
        r"""Converts `index`-th pointcloud to either a `plotly.graph_objects.Figure` or a
        `plotly.graph_objects.Scatter3d` object (for visualization).

        Args:
            index (int): Index of which pointcloud (from the batch of pointclouds) to convert to plotly
                representation.
            include_colors (bool): If True, will include point colors in the returned object. Default: True
            max_num_points (int): Maximum number of points to include in the returned object. If None,
                will not set a max size (will not downsample). Default: 200000
            as_figure (bool): If True, returns a `plotly.graph_objects.Figure` object which can easily
                be visualized by calling `.show()` on. Otherwise, returns a
                `plotly.graph_objects.Scatter3d` object. Default: True
            point_size (int): Point size radius (for visualization). Default: 2

        Returns:
            plotly.graph_objects.Figure or plotly.graph_objects.Scatter3d: If `as_figure` is True, will return
            `plotly.graph_objects.Figure` object from the `index`-th pointcloud. Else,
            returns `plotly.graph_objects.Scatter3d` object from the `index`-th pointcloud.
        """
        import plotly.graph_objects as go

        if not isinstance(index, int):
            raise TypeError("Index should be int, but was {}.".format(type(index)))
        num_points = self.num_points_per_pointcloud[index]
        torch_points = self.points_list[index]
        subsample = max_num_points is not None and max_num_points < num_points
        if subsample:
            perm = torch.randperm(num_points)
            point_inds = perm[:max_num_points]
            torch_points = torch_points[point_inds]
        numpy_points = torch_points.detach().cpu().numpy()

        marker_dict = {"size": point_size}

        if self.has_colors and include_colors:
            torch_colors = self.colors_list[index]
            if subsample:
                torch_colors = torch_colors[point_inds]
            # if colors > 1, assume 255 range
            if (torch_colors.max() < 1.1).item():
                torch_colors = torch_colors * 255
            torch_colors = torch.clamp(torch_colors, min=0.0, max=255.0)
            numpy_colors = torch_colors.detach().cpu().numpy().astype("uint8")
            marker_dict["color"] = numpy_colors

        scatter3d = go.Scatter3d(
            x=numpy_points[..., 0],
            y=numpy_points[..., 1],
            z=numpy_points[..., 2],
            mode="markers",
            marker=marker_dict,
        )

        if not as_figure:
            return scatter3d

        fig = go.Figure(data=[scatter3d])
        fig.update_layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    visible=False,
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    visible=False,
                ),
                zaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    visible=False,
                ),
            ),
        )

        return fig

    def save_to_h5(
        self,
        save_dest: str,
        index: Optional[int] = 0,
        include_points: bool = True,
        include_colors: bool = True,
        include_features: bool = True,
        include_embeddings: bool = True,
        include_confidences: bool = True,
    ):
        r"""Saves `index`-th pointcloud to disk.

        Args:
            save_dest (str): Path to save the pointcloud to.
            index (int): Index of which pointcloud (from the batch of pointclouds) to save to disk. Tbh idk what this is.
            include_points (bool): If True, will include point coordinates in the saved file. Default: True
            include_colors (bool): If True, will include point colors in the saved file. Default: True
            include_normals (bool): If True, will include point normals in the saved file. Default: True
            include_features (bool): If True, will include point features in the saved file. Default: True
            include_embeddings (bool): If True, will include point embeddings in the saved file. Default: True
            include_confidences (bool): If True, will include point confidences in the saved file. Default: True
        """
        import h5py

        if not isinstance(index, int):
            raise TypeError("Index should be int, but was {}.".format(type(index)))
        if not isinstance(save_dest, str):
            raise TypeError(
                "save_dest should be str, but was {}.".format(type(save_dest))
            )

        folder_path = f"{save_dest}/pointclouds"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if include_points and self.has_points:
            pc_points = self.points_list[index].detach().cpu()
            with h5py.File(f"{folder_path}/pc_points.h5", "w") as f:
                f.create_dataset("pc_points", data=pc_points)
        if include_colors and self.has_colors:
            pc_colors = self.colors_list[index].detach().cpu()
            with h5py.File(f"{folder_path}/pc_colors.h5", "w") as f:
                f.create_dataset("pc_colors", data=pc_colors)
        if include_features and self.has_features:
            pc_features = self.features_list[index].detach().cpu()
            with h5py.File(f"{folder_path}/pc_features.h5", "w") as f:
                f.create_dataset("pc_features", data=pc_features)
        if include_embeddings and self.has_embeddings:
            pc_embeddings = self.embeddings_list[index].detach().cpu()
            with h5py.File(f"{folder_path}/pc_embeddings.h5", "w") as f:
                f.create_dataset("pc_embeddings", data=pc_embeddings)
        if include_confidences and self.has_confidences:
            pc_confidences = self.confidences_list[index].detach().cpu()
            with h5py.File(f"{folder_path}/pc_confidences.h5", "w") as f:
                f.create_dataset("pc_confidences", data=pc_confidences)

    def _assert_set_padded(self, value: torch.Tensor, first_2_dims_only: bool = False):
        r"""Checks if value can be set as a padded representation attribute

        Args:
            value (torch.Tensor): value we want to set as one of the padded representation attributes
            first_2_dims_only (bool): If True, will only check if first 2 dimensions of value are the same as
                `self.points_padded`. Otherwise will check the entire shape. Default: False
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError("value must be torch.Tensor. Got {}".format(type(value)))
        if not self.has_points:
            raise ValueError(
                "cannot set padded representation for an empty pointclouds object"
            )
        if self.device != torch.device(value.device):
            raise ValueError(
                "value must have the same device as pointclouds object: {} != {}".format(
                    value.device, torch.device(self.device)
                )
            )
        if value.ndim != 3:
            raise ValueError("value.ndim should be 3. Got {}".format(value.ndim))
        if first_2_dims_only and self.points_padded.shape[:2] != value.shape[:2]:
            raise ValueError(
                "first 2 dims of value tensor and points tensor should have same shape, but didn't: {} != {}.".format(
                    value.shape[:2], self.points_padded.shape[:2]
                )
            )
        if (not first_2_dims_only) and self.points_padded.shape != value.shape:
            raise ValueError(
                "value tensor and points tensor should have same shape, but didn't: {} != {}.".format(
                    value.shape, self.points_padded.shape
                )
            )
        if not all(
            [
                value[b][N_b:].eq(0).all().item()
                for b, N_b in enumerate(self.num_points_per_pointcloud)
            ]
        ):
            raise ValueError(
                "value must have zeros wherever pointclouds.points_padded has zero padding."
            )

    def _assert_set_list(self, value: List[torch.Tensor], first_dim_only: bool = False):
        r"""Checks if value can be set as a list representation attribute

        Args:
            value (list of torch.Tensor): value we want to set as one of the list representation attributes
            first_dim_only (bool): If True, will only check if first dimension of value is the same as
                `self.points_padded`. Otherwise will check the entire shape. Default: False
        """
        if not isinstance(value, list):
            raise TypeError(
                "value must be list of torch.Tensors. Got {}".format(type(value))
            )
        if not self.has_points:
            raise ValueError(
                "cannot set list representation for an empty pointclouds object"
            )
        if len(self) != len(value):
            raise ValueError(
                "value must have same length as pointclouds.points_list. Got {} != {}.".format(
                    len(value), len(self)
                )
            )
        if any([v.ndim != 2 for v in value]):
            raise ValueError("ndim of all tensors in value list should be 2")
        if first_dim_only and any(
            [
                self.points_list[b].shape[:1] != value[b].shape[:1]
                for b in range(len(self))
            ]
        ):
            raise ValueError(
                "shape of first 2 dims of tensors in value and pointclouds.points_list must match"
            )
        if (not first_dim_only) and any(
            [self.points_list[b].shape != value[b].shape for b in range(len(self))]
        ):
            raise ValueError(
                "shape of tensors in value and pointclouds.points_list must match"
            )
