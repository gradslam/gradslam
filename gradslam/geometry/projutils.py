"""
Projective geometry utility functions.
"""

from typing import Optional

import torch


def homogenize_points(pts: torch.Tensor):
    r"""Convert a set of points to homogeneous coordinates.

    Args:
        pts (torch.Tensor): Tensor containing points to be homogenized.

    Returns:
        torch.Tensor: Homogeneous coordinates of pts.

    Shape:
        - pts: :math:`(N, *, K)` where :math:`N` indicates the number of points in a cloud if
          the shape is :math:`(N, K)` and indicates batchsize if the number of dimensions is greater than 2.
          Also, :math:`*` means any number of additional dimensions, and `K` is the dimensionality of each point.
        - Output: :math:`(N, *, K + 1)` where all but the last dimension are the same shape as `pts`.

    Examples::

        >>> pts = torch.rand(10, 3)
        >>> pts_homo = homogenize_points(pts)
        >>> pts_homo.shape
        torch.Size([10, 4])
    """
    if not isinstance(pts, torch.Tensor):
        raise TypeError(
            "Expected input type torch.Tensor. Got {} instead".format(type(pts))
        )
    if pts.dim() < 2:
        raise ValueError(
            "Input tensor must have at least 2 dimensions. Got {} instad.".format(
                pts.dim()
            )
        )

    return torch.nn.functional.pad(pts, (0, 1), "constant", 1.0)


def unhomogenize_points(pts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""Convert a set of points from homogeneous coordinates to Euclidean
    coordinates. This is usually done by taking each point :math:`(X, Y, Z, W)` and dividing it by
    the last coordinate :math:`(w)`.

    Args:
        pts (torch.Tensor): Tensor containing points to be unhomogenized.

    Returns:
        torch.Tensor: 'Unhomogenized' points

    Shape:
        - pts: :math:`(N, *, K)` where :math:`N` indicates the number of points in a cloud if
          the shape is :math:`(N, K)` and indicates batchsize if the number of dimensions is greater than 2.
          Also, :math:`*` means any number of additional dimensions, and `K` is the dimensionality of each point.
        - output: :math:`(N, *, K-1)` where all but the last dimension are the same shape as `pts`.

    Examples::

        >>> pts = torch.rand(10, 3)
        >>> pts_unhomo = unhomogenize_points(pts)
        >>> pts_unhomo.shape
        torch.Size([10, 2])
    """
    if not isinstance(pts, torch.Tensor):
        raise TypeError(
            "Expected input type torch.Tensor. Instead got {}".format(type(pts))
        )
    if pts.dim() < 2:
        raise ValueError(
            "Input tensor must have at least 2 dimensions. Got {} instad.".format(
                pts.dim()
            )
        )

    # Get points with the last coordinate (scale) as 0 (points at infinity).
    w: torch.Tensor = pts[..., -1:]
    # Determine the scale factor each point needs to be multiplied by
    # For points at infinity, use a scale factor of 1 (used by OpenCV
    # and by kornia)
    # https://github.com/opencv/opencv/pull/14411/files
    scale: torch.Tensor = torch.where(torch.abs(w) > eps, 1.0 / w, torch.ones_like(w))

    return scale * pts[..., :-1]


def project_points(
    cam_coords: torch.Tensor, proj_mat: torch.Tensor, eps: Optional[float] = 1e-6
) -> torch.Tensor:
    r"""Projects points from the camera coordinate frame to the image (pixel) frame.

    Args:
        cam_coords (torch.Tensor): pixel coordinates (defined in the
            frame of the first camera).
        proj_mat (torch.Tensor): projection matrix between the reference
            and the non-reference camera frame.

    Returns:
        torch.Tensor: Image (pixel) coordinates corresponding to the input 3D points.

    Shapes:
        - cam_coords: :math:`(N, *, 3)` or :math:`(*, 4)` where :math:`*` indicates an arbitrary number of dimensions.
          Here :math:`N` indicates the number of points in a cloud if the shape is :math:`(N, 3)` and indicates
          batchsize if the number of dimensions is greater than 2.
        - proj_mat: :math:`(*, 4, 4)` where :math:`*` indicates an arbitrary number of dimensions.
          dimension contains a :math:`(4, 4)` camera projection matrix.
        - Output: :math:`(N, *, 2)`, where :math:`*` indicates the same dimensions as in `cam_coords`.
          Here :math:`N` indicates the number of points in a cloud if the shape is :math:`(N, 3)` and indicates
          batchsize if the number of dimensions is greater than 2.

    Examples::

        >>> # Case 1: Input cam_coords are homogeneous, no batchsize dimension.
        >>> cam_coords = torch.rand(10, 4)
        >>> proj_mat = torch.rand(4, 4)
        >>> pixel_coords = project_points(cam_coords, proj_mat)
        >>> pixel_coords.shape
        torch.Size([10, 2])
        >>> # Case 2: Input cam_coords are homogeneous and batched. Broadcast proj_mat across batch.
        >>> cam_coords = torch.rand(2, 10, 4)
        >>> proj_mat = torch.rand(4, 4)
        >>> pixel_coords = project_points(cam_coords, proj_mat)
        >>> pixel_coords.shape
        torch.Size([2, 10, 2])
        >>> # Case 3: Input cam_coords are homogeneous and batched. A different proj_mat applied to each element.
        >>> cam_coords = torch.rand(2, 10, 4)
        >>> proj_mat = torch.rand(2, 4, 4)
        >>> pixel_coords = project_points(cam_coords, proj_mat)
        >>> pixel_coords.shape
        torch.Size([2, 10, 2])
        >>> # Case 4: Similar to case 1, but cam_coords are unhomogeneous.
        >>> cam_coords = torch.rand(10, 3)
        >>> proj_mat = torch.rand(4, 4)
        >>> pixel_coords = project_points(cam_coords, proj_mat)
        >>> pixel_coords.shape
        torch.Size([10, 2])
        >>> # Case 5: Similar to case 2, but cam_coords are unhomogeneous.
        >>> cam_coords = torch.rand(2, 10, 3)
        >>> proj_mat = torch.rand(4, 4)
        >>> pixel_coords = project_points(cam_coords, proj_mat)
        >>> pixel_coords.shape
        torch.Size([2, 10, 2])
        >>> # Case 6: Similar to case 3, but cam_coords are unhomogeneous.
        >>> cam_coords = torch.rand(2, 10, 3)
        >>> proj_mat = torch.rand(2, 4, 4)
        >>> pixel_coords = project_points(cam_coords, proj_mat)
        >>> pixel_coords.shape
        torch.Size([2, 10, 2])
    """
    # Based on
    # https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py#L43
    # and Kornia.
    if not torch.is_tensor(cam_coords):
        raise TypeError(
            "Expected input cam_coords to be of type torch.Tensor. Got {0} instead.".format(
                type(cam_coords)
            )
        )
    if not torch.is_tensor(proj_mat):
        raise TypeError(
            "Expected input proj_mat to be of type torch.Tensor. Got {0} instead.".format(
                type(proj_mat)
            )
        )
    if cam_coords.dim() < 2:
        raise ValueError(
            "Input cam_coords must have at least 2 dims. Got {0} instead.".format(
                cam_coords.dim()
            )
        )
    if cam_coords.shape[-1] not in (3, 4):
        raise ValueError(
            "Input cam_coords must have shape (*, 3), or (*, 4). Got {0} instead.".format(
                cam_coords.shape
            )
        )
    if proj_mat.dim() < 2:
        raise ValueError(
            "Input proj_mat must have at least 2 dims. Got {0} instead.".format(
                proj_mat.dim()
            )
        )
    if proj_mat.shape[-1] != 4 or proj_mat.shape[-2] != 4:
        raise ValueError(
            "Input proj_mat must have shape (*, 4, 4). Got {0} instead.".format(
                proj_mat.shape
            )
        )
    if proj_mat.dim() > 2 and proj_mat.dim() != cam_coords.dim():
        raise ValueError(
            "Input proj_mat must either have 2 dimensions, or have equal number of dimensions to cam_coords. "
            "Got {0} instead.".format(proj_mat.dim())
        )
    if proj_mat.dim() > 2 and proj_mat.shape[0] != cam_coords.shape[0]:
        raise ValueError(
            "Batch sizes of proj_mat and cam_coords do not match. Shapes: {0} and {1} respectively.".format(
                proj_mat.shape, cam_coords.shape
            )
        )

    # Determine whether or not to homogenize `cam_coords`.
    to_homogenize: bool = cam_coords.shape[-1] == 3

    pts_homo = None
    if to_homogenize:
        pts_homo: torch.Tensor = homogenize_points(cam_coords)
    else:
        pts_homo: torch.Tensor = cam_coords

    # Determine whether `proj_mat` needs to be expanded to match dims of `cam_coords`.
    to_expand_proj_mat: bool = (proj_mat.dim() == 2) and (pts_homo.dim() > 2)
    if to_expand_proj_mat:
        while proj_mat.dim() < pts_homo.dim():
            proj_mat = proj_mat.unsqueeze(0)

    # Whether to perform simple matrix multiplaction instead of batch matrix multiplication.
    need_bmm: bool = pts_homo.dim() > 2

    if not need_bmm:
        pts: torch.Tensor = torch.matmul(proj_mat.unsqueeze(0), pts_homo.unsqueeze(-1))
    else:
        pts: torch.Tensor = torch.matmul(proj_mat.unsqueeze(-3), pts_homo.unsqueeze(-1))

    # Remove the extra dimension resulting from torch.matmul()
    pts = pts.squeeze(-1)
    # Unhomogenize and stack.
    x: torch.Tensor = pts[..., 0]
    y: torch.Tensor = pts[..., 1]
    z: torch.Tensor = pts[..., 2]
    u: torch.Tensor = x / torch.where(z != 0, z, torch.ones_like(z))
    v: torch.Tensor = y / torch.where(z != 0, z, torch.ones_like(z))

    return torch.stack((u, v), dim=-1)


def unproject_points(
    pixel_coords: torch.Tensor, intrinsics_inv: torch.Tensor, depths: torch.Tensor
) -> torch.Tensor:
    r"""Unprojects points from the image (pixel) frame to the camera coordinate frame.

    Args:
        pixel_coords (torch.Tensor): pixel coordinates.
        intrinsics_inv (torch.Tensor): inverse of the camera intrinsics matrix.
        depths (torch.Tensor): per-pixel depth estimates.

    Returns:
        torch.Tensor: camera coordinates

    Shapes:
        - pixel_coords: :math:`(N, *, 2)` or :math:`(*, 3)`, where * indicates an arbitrary number of dimensions.
          Here :math:`N` indicates the number of points in a cloud if the shape is :math:`(N, 3)` and indicates
          batchsize if the number of dimensions is greater than 2.
        - intrinsics_inv: :math:`(*, 3, 3)`, where * indicates an arbitrary number of dimensions.
        - depths: :math:`(N, *)` where * indicates the same number of dimensions as in `pixel_coords`.
          Here :math:`N` indicates the number of points in a cloud if the shape is :math:`(N, 3)` and indicates
          batchsize if the number of dimensions is greater than 2.
        - output: :math:`(N, *, 3)` where * indicates the same number of dimensions as in `pixel_coords`.
          Here :math:`N` indicates the number of points in a cloud if the shape is :math:`(N, 3)` and indicates
          batchsize if the number of dimensions is greater than 2.

    Examples::

        >>> # Case 1: Input pixel_coords are homogeneous, no batchsize dimension.
        >>> pixel_coords = torch.rand(10, 3)
        >>> intrinsics_inv = torch.rand(3, 3)
        >>> depths = torch.rand(10)
        >>> cam_coords = unproject_points(pixel_coords, intrinsics_inv, depths)
        >>> cam_coords.shape
        torch.Size([10, 3])
        >>> # Case 2: Input pixel_coords are homogeneous, with a batchsize dimension. But, intrinsics_inv is not.
        >>> pixel_coords = torch.rand(2, 10, 3)
        >>> intrinsics_inv = torch.rand(3, 3)
        >>> depths = torch.rand(2, 10)
        >>> cam_coords = unproject_points(pixel_coords, intrinsics_inv, depths)
        >>> cam_coords.shape
        torch.Size([2, 10, 3])
        >>> # Case 3: Input pixel_coords are homogeneous, with a batchsize dimension. intrinsics_inv is batched.
        >>> pixel_coords = torch.rand(2, 10, 3)
        >>> intrinsics_inv = torch.rand(2, 3, 3)
        >>> depths = torch.rand(2, 10)
        >>> cam_coords = unproject_points(pixel_coords, intrinsics_inv, depths)
        >>> cam_coords.shape
        torch.Size([2, 10, 3])
        >>> # Case 4: Similar to case 1, but input pixel_coords are unhomogeneous.
        >>> pixel_coords = torch.rand(10, 2)
        >>> intrinsics_inv = torch.rand(3, 3)
        >>> depths = torch.rand(10)
        >>> cam_coords = unproject_points(pixel_coords, intrinsics_inv, depths)
        >>> cam_coords.shape
        torch.Size([10, 3])
        >>> # Case 5: Similar to case 2, but input pixel_coords are unhomogeneous.
        >>> pixel_coords = torch.rand(2, 10, 2)
        >>> intrinsics_inv = torch.rand(3, 3)
        >>> depths = torch.rand(2, 10)
        >>> cam_coords = unproject_points(pixel_coords, intrinsics_inv, depths)
        >>> cam_coords.shape
        torch.Size([2, 10, 3])
        >>> # Case 6: Similar to case 3, but input pixel_coords are unhomogeneous.
        >>> pixel_coords = torch.rand(2, 10, 2)
        >>> intrinsics_inv = torch.rand(2, 3, 3)
        >>> depths = torch.rand(2, 10)
        >>> cam_coords = unproject_points(pixel_coords, intrinsics_inv, depths)
        >>> cam_coords.shape
        torch.Size([2, 10, 3])
    """
    if not torch.is_tensor(pixel_coords):
        raise TypeError(
            "Expected input pixel_coords to be of type torch.Tensor. Got {0} instead.".format(
                type(pixel_coords)
            )
        )
    if not torch.is_tensor(intrinsics_inv):
        raise TypeError(
            "Expected intrinsics_inv to be of type torch.Tensor. Got {0} instead.".format(
                type(intrinsics_inv)
            )
        )
    if not torch.is_tensor(depths):
        raise TypeError(
            "Expected depth to be of type torch.Tensor. Got {0} instead.".format(
                type(depths)
            )
        )
    if pixel_coords.dim() < 2:
        raise ValueError(
            "Input pixel_coords must have at least 2 dims. Got {0} instead.".format(
                pixel_coords.dim()
            )
        )
    if pixel_coords.shape[-1] not in (2, 3):
        raise ValueError(
            "Input pixel_coords must have shape (*, 2), or (*, 2). Got {0} instead.".format(
                pixel_coords.shape
            )
        )
    if intrinsics_inv.dim() < 2:
        raise ValueError(
            "Input intrinsics_inv must have at least 2 dims. Got {0} instead.".format(
                intrinsics_inv.dim()
            )
        )
    if intrinsics_inv.shape[-1] != 3 or intrinsics_inv.shape[-2] != 3:
        raise ValueError(
            "Input intrinsics_inv must have shape (*, 3, 3). Got {0} instead.".format(
                intrinsics_inv.shape
            )
        )
    if intrinsics_inv.dim() > 2 and intrinsics_inv.dim() != pixel_coords.dim():
        raise ValueError(
            "Input intrinsics_inv must either have 2 dimensions, or have equal number of dimensions to pixel_coords. "
            "Got {0} instead.".format(intrinsics_inv.dim())
        )
    if intrinsics_inv.dim() > 2 and intrinsics_inv.shape[0] != pixel_coords.shape[0]:
        raise ValueError(
            "Batch sizes of intrinsics_inv and pixel_coords do not match. Shapes: {0} and {1} respectively.".format(
                intrinsics_inv.shape, pixel_coords.shape
            )
        )
    if pixel_coords.shape[:-1] != depths.shape:
        raise ValueError(
            "Input pixel_coords and depths must have the same shape for all dimensions except the last. "
            " Got {0} and {1} respectively.".format(pixel_coords.shape, depths.shape)
        )

    # Determine whether or not to homogenize `pixel_coords`.
    to_homogenize: bool = pixel_coords.shape[-1] == 2

    pts_homo = None
    if to_homogenize:
        pts_homo: torch.Tensor = homogenize_points(pixel_coords)
    else:
        pts_homo: torch.Tensor = pixel_coords

    # Determine whether `intrinsics_inv` needs to be expanded to match dims of `pixel_coords`.
    to_expand_intrinsics_inv: bool = (intrinsics_inv.dim() == 2) and (
        pts_homo.dim() > 2
    )
    if to_expand_intrinsics_inv:
        while intrinsics_inv.dim() < pts_homo.dim():
            intrinsics_inv = intrinsics_inv.unsqueeze(0)

    # Whether to perform simple matrix multiplaction instead of batch matrix multiplication.
    need_bmm: bool = pts_homo.dim() > 2

    if not need_bmm:
        pts: torch.Tensor = torch.matmul(
            intrinsics_inv.unsqueeze(0), pts_homo.unsqueeze(-1)
        )
    else:
        pts: torch.Tensor = torch.matmul(
            intrinsics_inv.unsqueeze(-3), pts_homo.unsqueeze(-1)
        )

    # Remove the extra dimension resulting from torch.matmul()
    pts = pts.squeeze(-1)

    return pts * depths.unsqueeze(-1)


def inverse_intrinsics(K: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    r"""Efficient inversion of intrinsics matrix

    Args:
        K (torch.Tensor): Intrinsics matrix
        eps (float): Epsilon for numerical stability

    Returns:
        torch.Tensor: Inverse of intrinsics matrices

    Shape:
        - K: :math:`(*, 4, 4)` or :math:`(*, 3, 3)`
        - Kinv: Matches shape of `K` (:math:`(*, 4, 4)` or :math:`(*, 3, 3)`)
    """
    if not torch.is_tensor(K):
        raise TypeError(
            "Expected K to be of type torch.Tensor. Got {0} instead.".format(type(K))
        )
    if K.dim() < 2:
        raise ValueError(
            "Input K must have at least 2 dims. Got {0} instead.".format(K.dim())
        )
    if not (
        (K.shape[-1] == 3 and K.shape[-2] == 3)
        or (K.shape[-1] == 4 and K.shape[-2] == 4)
    ):
        raise ValueError(
            "Input K must have shape (*, 4, 4) or (*, 3, 3). Got {0} instead.".format(
                K.shape
            )
        )

    Kinv = torch.zeros_like(K)

    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]

    Kinv[..., 0, 0] = 1.0 / (fx + eps)
    Kinv[..., 1, 1] = 1.0 / (fy + eps)
    Kinv[..., 0, 2] = -1.0 * cx / (fx + eps)
    Kinv[..., 1, 2] = -1.0 * cy / (fy + eps)
    Kinv[..., 2, 2] = 1
    Kinv[..., -1, -1] = 1
    return Kinv
