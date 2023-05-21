import warnings
from typing import Union

import torch
from kornia.geometry.linalg import inverse_transformation

from ..geometry.geometryutils import create_meshgrid
from ..structures.pointclouds import Pointclouds
from ..structures.rgbdimages import RGBDImages
from ..structures.utils import pointclouds_from_rgbdimages

__all__ = ["update_map_fusion", "update_map_aggregate"]


def get_alpha(
    points: torch.Tensor,
    sigma: Union[torch.Tensor, float, int],
    dim: int = -1,
    keepdim: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    r"""Computes sample confidence alpha.
    (See section 4.1 of Point-based Fusion paper: http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf )

    Args:
        points (torch.Tensor): Tensor of points.
        sigma (torch.Tensor or float or int): Standard deviation of the Gaussian. Original paper uses 0.6 emperically.
        dim (int): Dimension along which :math:`(X, Y, Z)` of points is stored. Default: -1
        keepdim (bool): Whether the output tensor has `dim` retained or not. Default: False
        eps (float): Minimum value for alpha (to avoid numerical instability). Default: 1e-7

    Returns:
        alpha (torch.Tensor): Sample confidence.

    Shape:
        - points: :math:`(*, 3, *)`
        - sigma: Scalar
        - alpha: Same shape as input points without the `dim`-th dimension.
    """
    if not torch.is_tensor(points):
        raise TypeError(
            "Expected input points to be of type torch.Tensor. Got {0} instead.".format(
                type(points)
            )
        )
    if not (
        torch.is_tensor(sigma) or isinstance(sigma, float) or isinstance(sigma, int)
    ):
        raise TypeError(
            "Expected input sigma to be of type torch.Tensor or float or int. Got {0} instead.".format(
                type(sigma)
            )
        )
    if not isinstance(eps, float):
        raise TypeError(
            "Expected input eps to be of type float. Got {0} instead.".format(type(eps))
        )
    if points.shape[dim] != 3:
        raise ValueError(
            "Expected length of dim-th ({0}th) dimension to be 3. Got {1} instead.".format(
                dim, points.shape[dim]
            )
        )
    if torch.is_tensor(sigma) and sigma.ndim != 0:
        raise ValueError(
            "Expected sigma.ndim to be 0 (scalar). Got {0}.".format(sigma.ndim)
        )
    alpha = torch.exp(
        -torch.sum(points ** 2, dim, keepdim=keepdim) / (2 * (sigma ** 2))
    )
    alpha = torch.clamp(alpha, min=eps, max=1.01)  # make sure alpha is non-zero
    return alpha


def are_points_close(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    dist_th: Union[float, int],
    dim: int = -1,
) -> torch.Tensor:
    r"""Returns bool tensor indicating the euclidean distance between two tensors of vertices along given dimension
    `dim` is smaller than the given threshold value `dist_th`.

    Args:
        tensor1 (torch.Tensor): Input to compute distance on. `dim` must be of length 3 :math:`(X, Y, Z)`.
        tensor2 (torch.Tensor): Input to compute distance on. `dim` must be of length 3 :math:`(X, Y, Z)`.
        dist_th (float or int): Distance threshold.
        dim (int): The dimension to compute distance along. Default: -1

    Returns:
        Output (torch.Tensor): Tensor of bool

    Shape:
        - tensor1: :math:`(*, 3, *)`
        - tensor2: :math:`(*, 3, *)`
        - dist_th: Scalar
        - Output: Similar dimensions to `tensor1` except `dim` is squeezed and output tensor has 1 fewer dimension.
    """
    if not torch.is_tensor(tensor1):
        raise TypeError(
            "Expected input tensor1 to be of type torch.Tensor. Got {0} instead.".format(
                type(tensor1)
            )
        )
    if not torch.is_tensor(tensor2):
        raise TypeError(
            "Expected input tensor2 to be of type torch.Tensor. Got {0} instead.".format(
                type(tensor2)
            )
        )
    if not (isinstance(dist_th, float) or isinstance(dist_th, int)):
        raise TypeError(
            "Expected input dist_th to be of type float or int. Got {0} instead.".format(
                type(dist_th)
            )
        )
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            "tensor1 and tensor2 should have the same shape, but had shapes {0} and {1} respectively.".format(
                tensor1.shape, tensor2.shape
            )
        )
    if tensor1.shape[dim] != 3:
        raise ValueError(
            "Expected length of input tensors' dim-th ({0}th) dimension to be 3. Got {1} instead.".format(
                dim, tensor1.shape[dim]
            )
        )
    return (tensor1 - tensor2).norm(dim=dim) < dist_th


def are_normals_similar(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    dot_th: Union[float, int],
    dim: int = -1,
) -> torch.Tensor:
    r"""Returns bool tensor indicating dot product of two tensors containing normals along given dimension `dim` is
    greater than the given threshold value `dot_th`.

    Args:
        tensor1 (torch.Tensor): Input to compute dot product on. `dim` must be of length 3 :math:`(N_x, N_y, N_z)`.
        tensor2 (torch.Tensor): Input to compute dot product on. `dim` must be of length 3 :math:`(N_x, N_y, N_z)`.
        dot_th (float or int): Dot product threshold.
        dim (int): The dimension to compute product along. Default: -1

    Returns:
        Output (torch.Tensor): Tensor of bool

    Shape:
        - tensor1: :math:`(*, 3, *)`
        - tensor2: :math:`(*, 3, *)`
        - dot_th: Scalar
        - Output: Similar dimensions to `tensor1` except `dim` is squeezed and output tensor has 1 fewer dimension.
    """
    if not torch.is_tensor(tensor1):
        raise TypeError(
            "Expected input tensor1 to be of type torch.Tensor. Got {0} instead.".format(
                type(tensor1)
            )
        )
    if not torch.is_tensor(tensor2):
        raise TypeError(
            "Expected input tensor2 to be of type torch.Tensor. Got {0} instead.".format(
                type(tensor2)
            )
        )
    if not (isinstance(dot_th, float) or isinstance(dot_th, int)):
        raise TypeError(
            "Expected input dot_th to be of type float or int. Got {0} instead.".format(
                type(dot_th)
            )
        )
    if tensor1.shape != tensor2.shape:
        raise ValueError(
            "tensor1 and tensor2 should have the same shape, but had shapes {0} and {1} respectively.".format(
                tensor1.shape, tensor2.shape
            )
        )
    if tensor1.shape[dim] != 3:
        raise ValueError(
            "Expected length of input tensors' dim-th ({0}th) dimension to be 3. Got {1} instead.".format(
                dim, tensor1.shape[dim]
            )
        )
    dot_res = (tensor1 * tensor2).sum(dim)
    if dot_res.max() > 1.001:
        warnings.warn(
            "Max of dot product was {0} > 1. Inputs were not normalized along dim ({1}). Was this intentional?".format(
                dot_res.max(), dim
            ),
            RuntimeWarning,
        )
    return dot_res > dot_th


def find_active_map_points(
    pointclouds: Pointclouds,
    rgbdimages: RGBDImages,
) -> torch.Tensor:
    r"""Returns lookup table for indices of active global map points and their position inside the live frames.
    (See section 4.1 of Point-based Fusion paper: http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf )

    Args:
        pointclouds (gradslam.Pointclouds): Batch of `B` global maps
        rgbdimages (gradslam.RGBDImages): Batch of `B` live frames from the latest sequence. `poses`, `intrinsics`,
            heights and widths of frames are used.

    Returns:
        pc2im_bnhw (torch.Tensor): Active map points lookup table. Each row contains batch index `b`, point index (in
            pointclouds) `n`, and height and width index after projection to live frame `h` and `w` respectively.

    Shape:
        - pc2im_bnhw: :math:`(\text{num_active_map_points}, 4)`

    """
    if not isinstance(pointclouds, Pointclouds):
        raise TypeError(
            "Expected pointclouds to be of type gradslam.Pointclouds. Got {0}.".format(
                type(pointclouds)
            )
        )
    if not isinstance(rgbdimages, RGBDImages):
        raise TypeError(
            "Expected rgbdimages to be of type gradslam.RGBDImages. Got {0}.".format(
                type(rgbdimages)
            )
        )
    if rgbdimages.shape[1] != 1:
        raise ValueError(
            "Expected rgbdimages to have sequence length of 1. Got {0}.".format(
                rgbdimages.shape[1]
            )
        )
    device = pointclouds.device

    if not pointclouds.has_points:
        return torch.empty((0, 4), dtype=torch.int64, device=device)

    if len(rgbdimages) != len(pointclouds):
        raise ValueError(
            "Expected equal batch sizes for pointclouds and rgbdimages. Got {0} and {1} respectively.".format(
                len(pointclouds), len(rgbdimages)
            )
        )

    batch_size, seq_len, height, width = rgbdimages.shape

    tinv = inverse_transformation(rgbdimages.poses.squeeze(1))
    pointclouds_transformed = pointclouds.transform(tinv)
    is_front_of_plane = (
        pointclouds_transformed.points_padded[..., -1] > 0
    )  # don't consider missing depth values
    pointclouds_transformed.pinhole_projection_(
        rgbdimages.intrinsics.squeeze(1)
    )  # IN-PLACE operation
    img_plane_points = pointclouds_transformed.points_padded[..., :-1]  # width, height

    is_in_frame = (
        (img_plane_points[..., 0] > -1e-3)
        & (img_plane_points[..., 0] < width - 0.999)
        & (img_plane_points[..., 1] > -1e-3)
        & (img_plane_points[..., 1] < height - 0.999)
        & is_front_of_plane
        & pointclouds.nonpad_mask
    )
    in_plane_pos = img_plane_points.round().long()
    in_plane_pos = torch.cat(
        [
            in_plane_pos[..., 1:2].clamp(0, height - 1),
            in_plane_pos[..., 0:1].clamp(0, width - 1),
        ],
        -1,
    )  # height, width
    batch_size, num_points = in_plane_pos.shape[:2]
    batch_point_idx = (
        create_meshgrid(batch_size, num_points, normalized_coords=False)
        .squeeze(0)
        .to(device)
    )
    idx_and_plane_pos = torch.cat([batch_point_idx.long(), in_plane_pos], -1)
    pc2im_bnhw = idx_and_plane_pos[is_in_frame]  # (?, 4)

    if pc2im_bnhw.shape[0] == 0:
        warnings.warn("No active map points were found")

    return pc2im_bnhw


def find_similar_map_points(
    pointclouds: Pointclouds,
    rgbdimages: RGBDImages,
    pc2im_bnhw: torch.Tensor,
    dist_th: Union[float, int],
    dot_th: Union[float, int],
) -> torch.Tensor:
    r"""Returns lookup table for points from global maps that are close and have similar normals to points from live
    frames occupying the same pixel as their projection (onto that live frame).
    (See section 4.1 of Point-based Fusion paper: http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf )

    Args:
        pointclouds (gradslam.Pointclouds): Pointclouds of globalmaps
        rgbdimages (gradslam.RGBDImages): Live frames from the latest sequence
        pc2im_bnhw (torch.Tensor): Active map points lookup table. Each row contains batch index `b`, point index (in
            pointclouds) `n`, and height and width index after projection to live frame `h` and `w` respectively.
        dist_th (float or int): Distance threshold.
        dot_th (float or int): Dot product threshold.

    Returns:
        pc2im_bnhw_similar (torch.Tensor): Lookup table of points from global map that are close and have have normals
            that are similar to the live frame points.
        is_similar_mask (torch.Tensor): bool mask indicating which rows from input `pc2im_bnhw` are retained.

    Shape:
        - pc2im_bnhw: :math:`(\text{num_active_map_points}, 4)`
        - dist_th: Scalar
        - dot_th: Scalar
        - pc2im_bnhw_similar: :math:`(\text{num_similar_map_points}, 4)` where
            :math:`\text{num_similar_map_points}\leq\text{num_active_map_points}`
        - is_similar_mask: :math:`(\text{num_active_map_points})` where
            :math:`\text{num_similar_map_points}\leq\text{num_active_map_points}

    """
    if not isinstance(pointclouds, Pointclouds):
        raise TypeError(
            "Expected pointclouds to be of type gradslam.Pointclouds. Got {0}.".format(
                type(pointclouds)
            )
        )
    if not isinstance(rgbdimages, RGBDImages):
        raise TypeError(
            "Expected rgbdimages to be of type gradslam.RGBDImages. Got {0}.".format(
                type(rgbdimages)
            )
        )
    if not torch.is_tensor(pc2im_bnhw):
        raise TypeError(
            "Expected input pc2im_bnhw to be of type torch.Tensor. Got {0} instead.".format(
                type(pc2im_bnhw)
            )
        )
    if pc2im_bnhw.dtype != torch.int64:
        raise TypeError(
            "Expected input pc2im_bnhw to have dtype of torch.int64 (torch.long), not {0}.".format(
                pc2im_bnhw.dtype
            )
        )
    if rgbdimages.shape[1] != 1:
        raise ValueError(
            "Expected rgbdimages to have sequence length of 1. Got {0}.".format(
                rgbdimages.shape[1]
            )
        )
    if pc2im_bnhw.ndim != 2:
        raise ValueError(
            "Expected pc2im_bnhw.ndim of 2. Got {0}.".format(pc2im_bnhw.ndim)
        )
    if pc2im_bnhw.shape[1] != 4:
        raise ValueError(
            "Expected pc2im_bnhw.shape[1] to be 4. Got {0}.".format(pc2im_bnhw.shape[1])
        )

    device = pointclouds.device

    if not pointclouds.has_points or pc2im_bnhw.shape[0] == 0:
        return torch.empty((0, 4), dtype=torch.int64, device=device), torch.empty(
            0, dtype=torch.bool, device=device
        )

    if len(rgbdimages) != len(pointclouds):
        raise ValueError(
            "Expected equal batch sizes for pointclouds and rgbdimages. Got {0} and {1} respectively.".format(
                len(pointclouds), len(rgbdimages)
            )
        )
    if not pointclouds.has_normals:
        raise ValueError(
            "Pointclouds must have normals for finding similar map points, but did not."
        )

    vertex_maps = rgbdimages.global_vertex_map
    normal_maps = rgbdimages.global_normal_map

    frame_points = torch.zeros_like(pointclouds.points_padded)
    frame_normals = torch.zeros_like(pointclouds.normals_padded)

    # vertex_maps -> frame_points will be a one-many mapping
    frame_points[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] = vertex_maps[
        pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]
    ]
    frame_normals[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] = normal_maps[
        pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]
    ]

    # zero normals will automatically get rejected: rgbdimages missing depth values (and pointclouds paddings)
    is_close = are_points_close(frame_points, pointclouds.points_padded, dist_th)
    is_similar = are_normals_similar(frame_normals, pointclouds.normals_padded, dot_th)

    mask = is_close & is_similar  # shape (B, N)
    is_similar_mask = mask[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]]
    pc2im_bnhw_similar = pc2im_bnhw[is_similar_mask]

    if len(pc2im_bnhw_similar) == 0:
        warnings.warn(
            "No similar map points were found (despite total {0} active points across the batch)".format(
                pc2im_bnhw.shape[0]
            ),
            RuntimeWarning,
        )

    return pc2im_bnhw_similar, is_similar_mask


def find_best_unique_correspondences(
    pointclouds: Pointclouds,
    rgbdimages: RGBDImages,
    pc2im_bnhw: torch.Tensor,
) -> torch.Tensor:
    r"""Amongst global map points which project to the same frame pixel, find the ones which have the highest
    confidence counter (and if confidence counter is equal then find the closest one to viewing ray).
    (See section 4.1 of Point-based Fusion paper: http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf )

    Args:
        pointclouds (gradslam.Pointclouds): Pointclouds of globalmaps
        rgbdimages (gradslam.RGBDImages): Live frames from the latest sequence
        pc2im_bnhw (torch.Tensor): Similar map points lookup table. Each row contains batch index `b`, point index (in
            pointclouds) `n`, and height and width index after projection to live frame `h` and `w` respectively. This
            table can have different points (`b`s and `n`s) projecting to the same live frame pixel (same `h` and `w`)

    Returns:
        pc2im_bnhw_unique (torch.Tensor): Lookup table of one-to-one correspondences between points from the global map
            and live frames' points (pixels).

    Shape:
        - pc2im_bnhw: :math:`(\text{num_similar_map_points}, 4)`
        - pc2im_bnhw_unique: :math:`(\text{num_unique_correspondences}, 4)` where
            :math:`\text{num_unique_correspondences}\leq\text{num_similar_map_points}`

    """
    if not isinstance(pointclouds, Pointclouds):
        raise TypeError(
            "Expected pointclouds to be of type gradslam.Pointclouds. Got {0}.".format(
                type(pointclouds)
            )
        )
    if not torch.is_tensor(pc2im_bnhw):
        raise TypeError(
            "Expected input pc2im_bnhw to be of type torch.Tensor. Got {0} instead.".format(
                type(pc2im_bnhw)
            )
        )
    if pc2im_bnhw.dtype != torch.int64:
        raise TypeError(
            "Expected input pc2im_bnhw to have dtype of torch.int64 (torch.long), not {0}.".format(
                pc2im_bnhw.dtype
            )
        )
    if rgbdimages.shape[1] != 1:
        raise ValueError(
            "Expected rgbdimages to have sequence length of 1. Got {0}.".format(
                rgbdimages.shape[1]
            )
        )
    if pc2im_bnhw.ndim != 2:
        raise ValueError(
            "Expected pc2im_bnhw.ndim of 2. Got {0}.".format(pc2im_bnhw.ndim)
        )
    if pc2im_bnhw.shape[1] != 4:
        raise ValueError(
            "Expected pc2im_bnhw.shape[1] to be 4. Got {0}.".format(pc2im_bnhw.shape[1])
        )

    device = pointclouds.device

    if not pointclouds.has_points or pc2im_bnhw.shape[0] == 0:
        return torch.empty((0, 4), dtype=torch.int64, device=device)

    if len(rgbdimages) != len(pointclouds):
        raise ValueError(
            "Expected equal batch sizes for pointclouds and rgbdimages. Got {0} and {1} respectively.".format(
                len(pointclouds), len(rgbdimages)
            )
        )
    if not pointclouds.has_features:
        raise ValueError(
            "Pointclouds must have features for finding best unique correspondences, but did not."
        )

    # argsort so that duplicate B, H, W indices end next to each other, such that first duplicate has higher ccount
    # (& if ccount equal -> first duplicate has smaller ray dist)
    inv_ccounts = 1 / (
        pointclouds.features_padded[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] + 1e-20
    )  # shape: [P 1]
    # compute ray dist
    frame_points = rgbdimages.global_vertex_map[
        pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]
    ]
    ray_dists = (
        (
            (
                pointclouds.points_padded[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]]
                - frame_points
            )
            ** 2
        )
        .sum(-1)
        .unsqueeze(1)
    )
    # unique criteria: [B, H, W, 1/ccounts, ray_dists, N]
    unique_criteria_bhwcrn = [
        pc2im_bnhw[:, 0:1].float(),
        pc2im_bnhw[:, 2:4].float(),
        inv_ccounts,
        ray_dists,
        pc2im_bnhw[:, 1:2].float(),
    ]
    unique_criteria_bhwcrn = torch.cat(unique_criteria_bhwcrn, -1)

    # used torch.unique to sort rows (rows are unique): works as if we stable sorted rows ascendingly based on every
    # column going from right to left.
    # TODO: Might be a faster way? argsort(1e10 * pc2im_bnhw[:, 0] + 1e8 * pc2im_bnhw[:, 2:] + 1e6*inv_ccounts + ...)
    sorted_criteria = torch.unique(
        unique_criteria_bhwcrn.detach(), dim=0
    )  # pytorch issue #47851
    indices = sorted_criteria[:, -1].long()

    # find indices of the first occurrences of (sorted) duplicate B, H, W indices
    sorted_nonunique_inds = sorted_criteria[:, :3]  # shape: (?, 3)
    first_unique_mask = torch.ones(
        sorted_nonunique_inds.shape[0], dtype=bool, device=device
    )
    first_unique_mask[1:] = (
        sorted_nonunique_inds[1:, :3] - sorted_nonunique_inds[:-1, :3] != 0
    ).any(-1)

    first_unique = sorted_criteria[first_unique_mask]
    pc2im_bnhw_unique = torch.cat(
        [
            first_unique[:, 0:1].long(),
            first_unique[:, -1:].long(),
            first_unique[:, 1:3].long(),
        ],
        -1,
    )

    return pc2im_bnhw_unique


def find_correspondences(
    pointclouds: Pointclouds,
    rgbdimages: RGBDImages,
    dist_th: Union[float, int],
    dot_th: Union[float, int],
) -> torch.Tensor:
    r"""Returns a lookup table for inferring unique correspondences between points from the live frame and the global
    map (See section 4.1 of Point-based Fusion paper: http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf )

    Args:
        pointclouds (gradslam.Pointclouds): Pointclouds of global maps
        rgbdimages (gradslam.RGBDImages): Live frames from the latest sequence
        dist_th (float or int): Distance threshold.
        dot_th (float or int): Dot product threshold.

    Returns:
        pc2im_bnhw (torch.Tensor): Unique correspondence lookup table. Each row contains batch index `b`, point index
            (in pointclouds) `n`, and height and width index after projection to live frame `h` and `w` respectively.

    Shape:
        - pc2im_bnhw: :math:`(\text{num_unique_correspondences}, 4)`

    """
    pc2im_bnhw = find_active_map_points(pointclouds, rgbdimages)
    pc2im_bnhw, _ = find_similar_map_points(
        pointclouds, rgbdimages, pc2im_bnhw, dist_th, dot_th
    )
    pc2im_bnhw = find_best_unique_correspondences(pointclouds, rgbdimages, pc2im_bnhw)
    return pc2im_bnhw


def fuse_with_map(
    pointclouds: Pointclouds,
    rgbdimages: RGBDImages,
    pc2im_bnhw: torch.Tensor,
    sigma: Union[torch.Tensor, float, int],
    inplace: bool = False,
    use_embeddings: bool = False,  # KM
    embedding_fusion_method: str = "slam",  # KM
) -> Pointclouds:
    r"""Fuses points from live frames with global maps by merging corresponding points and appending new points.
    (See section 4.2 of Point-based Fusion paper: http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf )

    Args:
        pointclouds (gradslam.Pointclouds): Pointclouds of global maps. Must have points, colors, normals and features
            (ccounts).
        rgbdimages (gradslam.RGBDImages): Live frames from the latest sequence
        pc2im_bnhw (torch.Tensor): Unique correspondence lookup table. Each row contains batch index `b`, point index
            (in pointclouds) `n`, and height and width index after projection to live frame `h` and `w` respectively.
        sigma (torch.Tensor or float or int): Standard deviation of the Gaussian. Original paper uses 0.6 emperically.
        inplace (bool): Can optionally update the pointclouds in-place. Default: False

    Returns:
        pointclouds (gradslam.Pointclouds): Updated Pointclouds object containing global maps.

    Shape:
        - pc2im_bnhw: :math:`(\text{num_unique_correspondences}, 4)`
        - sigma: Scalar

    """
    if not isinstance(pointclouds, Pointclouds):
        raise TypeError(
            "Expected pointclouds to be of type gradslam.Pointclouds. Got {0}.".format(
                type(pointclouds)
            )
        )
    if not isinstance(rgbdimages, RGBDImages):
        raise TypeError(
            "Expected rgbdimages to be of type gradslam.RGBDImages. Got {0}.".format(
                type(rgbdimages)
            )
        )
    if not torch.is_tensor(pc2im_bnhw):
        raise TypeError(
            "Expected input pc2im_bnhw to be of type torch.Tensor. Got {0} instead.".format(
                type(pc2im_bnhw)
            )
        )
    if pc2im_bnhw.dtype != torch.int64:
        raise TypeError(
            "Expected input pc2im_bnhw to have dtype of torch.int64 (torch.long), not {0}.".format(
                pc2im_bnhw.dtype
            )
        )
    if pc2im_bnhw.ndim != 2:
        raise ValueError(
            "Expected pc2im_bnhw.ndim of 2. Got {0}.".format(pc2im_bnhw.ndim)
        )
    if pc2im_bnhw.shape[1] != 4:
        raise ValueError(
            "Expected pc2im_bnhw.shape[1] to be 4. Got {0}.".format(pc2im_bnhw.shape[1])
        )
    if pointclouds.has_points:
        if not pointclouds.has_normals:
            raise ValueError(
                "Pointclouds must have normals for map fusion, but did not."
            )
        if not pointclouds.has_colors:
            raise ValueError(
                "Pointclouds must have colors for map fusion, but did not."
            )
        if not pointclouds.has_features:
            raise ValueError(
                "Pointclouds must have features (ccounts) for map fusion, but did not."
            )

        # KM
        if use_embeddings:
            if not pointclouds.has_embeddings:
                # If a non-empty pointcloud does not have embeddings, raise an error
                raise ValueError(
                    "Pointclouds must have embeddings for map fusion, but did not."
                )

    # Fuse points (from live frame) with corresponding global map points
    vertex_maps = rgbdimages.global_vertex_map
    normal_maps = rgbdimages.global_normal_map
    rgb_image = rgbdimages.rgb_image
    alpha_image = get_alpha(rgbdimages.vertex_map, dim=4, keepdim=True, sigma=sigma)

    embeddings = rgbdimages.embeddings  # KM
    confidence_image = rgbdimages.confidence_image

    if pointclouds.has_points and pc2im_bnhw.shape[0] != 0:

        frame_points = torch.zeros_like(pointclouds.points_padded)
        frame_normals = torch.zeros_like(pointclouds.normals_padded)
        frame_colors = torch.zeros_like(pointclouds.colors_padded)
        frame_alphas = torch.zeros_like(pointclouds.features_padded)

        frame_points[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] = vertex_maps[
            pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]
        ]
        frame_normals[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] = normal_maps[
            pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]
        ]
        frame_colors[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] = rgb_image[
            pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]
        ]
        frame_alphas[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] = alpha_image[
            pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]
        ]

        map_ccounts = pointclouds.features_padded
        updated_ccounts = map_ccounts + frame_alphas
        # TODO: Add the condition for radius of points before applying averaging
        # TODO: Put the mapping + averaging into a function
        updated_points = (map_ccounts * pointclouds.points_padded) + (
            frame_alphas * frame_points
        )
        updated_normals = (map_ccounts * pointclouds.normals_padded) + (
            frame_alphas * frame_normals
        )
        updated_colors = (map_ccounts * pointclouds.colors_padded) + (
            frame_alphas * frame_colors
        )

        # Merge corresponding points
        inv_updated_ccounts = 1 / torch.where(
            updated_ccounts == 0, torch.ones_like(updated_ccounts), updated_ccounts
        )
        pointclouds.points_padded = updated_points * inv_updated_ccounts
        pointclouds.normals_padded = updated_normals * inv_updated_ccounts
        pointclouds.colors_padded = updated_colors * inv_updated_ccounts
        pointclouds.features_padded = updated_ccounts

        # If with embeddings, fuse them accordingly
        if pointclouds.has_embeddings:  # KM
            frame_embeddings = torch.zeros_like(
                pointclouds.embeddings_padded
            ).half()  # KM
            frame_embeddings[pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]] = embeddings[
                pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]
            ]  # KM
            # If all entries in a frame_embedding row are zero, the embedding is invalid
            # That embedding should be excluded from fusion
            # frame_embeddings: (batchsize, num map pts, embedding_dim)
            valid_embeddings_mask = (
                torch.count_nonzero(frame_embeddings, dim=-1)
                == frame_embeddings.shape[-1]
            ).unsqueeze(-1)

            # If RGBDImages has confidences, use it as the fusion factor
            if confidence_image is not None:  # KM
                frame_confidences = torch.zeros_like(pointclouds.confidences_padded)
                frame_confidences[
                    pc2im_bnhw[:, 0], pc2im_bnhw[:, 1]
                ] = confidence_image[
                    pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]
                ]

                # # When the frame_confidences is a all-one map, the following assertion always holds.
                # # This indicates that the zeros in frame_alpha are purely a results of indexing pc2im_bnhw
                # assert (frame_confidences == 0).sum() == (frame_alphas == 0).sum()

            # If not, use the alpha as the fusion factor
            else:
                frame_confidences = frame_alphas

            if embedding_fusion_method == "slam":
                map_ccounts_embedding = pointclouds.confidences_padded
                updated_embeddings = (
                    map_ccounts_embedding.half() * pointclouds.embeddings_padded
                    + frame_confidences.half()
                    * frame_embeddings
                    * valid_embeddings_mask.half()
                )
                updated_ccounts_embedding = (
                    map_ccounts_embedding
                    + frame_confidences * valid_embeddings_mask.half()
                )

                inv_updated_ccounts = 1 / torch.where(
                    updated_ccounts_embedding == 0,
                    torch.ones_like(updated_ccounts_embedding),
                    updated_ccounts_embedding,
                )
                pointclouds.embeddings_padded = updated_embeddings * inv_updated_ccounts
                pointclouds.confidences_padded = updated_ccounts_embedding
            elif embedding_fusion_method == "bayes":
                updated_embeddings = pointclouds.embeddings_padded
                updated_embeddings[frame_confidences.squeeze(-1) > 0] = (
                    updated_embeddings[frame_confidences.squeeze(-1) > 0]
                    * frame_embeddings[frame_confidences.squeeze(-1) > 0]
                )

                # # This will incur CUDA OOM
                # updated_embeddings = torch.nn.functional.normalize(
                #     updated_embeddings, dim=-1
                # )
                pointclouds.embeddings_padded = updated_embeddings

            else:
                raise ValueError(
                    f'Unknown embedding_fusion_method "{embedding_fusion_method}"'
                )

        # # print(pointclouds.embeddings_padded.shape)
        # print(f"pointclouds.feat: {pointclouds.features_padded.shape}")
        # print(f"pointclouds.embed: {pointclouds.embeddings_padded.shape}")

    # Append points (from live frame) that did not have correspondences (from global map)
    new_mask = torch.ones_like(vertex_maps[..., 0], dtype=bool)
    if pointclouds.has_points and pc2im_bnhw.shape[0] != 0:
        new_mask[pc2im_bnhw[:, 0], 0, pc2im_bnhw[:, 2], pc2im_bnhw[:, 3]] = 0
    new_mask = new_mask * rgbdimages.valid_depth_mask.squeeze(
        -1
    )  # don't add missing depths to map
    B = new_mask.shape[0]  # batch size

    new_points = [vertex_maps[b][new_mask[b]] for b in range(B)]
    new_normals = [normal_maps[b][new_mask[b]] for b in range(B)]
    new_colors = [rgb_image[b][new_mask[b]] for b in range(B)]
    new_features = [alpha_image[b][new_mask[b]] for b in range(B)]

    new_embeddings = None
    if embeddings is not None:
        new_embeddings = [embeddings[b][new_mask[b]] for b in range(B)]  # KM

    if confidence_image is not None:
        new_confidences = [confidence_image[b][new_mask[b]] for b in range(B)]
    else:
        new_confidences = [alpha_image[b][new_mask[b]] for b in range(B)]

    # print("Building new pointcloud...")
    new_pointclouds = Pointclouds(
        points=new_points,
        normals=new_normals,
        colors=new_colors,
        features=new_features,
        embeddings=new_embeddings,  # KM
        confidences=new_confidences,  # KM
    )
    # print(f"new_pointclouds.feat: {new_pointclouds.features_padded.shape}")
    # print(f"new_pointclouds.embed: {new_pointclouds.embeddings_padded.shape}")
    if not inplace:
        pointclouds = pointclouds.clone()
    pointclouds.append_points(new_pointclouds)

    return pointclouds


def update_map_aggregate(
    pointclouds: Pointclouds,
    rgbdimages: RGBDImages,
    inplace: bool = False,
    use_embeddings: bool = False,  # KM
) -> Pointclouds:
    r"""Aggregate points from live frames with global maps by appending the live frame points.

    Args:
        pointclouds (gradslam.Pointclouds): Pointclouds of global maps. Must have points, colors, normals and features
            (ccounts).
        rgbdimages (gradslam.RGBDImages): Live frames from the latest sequence
        inplace (bool): Can optionally update the pointclouds in-place. Default: False

    Returns:
        gradslam.Pointclouds: Updated Pointclouds object containing global maps.

    """
    if not isinstance(pointclouds, Pointclouds):
        raise TypeError(
            "Expected pointclouds to be of type gradslam.Pointclouds. Got {0}.".format(
                type(pointclouds)
            )
        )
    if not isinstance(rgbdimages, RGBDImages):
        raise TypeError(
            "Expected rgbdimages to be of type gradslam.RGBDImages. Got {0}.".format(
                type(rgbdimages)
            )
        )
    new_pointclouds = pointclouds_from_rgbdimages(
        rgbdimages,
        global_coordinates=True,
        use_embeddings=use_embeddings,  # KM
    )
    if not inplace:
        pointclouds = pointclouds.clone()
    pointclouds.append_points(new_pointclouds)
    return pointclouds


def update_map_fusion(
    pointclouds: Pointclouds,
    rgbdimages: RGBDImages,
    dist_th: Union[float, int],
    dot_th: Union[float, int],
    sigma: Union[torch.Tensor, float, int],
    inplace: bool = False,
    use_embeddings: bool = False,  # KM
    embedding_fusion_method: str = "slam",  # KM
) -> Pointclouds:
    r"""Updates pointclouds in-place given the live frame RGB-D images using PointFusion.
    (See Point-based Fusion `paper <http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf>`__).

    Args:
        pointclouds (gradslam.Pointclouds): Pointclouds of global maps. Must have points, colors, normals and features
            (ccounts).
        rgbdimages (gradslam.RGBDImages): Live frames from the latest sequence
        dist_th (float or int): Distance threshold.
        dot_th (float or int): Dot product threshold.
        sigma (torch.Tensor or float or int): Standard deviation of the Gaussian. Original paper uses 0.6 emperically.
        inplace (bool): Can optionally update the pointclouds in-place. Default: False

    Returns:
        gradslam.Pointclouds: Updated Pointclouds object containing global maps.

    """
    batch_size, seq_len, height, width = rgbdimages.shape
    pc2im_bnhw = find_correspondences(pointclouds, rgbdimages, dist_th, dot_th)
    pointclouds = fuse_with_map(
        pointclouds,
        rgbdimages,
        pc2im_bnhw,
        sigma,
        inplace,
        use_embeddings=use_embeddings,
        embedding_fusion_method=embedding_fusion_method,
    )

    return pointclouds
