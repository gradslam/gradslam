from typing import Optional, Union

import torch

from ..geometry.geometryutils import transform_pointcloud
from ..geometry.se3utils import se3_exp
from ..structures.pointclouds import Pointclouds
from ..structures.rgbdimages import RGBDImages

__all__ = [
    "solve_linear_system",
    "gauss_newton_solve",
    "point_to_plane_ICP",
    "point_to_plane_gradICP",
    "downsample_pointclouds",
    "downsample_rgbdimages",
]


def solve_linear_system(
    A: torch.Tensor, b: torch.Tensor, damp: Union[float, torch.Tensor] = 1e-8
):
    r"""Solves the normal equations of a linear system Ax = b, given the constraint matrix A and the coefficient vector
    b. Note that this solves the normal equations, not the linear system. That is, solves :math:`A^T A x = A^T b`,
    not :math:`Ax = b`.

    Args:
        A (torch.Tensor): The constraint matrix of the linear system.
        b (torch.Tensor): The coefficient vector of the linear system.
        damp (float or torch.Tensor): Damping coefficient to optionally condition the linear system (in practice,
            a damping coefficient of :math:`\rho` means that we are solving a modified linear system that adds a tiny
            :math:`\rho` to each diagonal element of the constraint matrix :math:`A`, so that the linear system
            becomes :math:`(A^TA + \rho I)x = b`, where :math:`I` is the identity matrix of shape
            :math:`(\text{num_of_variables}, \text{num_of_variables})`. Default: 1e-8

    Returns:
        torch.Tensor: Solution vector of the normal equations of the linear system

    Shape:
        - A: :math:`(\text{num_of_equations}, \text{num_of_variables})`
        - b: :math:`(\text{num_of_equations}, 1)`
        - Output: :math:`(\text{num_of_variables}, 1)`
    """
    if not torch.is_tensor(A):
        raise TypeError(
            "Expected A to be of type torch.Tensor. Got {0}.".format(type(A))
        )
    if not torch.is_tensor(b):
        raise TypeError(
            "Expected b to be of type torch.Tensor. Got {0}.".format(type(b))
        )
    if not (isinstance(damp, float) or torch.is_tensor(damp)):
        raise TypeError(
            "Expected damp to be of type float or torch.Tensor. Got {0}.".format(
                type(damp)
            )
        )
    if torch.is_tensor(damp) and damp.ndim != 0:
        raise ValueError(
            "Expected torch.Tensor damp to have ndim=0 (scalar). Got {0}.".format(
                damp.ndim
            )
        )
    if A.ndim != 2:
        raise ValueError("A should have ndim=2, but had ndim={}".format(A.ndim))
    if b.ndim != 2:
        raise ValueError("b should have ndim=2, but had ndim={}".format(b.ndim))
    if b.shape[1] != 1:
        raise ValueError("b.shape[1] should 1, but was {0}".format(b.shape[1]))
    if A.shape[0] != b.shape[0]:
        raise ValueError(
            "A.shape[0] and b.shape[0] should be equal ({0} != {1})".format(
                A.shape[0], b.shape[0]
            )
        )
    damp = (
        damp
        if torch.is_tensor(damp)
        else torch.tensor(damp, dtype=A.dtype, device=A.device)
    )

    # Construct the normal equations
    A_t = torch.transpose(A, 0, 1)
    damp_matrix = torch.eye(A.shape[1]).to(A.device)
    At_A = torch.matmul(A_t, A) + damp_matrix * damp

    # Solve the normal equations (for now, by inversion!)
    return torch.matmul(torch.inverse(At_A), torch.matmul(A_t, b))


def gauss_newton_solve(
    src_pc: torch.Tensor,
    tgt_pc: torch.Tensor,
    tgt_normals: torch.Tensor,
    dist_thresh: Union[float, int, None] = None,
):
    r"""Computes Gauss Newton step by forming linear equation. Points from `src_pc` which have a distance greater
    than `dist_thresh` to the closest point in `tgt_pc` will be filtered.

    Args:
        src_pc (torch.Tensor): Source pointcloud (the pointcloud that needs warping).
        tgt_pc (torch.Tensor): Target pointcloud (the pointcloud to which the source pointcloud must be warped to).
        tgt_normals (torch.Tensor): Per-point normal vectors for each point in the target pointcloud.
        dist_thresh (float or int or None): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
            Default: None

    Returns:
        tuple: tuple containing:

        - A (torch.Tensor): linear system equation
        - b (torch.Tensor): linear system residual
        - chamfer_indices (torch.Tensor): Index of the closest point in `tgt_pc` for each point in `src_pc`
            that was not filtered out.

    Shape:
        - src_pc: :math:`(1, N_s, 3)`
        - tgt_pc: :math:`(1, N_t, 3)`
        - tgt_normals: :math:`(1, N_t, 3)`
        - A: :math:`(N_sf, 6)` where :math:`N_sf \leq N_s`
        - b: :math:`(N_sf, 1)` where :math:`N_sf \leq N_s`
        - chamfer_indices: :math:`(1, N_sf)` where :math:`N_sf \leq N_s`
    """
    if not torch.is_tensor(src_pc):
        raise TypeError(
            "Expected src_pc to be of type torch.Tensor. Got {0}.".format(type(src_pc))
        )
    if not torch.is_tensor(tgt_pc):
        raise TypeError(
            "Expected tgt_pc to be of type torch.Tensor. Got {0}.".format(type(tgt_pc))
        )
    if not torch.is_tensor(tgt_normals):
        raise TypeError(
            "Expected tgt_normals to be of type torch.Tensor. Got {0}.".format(
                type(tgt_normals)
            )
        )
    if not (
        isinstance(dist_thresh, float)
        or isinstance(dist_thresh, int)
        or dist_thresh is None
    ):
        raise TypeError(
            "Expected dist_thresh to be of type float or int. Got {0}.".format(
                type(dist_thresh)
            )
        )
    if src_pc.ndim != 3:
        raise ValueError(
            "src_pc should have ndim=3, but had ndim={}".format(src_pc.ndim)
        )
    if tgt_pc.ndim != 3:
        raise ValueError(
            "tgt_pc should have ndim=3, but had ndim={}".format(tgt_pc.ndim)
        )
    if tgt_normals.ndim != 3:
        raise ValueError(
            "tgt_normals should have ndim=3, but had ndim={}".format(tgt_normals.ndim)
        )
    if src_pc.shape[0] != 1:
        raise ValueError(
            "src_pc.shape[0] should be 1, but was {} instead".format(src_pc.shape[0])
        )
    if tgt_pc.shape[0] != 1:
        raise ValueError(
            "tgt_pc.shape[0] should be 1, but was {} instead".format(tgt_pc.shape[0])
        )
    if tgt_normals.shape[0] != 1:
        raise ValueError(
            "tgt_normals.shape[0] should be 1, but was {} instead".format(
                tgt_normals.shape[0]
            )
        )
    if tgt_pc.shape[1] != tgt_normals.shape[1]:
        raise ValueError(
            "tgt_pc.shape[1] and tgt_normals.shape[1] must be equal. Got {0}!={1}".format(
                tgt_pc.shape[1], tgt_normals.shape[1]
            )
        )
    if src_pc.shape[2] != 3:
        raise ValueError(
            "src_pc.shape[2] should be 3, but was {} instead".format(src_pc.shape[2])
        )
    if tgt_pc.shape[2] != 3:
        raise ValueError(
            "tgt_pc.shape[2] should be 3, but was {} instead".format(tgt_pc.shape[2])
        )
    if tgt_normals.shape[2] != 3:
        raise ValueError(
            "tgt_normals.shape[2] should be 3, but was {} instead".format(
                tgt_normals.shape[2]
            )
        )

    src_pc = src_pc.contiguous()
    tgt_pc = tgt_pc.contiguous()
    tgt_normals = tgt_normals.contiguous()

    from chamferdist.chamfer import knn_points

    _KNN = knn_points(src_pc, tgt_pc)
    dist1, idx1 = _KNN.dists.squeeze(-1), _KNN.idx.squeeze(-1)

    dist_filter = (
        torch.ones_like(dist1[0], dtype=torch.bool)
        if dist_thresh is None
        else dist1[0] < dist_thresh
    )
    chamfer_indices = idx1[0][dist_filter].long()

    sx = src_pc[0, dist_filter, 0].view(-1, 1)
    sy = src_pc[0, dist_filter, 1].view(-1, 1)
    sz = src_pc[0, dist_filter, 2].view(-1, 1)

    # Closest point/normal to each source point
    assoc_pts = torch.index_select(tgt_pc, 1, chamfer_indices)
    assoc_normals = torch.index_select(tgt_normals, 1, chamfer_indices)

    # Closest destination point to each source point
    dx = assoc_pts[0, :, 0].view(-1, 1)
    dy = assoc_pts[0, :, 1].view(-1, 1)
    dz = assoc_pts[0, :, 2].view(-1, 1)

    nx = assoc_normals[0, :, 0].view(-1, 1)
    ny = assoc_normals[0, :, 1].view(-1, 1)
    nz = assoc_normals[0, :, 2].view(-1, 1)

    A = torch.cat(
        [nx, ny, nz, nz * sy - ny * sz, nx * sz - nz * sx, ny * sx - nx * sy], 1
    )
    b = nx * (dx - sx) + ny * (dy - sy) + nz * (dz - sz)

    return A, b, chamfer_indices


def point_to_plane_ICP(
    src_pc: torch.Tensor,
    tgt_pc: torch.Tensor,
    tgt_normals: torch.Tensor,
    initial_transform: Optional[torch.Tensor] = None,
    numiters: int = 20,
    damp: float = 1e-8,
    dist_thresh: Union[float, int, None] = None,
):
    r"""Computes a rigid transformation between `tgt_pc` (target pointcloud) and `src_pc` (source pointcloud) using a
    point-to-plane error metric and the LM (Levenberg–Marquardt) solver.

    Args:
        src_pc (torch.Tensor): Source pointcloud (the pointcloud that needs warping).
        tgt_pc (torch.Tensor): Target pointcloud (the pointcloud to which the source pointcloud must be warped to).
        tgt_normals (torch.Tensor): Per-point normal vectors for each point in the target pointcloud.
        initial_transform (torch.Tensor or None): The initial estimate of the transformation between 'src_pc'
            and 'tgt_pc'. If None, will use the identity matrix as the initial transform. Default: None
        numiters (int): Number of iterations to run the optimization for. Default: 20
        damp (float): Damping coefficient for nonlinear least-squares. Default: 1e-8
        dist_thresh (float or int or None): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
            Default: None

    Returns:
        tuple: tuple containing:

        - transform (torch.Tensor): linear system residual
        - chamfer_indices (torch.Tensor): Index of the closest point in `tgt_pc` for each point in `src_pc` that was not
          filtered out.

    Shape:
        - src_pc: :math:`(1, N_s, 3)`
        - tgt_pc: :math:`(1, N_t, 3)`
        - tgt_normals: :math:`(1, N_t, 3)`
        - initial_transform: :math:`(4, 4)`
        - transform: :math:`(4, 4)`
        - chamfer_indices: :math:`(1, N_sf)` where :math:`N_sf \leq N_s`

    """
    if not torch.is_tensor(src_pc):
        raise TypeError(
            "Expected src_pc to be of type torch.Tensor. Got {0}.".format(type(src_pc))
        )
    if not torch.is_tensor(tgt_pc):
        raise TypeError(
            "Expected tgt_pc to be of type torch.Tensor. Got {0}.".format(type(tgt_pc))
        )
    if not torch.is_tensor(tgt_normals):
        raise TypeError(
            "Expected tgt_normals to be of type torch.Tensor. Got {0}.".format(
                type(tgt_normals)
            )
        )
    if not (torch.is_tensor(initial_transform) or initial_transform is None):
        raise TypeError(
            "Expected initial_transform to be of type torch.Tensor. Got {0}.".format(
                type(initial_transform)
            )
        )
    if not isinstance(numiters, int):
        raise TypeError(
            "Expected numiters to be of type int. Got {0}.".format(type(numiters))
        )
    if initial_transform.ndim != 2:
        raise ValueError(
            "Expected initial_transform.ndim to be 2. Got {0}.".format(
                initial_transform.ndim
            )
        )
    if not (initial_transform.shape[0] == 4 and initial_transform.shape[1] == 4):
        raise ValueError(
            "Expected initial_transform.shape to be (4, 4). Got {0}.".format(
                initial_transform.shape
            )
        )
    src_pc = src_pc.contiguous()
    tgt_pc = tgt_pc.contiguous()
    tgt_normals = tgt_normals.contiguous()
    dtype = src_pc.dtype
    device = src_pc.device
    damp = torch.tensor(damp, dtype=dtype, device=device)

    # include initial transform
    initial_transform = (
        torch.eye(4, dtype=dtype, device=device)
        if initial_transform is None
        else initial_transform
    )
    src_pc = transform_pointcloud(src_pc[0], initial_transform).unsqueeze(0)
    transform = initial_transform

    for it in range(numiters):
        # Form the linear system and compute the residual
        A, b, chamfer_indices = gauss_newton_solve(
            src_pc, tgt_pc, tgt_normals, dist_thresh
        )
        residual = b[:, 0]

        # Solve the linear system
        xi = solve_linear_system(A, b, damp)

        # Apply exponential to find the transform
        residual_transform = se3_exp(xi)

        # Find error
        err = torch.dot(residual.t(), residual)
        pc_error = torch.sqrt(torch.sum((torch.mm(A, xi) - b) ** 2))

        # Lookahead error (for LM)
        # calculate transformed cloud
        one_step_pc = transform_pointcloud(src_pc[0], residual_transform).unsqueeze(0)

        # Form new linear system and compute one-step residual
        _, one_step_b, chamfer_indices_onestep = gauss_newton_solve(
            one_step_pc, tgt_pc, tgt_normals, dist_thresh
        )
        one_step_residual = one_step_b[:, 0]

        # Find new error
        new_err = torch.dot(one_step_residual.t(), one_step_residual)

        if new_err < err:
            # We are in a trust region
            src_pc = one_step_pc
            damp = damp / 2

            # update transform
            transform = torch.mm(residual_transform, transform)

        else:
            damp = damp * 2

    return transform, chamfer_indices


def point_to_plane_gradICP(
    src_pc: torch.Tensor,
    tgt_pc: torch.Tensor,
    tgt_normals: torch.Tensor,
    initial_transform: Optional[torch.Tensor] = None,
    numiters: int = 20,
    damp: float = 1e-8,
    dist_thresh: Union[float, int, None] = None,
    lambda_max: Union[float, int] = 2.0,
    B: Union[float, int] = 1.0,
    B2: Union[float, int] = 1.0,
    nu: Union[float, int] = 200.0,
):
    r"""Computes a rigid transformation between `tgt_pc` (target pointcloud) and `src_pc` (source pointcloud) using a
    point-to-plane error metric and gradLM (:math:`\nabla LM`) solver (See gradLM section of 
    `the gradSLAM paper <https://arxiv.org/abs/1910.10672>`__).  The iterate and damping coefficient are updated by:

    .. math::

        lambda_1 = Q_\lambda(r_0, r_1) & = \lambda_{min} + \frac{\lambda_{max} -
        \lambda_{min}}{1 + e^{-B (r_1 - r_0)}} \\
        Q_x(r_0, r_1) & = x_0 + \frac{\delta x_0}{\sqrt[nu]{1 + e^{-B2*(r_1 - r_0)}}}`

    Args:
        src_pc (torch.Tensor): Source pointcloud (the pointcloud that needs warping).
        tgt_pc (torch.Tensor): Target pointcloud (the pointcloud to which the source pointcloud must be warped to).
        tgt_normals (torch.Tensor): Per-point normal vectors for each point in the target pointcloud.
        initial_transform (torch.Tensor or None): The initial estimate of the transformation between 'src_pc' 
            and 'tgt_pc'. If None, will use the identity matrix as the initial transform. Default: None
        numiters (int): Number of iterations to run the optimization for. Default: 20
        damp (float): Damping coefficient for nonlinear least-squares. Default: 1e-8
        dist_thresh (float or int or None): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
            Default: None
        lambda_max (float or int): Maximum value the damping function can assume (`lambda_min` will be 
            :math:`\frac{1}{\text{lambda_max}}`)
        B (float or int): gradLM falloff control parameter
        B2 (float or int): gradLM control parameter
        nu (float or int): gradLM control parameter

    Returns:
        tuple: tuple containing:

        - transform (torch.Tensor): linear system residual
        - chamfer_indices (torch.Tensor): Index of the closest point in `tgt_pc` for each point in `src_pc` that was not
          filtered out.

    Shape:
        - src_pc: :math:`(1, N_s, 3)`
        - tgt_pc: :math:`(1, N_t, 3)`
        - tgt_normals: :math:`(1, N_t, 3)`
        - initial_transform: :math:`(4, 4)`
        - transform: :math:`(4, 4)`
        - chamfer_indices: :math:`(1, N_sf)` where :math:`N_sf \leq N_s`

    """
    if not torch.is_tensor(src_pc):
        raise TypeError(
            "Expected src_pc to be of type torch.Tensor. Got {0}.".format(type(src_pc))
        )
    if not torch.is_tensor(tgt_pc):
        raise TypeError(
            "Expected tgt_pc to be of type torch.Tensor. Got {0}.".format(type(tgt_pc))
        )
    if not torch.is_tensor(tgt_normals):
        raise TypeError(
            "Expected tgt_normals to be of type torch.Tensor. Got {0}.".format(
                type(tgt_normals)
            )
        )
    if not (torch.is_tensor(initial_transform) or initial_transform is None):
        raise TypeError(
            "Expected initial_transform to be of type torch.Tensor. Got {0}.".format(
                type(initial_transform)
            )
        )
    if not isinstance(numiters, int):
        raise TypeError(
            "Expected numiters to be of type int. Got {0}.".format(type(numiters))
        )
    if not (isinstance(lambda_max, float) or isinstance(lambda_max, int)):
        raise TypeError(
            "Expected lambda_max to be of type float or int; got {0}".format(
                type(lambda_max)
            )
        )
    if not (isinstance(B, float) or isinstance(B, int)):
        raise TypeError(
            "Expected B to be of type float or int; got {0}".format(type(B))
        )
    if not (isinstance(B2, float) or isinstance(B2, int)):
        raise TypeError(
            "Expected B2 to be of type float or int; got {0}".format(type(B2))
        )
    if not (isinstance(nu, float) or isinstance(nu, int)):
        raise TypeError(
            "Expected nu to be of type float or int; got {0}".format(type(nu))
        )
    if initial_transform.ndim != 2:
        raise ValueError(
            "Expected initial_transform.ndim to be 2. Got {0}.".format(
                initial_transform.ndim
            )
        )
    if not (initial_transform.shape[0] == 4 and initial_transform.shape[1] == 4):
        raise ValueError(
            "Expected initial_transform.shape to be (4, 4). Got {0}.".format(
                initial_transform.shape
            )
        )
    src_pc = src_pc.contiguous()
    tgt_pc = tgt_pc.contiguous()
    tgt_normals = tgt_normals.contiguous()
    dtype = src_pc.dtype
    device = src_pc.device
    damp = torch.tensor(damp, dtype=dtype, device=device)
    lambda_min = 1 / lambda_max

    # include initial transform
    initial_transform = (
        torch.eye(4, dtype=dtype, device=device)
        if initial_transform is None
        else initial_transform
    )
    src_pc = transform_pointcloud(src_pc[0], initial_transform).unsqueeze(0)
    transform = initial_transform

    for it in range(numiters):
        # Form the linear system and compute the residual
        A, b, chamfer_indices = gauss_newton_solve(
            src_pc, tgt_pc, tgt_normals, dist_thresh
        )
        residual = b[:, 0]

        # Solve the linear system
        xi = solve_linear_system(A, b, damp)

        # Apply exponential to find the transform
        residual_transform = se3_exp(xi)

        # Find error
        err = torch.dot(residual.t(), residual)
        pc_error = torch.sqrt(torch.sum((torch.mm(A, xi) - b) ** 2))

        # Lookahead error (for LM)
        # calculate transformed cloud
        one_step_pc = transform_pointcloud(src_pc[0], residual_transform).unsqueeze(0)

        # Form new linear system and compute one-step residual
        _, one_step_b, chamfer_indices_onestep = gauss_newton_solve(
            one_step_pc, tgt_pc, tgt_normals, dist_thresh
        )
        one_step_residual = one_step_b[:, 0]

        # Find new error
        new_err = torch.dot(one_step_residual.t(), one_step_residual)

        # smooth LM routine
        errdiff = new_err - err

        # NEW ADDITION: clamping to ensure gradient flow
        errdiff = errdiff.clamp(-70.0, 70.0)

        damp_new = lambda_min + (lambda_max - lambda_min) / (
            1 + torch.exp(-B * errdiff)
        )
        damp = damp * damp_new

        # residual transform should be perturbed by sigmoid
        sigmoid = 1 / ((1 + torch.exp(-B2 * errdiff)) ** (1 / nu))

        # calculate perturbation
        residual_transform = se3_exp(sigmoid * xi)
        src_pc = transform_pointcloud(src_pc[0], residual_transform).unsqueeze(0)
        transform = torch.mm(residual_transform, transform)

    return transform, chamfer_indices


def downsample_pointclouds(
    pointclouds: Pointclouds, pc2im_bnhw: torch.Tensor, ds_ratio: int
) -> Pointclouds:
    r"""Downsamples active points of pointclouds (points that project inside the live frame) and removes non-active
    points.

    Args:
        pointclouds (gradslam.Pointclouds): Pointclouds to downsample
        pc2im_bnhw (torch.Tensor): Active map points lookup table. Each row contains batch index `b`, point
            index (in pointclouds) `n`, and height and width index after projection to live frame `h` and `w`
            respectively.
        ds_ratio (int): Downsampling ratio

    Returns:
        gradslam.Pointclouds: Downsampled pointclouds

    Shape:
        - pc2im_bnhw: :math:`(\text{num_active_map_points}, 4)`

    """
    if not isinstance(pointclouds, Pointclouds):
        raise TypeError(
            "Expected pointclouds to be of type gradslam.Pointclouds. Got {0}.".format(
                type(pointclouds)
            )
        )
    if not torch.is_tensor(pc2im_bnhw):
        raise TypeError(
            "Expected pc2im_bnhw to be of type torch.Tensor. Got {0}.".format(
                type(pc2im_bnhw)
            )
        )
    if not isinstance(ds_ratio, int):
        raise TypeError(
            "Expected ds_ratio to be of type int. Got {0}.".format(type(ds_ratio))
        )
    if pc2im_bnhw.ndim != 2:
        raise ValueError(
            "Expected pc2im_bnhw to have ndim=2. Got {0}.".format(pc2im_bnhw.ndim)
        )
    if pc2im_bnhw.shape[1] != 4:
        raise ValueError(
            "pc2im_bnhw.shape[1] must be 4, but was {0}.".format(pc2im_bnhw.shape[1])
        )

    B = len(pointclouds)

    # Find indices of points to keep after downsampling
    pc2im_bnhw = pc2im_bnhw[pc2im_bnhw[..., 2] % ds_ratio == 0]
    pc2im_bnhw = pc2im_bnhw[pc2im_bnhw[..., 3] % ds_ratio == 0]

    # Downsample points and normals
    maps_points = [
        pointclouds.points_list[b][pc2im_bnhw[pc2im_bnhw[..., 0] == b][..., 1]]
        for b in range(B)
    ]
    maps_normals = (
        None
        if pointclouds.normals_list is None
        else [
            pointclouds.normals_list[b][pc2im_bnhw[pc2im_bnhw[..., 0] == b][..., 1]]
            for b in range(B)
        ]
    )
    maps_colors = (
        None
        if pointclouds.colors_list is None
        else [
            pointclouds.colors_list[b][pc2im_bnhw[pc2im_bnhw[..., 0] == b][..., 1]]
            for b in range(B)
        ]
    )
    return Pointclouds(points=maps_points, normals=maps_normals, colors=maps_colors)


def downsample_rgbdimages(rgbdimages: RGBDImages, ds_ratio: int) -> Pointclouds:
    r"""Downsamples points and normals of RGBDImages and returns a gradslam.Pointclouds object

    Args:
        rgbdimages (gradslam.RGBDImages): RGBDImages to downsample
        ds_ratio (int): Downsampling ratio

    Returns:
        gradslam.Pointclouds: Downsampled points and normals

    """
    if not isinstance(rgbdimages, RGBDImages):
        raise TypeError(
            "Expected rgbdimages to be of type gradslam.RGBDImages. Got {0}.".format(
                type(rgbdimages)
            )
        )
    if not isinstance(ds_ratio, int):
        raise TypeError(
            "Expected ds_ratio to be of type int. Got {0}.".format(type(ds_ratio))
        )
    if rgbdimages.shape[1] != 1:
        raise ValueError(
            "Sequence length of rgbdimages must be 1, but was {0}.".format(
                rgbdimages.shape[1]
            )
        )

    B = len(rgbdimages)

    # Valid depths mask
    mask = rgbdimages.valid_depth_mask.squeeze(-1)[..., ::ds_ratio, ::ds_ratio]

    # Downsample points and normals
    points = [
        rgbdimages.global_vertex_map[b][..., ::ds_ratio, ::ds_ratio, :][mask[b]]
        for b in range(B)
    ]
    normals = [
        rgbdimages.global_normal_map[b][..., ::ds_ratio, ::ds_ratio, :][mask[b]]
        for b in range(B)
    ]
    colors = [
        rgbdimages.rgb_image[b][..., ::ds_ratio, ::ds_ratio, :][mask[b]]
        for b in range(B)
    ]
    return Pointclouds(points=points, normals=normals, colors=colors)
