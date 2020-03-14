import numpy as np

import open3d as o3d
import torch
from chamferdist import ChamferDistance
# from se3utils import SE3_exp

chamferDist = ChamferDistance()

debug_ = False


def transform_pointcloud(pointcloud: torch.Tensor, transform: torch.Tensor):
    r""" Applies a rigid-body transformation to a pointcloud.

    Args:
        pointcloud (torch.Tensor): Pointcloud to be transformed
                                   (shape: numpts x 3)
        transform (torch.Tensor): An SE(3) rigid-body transform matrix
                                  (shape: 4 x 4)

    Returns:
        transformed_pointcloud (torch.Tensor): Rotated and translated cloud
                                               (shape: numpts x 3)

    """

    # Ensure that transform and pointcloud are on the same device
    if transform.device != pointcloud.device:
        transform = transform.to(pointcloud.device)

    # Rotation matrix
    rmat = transform[:3, :3]
    # Translation vector
    tvec = transform[:3, 3]

    # Transpose the pointcloud (to enable broadcast of rotation to each point)
    pointcloud_transpose = torch.transpose(pointcloud, 0, 1)
    # Rotate and translate cloud
    tf_cloud = torch.matmul(rmat, pointcloud_transpose) + tvec.unsqueeze(1)
    # Transpose the transformed cloud to original dimensions
    tf_cloud = torch.transpose(tf_cloud, 0, 1)

    return tf_cloud


def transform_normals(normals: torch.Tensor, transform: torch.Tensor):
    r"""Applies a rotation to a tensor containing point normals.

    Args:
        normals (torch.Tensor): Normal vectors (shape: numpoints x 3)
    """

    if transform.device != normals.device:
        transform = transform.to(normals.device)

    # Rotation
    R = transform[:3, :3]

    # apply transpose to normals
    normals_transpose = torch.transpose(normals, 0, 1)

    # transpose after transform
    tgt_normals = torch.transpose(torch.matmul(R, normals_transpose), 0, 1)

    return tgt_normals


def solve_linear_system(
    A: torch.Tensor, b: torch.Tensor, damping_coefficient: float = 0.0
):
    r"""Solves the normal equations of a linear system Ax = b,
    given the constraint matrix A and the coefficient vector b.

    NOTE:
        We actually solve the normal equations, not the linear
        system itself. That is, we solve :math:`A^T A x = A^T b`,
        not :math:`Ax = b`.

    Args:
        A (torch.Tensor): The constraint matrix of the linear system.
                          (shape: num_of_equations x num_of_variables)
        b (torch.Tensor): The coefficient vector of the linear system.
                          (shape: num_of_variables x 1)
        damping_coefficient (torch.Tensor, optional): Damping coefficient
            to optionally condition the linear system (in practice, a
            damping coefficient of :math:`\rho` means that we are solving
            a modified linear system that adds a tiny :math:`\rho` to each
            diagonal element of the constraint matrix :math:`A`, so that
            the linear system becomes :math:`(A^TA + \rho I)x = b`, where
            :math:`I` is the identity matrix of shape num_of_variables x
            num_of_variables) (shape: 1).

    Returns:
        (torch.Tensor): Solution vector of the normal equations of the
                        linear system (shape: num_of_variables x 1)
    """

    if b.device != A.device:
        b = b.to(A.device)
    if damping_coefficient.device != A.device:
        damping_coefficient = damping_coefficient.to(A.device)

    # Construct the normal equations
    A_t = torch.transpose(A, 0, 1)
    damping_matrix = torch.eye(A.shape[1]).to(A.device)
    At_A = torch.matmul(A_t, A) + damping_matrix * damping_coefficient

    # Solve the normal equations (for now, by inversion!)
    return torch.matmul(torch.inverse(At_A), torch.matmul(A_t, b))


def gauss_newton_solve(
    src_pc: torch.Tensor, tgt_pc: torch.Tensor, tgt_normals: torch.Tensor
):
    r"""Computes Gauss Newton step by forming linear equation

    Args: 
        src_pc (torch.Tensor): Source pointcloud (the pointcloud that needs
            warping) (shape: 1 x M x 3).
        tgt_pc (torch.Tensor): Target pointcloud (the pointcloud to which the
            source pointcloud must be warped to) (shape: 1 x N x 3).
        tgt_normals (torch.Tensor): Per-point normal vectors for each point
            in the target pointcloud (shape: 1 x N x 3).

        Returns:
        A (torch.Tensor): linear system equation
        b (torch.Tensor): linear system residual
    """
    src_pc = src_pc.cuda().contiguous()

    sx = src_pc[0, :, 0].view(-1, 1)
    sy = src_pc[0, :, 1].view(-1, 1)
    sz = src_pc[0, :, 2].view(-1, 1)

    dist1, _, idx1, _ = chamferDist(src_pc, tgt_pc)

    dist_filter = (dist1[0] < 0.1).byte()
    idx1_filtered = idx1[0][dist_filter].long()

    assoc_pts = torch.index_select(tgt_pc, 1, idx1[0].long().flatten())

    # Closest normal to each source point
    assoc_normals = torch.index_select(tgt_normals, 0, idx1[0].long().flatten())

    nx = assoc_normals[:, 0].view(-1, 1)
    ny = assoc_normals[:, 1].view(-1, 1)
    nz = assoc_normals[:, 2].view(-1, 1)

    # Closest destination point to each source point
    dx = assoc_pts[0, :, 0].view(-1, 1)
    dy = assoc_pts[0, :, 1].view(-1, 1)
    dz = assoc_pts[0, :, 2].view(-1, 1)

    A = torch.cat(
        [nx, ny, nz, nz * sy - ny * sz, nx * sz - nz * sx, ny * sx - nx * sy], 1
    )
    b = nx * (dx - sx) + ny * (dy - sy) + nz * (dz - sz)

    # return A,b, idx1_filtered.unsqueeze(0)
    return A, b, idx1


def visualize_two_clouds(src_pc: torch.Tensor, tgt_pc: torch.Tensor, m: int, n: int):

    r"""Debug function to visualize pointclouds during registration

    Args: 
        src_pc (torch.Tensor): Source pointcloud (the pointcloud that needs
            warping) (shape: 1 x M x 3).
        tgt_pc (torch.Tensor): Target pointcloud (the pointcloud to which the
            source pointcloud must be warped to) (shape: 1 x N x 3).

    Returns:
        None
    """

    # reshape cloud for o3d
    disp_pts = tgt_pc.reshape(m * n, 3)
    disp_color = np.zeros((m * n, 3))
    disp_color[:, 0] = 255  # red

    orig_pts = src_pc.reshape(m * n, 3)
    orig_color = np.zeros((m * n, 3))
    orig_color[:, 1] = 255  # green

    # o3d objects to view clouds
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(disp_pts.cpu().detach().numpy())
    pcd1.colors = o3d.utility.Vector3dVector(disp_color)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(orig_pts.cpu().detach().numpy())
    pcd2.colors = o3d.utility.Vector3dVector(orig_color)

    o3d.visualization.draw_geometries([pcd1, pcd2])


def generic_visualize_two_clouds(src_pc: torch.Tensor, tgt_pc: torch.Tensor):

    r"""Debug function to visualize pointclouds during registration

    Args: 
        src_pc (torch.Tensor): Source pointcloud (the pointcloud that needs
            warping) (shape: 1 x M x 3).
        tgt_pc (torch.Tensor): Target pointcloud (the pointcloud to which the
            source pointcloud must be warped to) (shape: 1 x N x 3).

    Returns:
        None
    """

    # reshape cloud for o3d
    disp_pts = tgt_pc.reshape(-1, 3)
    disp_color = np.zeros((disp_pts.shape[0], 3))
    disp_color[:, 0] = 255  # red

    orig_pts = src_pc.reshape(-1, 3)
    orig_color = np.zeros((orig_pts.shape[0], 3))
    orig_color[:, 1] = 255  # green

    # o3d objects to view clouds
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(disp_pts.cpu().detach().numpy())
    pcd1.colors = o3d.utility.Vector3dVector(disp_color)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(orig_pts.cpu().detach().numpy())
    pcd2.colors = o3d.utility.Vector3dVector(orig_color)

    o3d.visualization.draw_geometries([pcd1, pcd2])


def point_to_plane_ICP(
    src_pc: torch.Tensor,
    tgt_pc: torch.Tensor,
    tgt_normals: torch.Tensor,
    initial_transform: torch.Tensor,
    m: int,
    n: int,
    numiters: int = 20,
    tol: float = 1e-5,
    damp: torch.Tensor = torch.FloatTensor([0.0]),
):
    r"""Computes a rigid transformation between `new_pc` (target pointcloud)
    and `src_pc` (source pointcloud) using a point-to-plane error metric.

    Args:
        src_pc (torch.Tensor): Source pointcloud (the pointcloud that needs
            warping) (shape: 1 x M x 3).
        tgt_pc (torch.Tensor): Target pointcloud (the pointcloud to which the
            source pointcloud must be warped to) (shape: 1 x N x 3).
        tgt_normals (torch.Tensor): Per-point normal vectors for each point
            in the target pointcloud (shape: 1 x N x 3).
        m (int): downsampled height
        n (int): downsampled width
        numiters (int): Number of iterations to run the optimization for
            (default: 20).
        tol (float): Tolerance for the optimizer convergence criteria
            (default: 1e-5).
        damp (torch.Tensor, optional): Damping coefficient for nonlinear
            least-squares (default: 0.) (shape: 1). TODO: Not used.
    """

    tgt_pc = tgt_pc.cuda().contiguous()
    tgt_normals = tgt_normals.cuda().contiguous()

    # if (debug_):
    #     visualize_two_clouds(src_pc, tgt_pc, m, n)

    damp = torch.FloatTensor([1e-8]).to(src_pc.device)
    damp_coeff = damp.clone()

    # include initial transform
    src_pc = transform_pointcloud(src_pc[0], initial_transform).unsqueeze(0)
    transform = initial_transform

    # if not to include initial transform
    # transform = torch.eye(4).cuda()

    for it in range(numiters):

        # Form the linear system and compute the residual
        A, b, chamfer_indices = gauss_newton_solve(src_pc, tgt_pc, tgt_normals)
        residual = b[:, 0]

        # Solve the linear system
        xi = solve_linear_system(A, b, damp)
        # xi.register_hook(lambda grad: print("Gradients xi: ", grad))

        # Apply exponential to find the transform
        residual_transform = SE3_exp(xi)
        # residual_transform.register_hook(lambda grad: print("Residual transform grad: ", grad))

        # Find error
        err = torch.dot(residual.t(), residual)
        pc_error = torch.sqrt(torch.sum((torch.mm(A, xi) - b) ** 2))

        # Lookahead error (for LM)

        # calculate transformed cloud
        one_step_pc = transform_pointcloud(src_pc[0], residual_transform).unsqueeze(0)

        # Form new linear system and compute one-step residual
        _, one_step_b, chamfer_indices_onestep = gauss_newton_solve(
            one_step_pc, tgt_pc, tgt_normals
        )
        one_step_residual = one_step_b[:, 0]

        # Find new error
        new_err = torch.dot(one_step_residual.t(), one_step_residual)

        if debug_:
            print(err, new_err, pc_error)

        if new_err < err:
            # We are in a trust region
            src_pc = one_step_pc
            damp_coeff = damp_coeff / 2

            # update transform
            transform = torch.mm(residual_transform, transform)

        else:
            damp_coeff = damp_coeff * 2

    if debug_:
        visualize_two_clouds(src_pc, tgt_pc, m, n)
        print(err, new_err, pc_error)

    return transform, chamfer_indices


def point_to_plane_ICP_smooth(
    src_pc: torch.Tensor,
    tgt_pc: torch.Tensor,
    tgt_normals: torch.Tensor,
    initial_transform: torch.Tensor,
    m: int,
    n: int,
    numiters: int = 10,
    tol: float = 1e-5,
    damp: torch.Tensor = torch.FloatTensor([0.0]),
    lambda_max: float = 2.0,
    B: float = 1.0,
    B2: float = 1.0,
    nu: float = 200.0,
):
    r"""Computes a rigid transformation between `new_pc` (target pointcloud)
    and `src_pc` (source pointcloud) using a point-to-plane error metric.

    Args:
        src_pc (torch.Tensor): Source pointcloud (the pointcloud that needs
            warping) (shape: 1 x M x 3).
        tgt_pc (torch.Tensor): Target pointcloud (the pointcloud to which the
            source pointcloud must be warped to) (shape: 1 x N x 3).
        tgt_normals (torch.Tensor): Per-point normal vectors for each point
            in the target pointcloud (shape: 1 x N x 3).
        m (int): downsampled height
        n (int): downsampled width
        numiters (int): Number of iterations to run the optimization for
            (default: 10).
        tol (float): Tolerance for the optimizer convergence criteria
            (default: 1e-5).
        damp (torch.Tensor, optional): Damping coefficient for nonlinear
            least-squares (default: 0.) (shape: 1).
    """

    lambda_min = 1 / lambda_max
    tgt_pc = tgt_pc.cuda().contiguous()
    tgt_normals = tgt_normals.cuda().contiguous()

    if debug_:
        visualize_two_clouds(src_pc, tgt_pc, m, n)

    damp = torch.FloatTensor([1e-8]).to(src_pc.device)
    damp_coeff = damp.clone()

    # include initial transform
    src_pc = transform_pointcloud(src_pc[0], initial_transform).unsqueeze(0)
    transform = initial_transform

    # if not to include initial transform
    # transform = torch.eye(4).cuda()

    for it in range(numiters):

        # Form the linear system and compute the residual
        A, b, chamfer_indices = gauss_newton_solve(src_pc, tgt_pc, tgt_normals)
        residual = b[:, 0]

        # Solve the linear system
        xi = solve_linear_system(A, b, damp)

        # Apply exponential to find the transform
        residual_transform = SE3_exp(xi)

        # Find error
        err = torch.dot(residual.t(), residual)
        pc_error = torch.sqrt(torch.sum((torch.mm(A, xi) - b) ** 2))

        # Lookahead error (for LM)

        # calculate transformed cloud
        one_step_pc = transform_pointcloud(src_pc[0], residual_transform).unsqueeze(0)

        # Form new linear system and compute one-step residual
        _, one_step_b, chamfer_indices_onestep = gauss_newton_solve(
            one_step_pc, tgt_pc, tgt_normals
        )
        one_step_residual = one_step_b[:, 0]

        # Find new error
        new_err = torch.dot(one_step_residual.t(), one_step_residual)

        if debug_:
            print(err, new_err, pc_error)

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

        if debug_:
            sigmoid.register_hook(lambda grad: print("Error diff: ", errdiff))
            sigmoid.register_hook(lambda grad: print("Gradients Sigmoid: ", grad))

        if debug_:
            print(sigmoid)
        # calculate perturbation
        residual_transform = SE3_exp(sigmoid * xi)
        src_pc = transform_pointcloud(src_pc[0], residual_transform).unsqueeze(0)
        transform = torch.mm(residual_transform, transform)

    if debug_:
        visualize_two_clouds(src_pc, tgt_pc, m, n)

    # print (err, new_err, pc_error)
    return transform, chamfer_indices


def ICP(global_map, new_frame, initial_transform, grad_LM, ds_ratio):
    r"""Wrapper for ICP for pointfusion
    """
    last_pose = global_map.current_pose

    # downsample global map's frame projections, convert to pointcloud
    ref_pointcloud = (
        global_map.last_proj_vertex_map[::ds_ratio, ::ds_ratio]
        .reshape(-1, 3)
        .unsqueeze(0)
    )
    ref_pc_normals = global_map.last_proj_normal_map[::ds_ratio, ::ds_ratio].reshape(
        -1, 3
    )
    # ref_pc_normals.retain_grad() # Set torch.autograd.set_detect_anomaly to False to use
    # ref_pc_normals.register_hook(lambda grad: print("\nGradients ref_pc_normals:", torch.max(grad)))

    # downsample new vertex map, convert to pointcloud in global coords
    new_pointcloud = new_frame.vertex_map[::ds_ratio, ::ds_ratio].reshape(-1, 3)
    new_pointcloud = transform_pointcloud(new_pointcloud, last_pose).unsqueeze(0)
    h_ds = new_frame.h // ds_ratio
    w_ds = new_frame.w // ds_ratio

    # TODO: in point2planeICP documentation, tgt_normals shape should be Nx3 not 1xNx3
    if grad_LM:
        transform, correspondence = point_to_plane_ICP_smooth(
            new_pointcloud,
            ref_pointcloud,
            ref_pc_normals,
            initial_transform,
            h_ds,
            w_ds,
        )
    else:
        transform, correspondence = point_to_plane_ICP(
            new_pointcloud,
            ref_pointcloud,
            ref_pc_normals,
            initial_transform,
            h_ds,
            w_ds,
        )

    return transform, correspondence
