from typing import Union

from tqdm import tqdm

import torch
import torch.nn.functional as F


def find_correspondences(
    pc_points: torch.Tensor,
    pc_normals: torch.Tensor,
    pc_colors: torch.Tensor,
    pc_ccounts: torch.Tensor,
    input_points: torch.Tensor,
    input_normals: torch.Tensor,
    transform: torch.Tensor,
    h: torch.Tensor,
    w: torch.Tensor,
    K: torch.Tensor,
    dist2_th: float,
    dot_th: float,
):
    r"""Finds and returns points from the global map which correspond to the 
    points from the new frame (See section 4.1 of PointFusion paper).

    Args:
        pc_points (torch.Tensor): Global map's pointcloud in global 
                                  coords (shape: num_points x 3)
        pc_normals (torch.Tensor): Global map's pointcloud normals in 
                                   global coords (shape: num_points x 3)
        pc_colors (torch.Tensor): Global map's pointcloud colors
                                  (shape: num_points x 3)
        pc_ccounts (torch.Tensor): Global map's pointcloud confidence counter
                                   (shape: num_points)
        input_points (torch.Tensor): Vertices of the new frame in global 
                                     coords (shape: height x width x 3)
        input_normals (torch.Tensor): Normals of new frame's points in 
                                      global coords (shape: height x width x 3)
        transform (torch.Tensor): The global camera pose (shape: 4 x 4)
        h (int): Frame height
        w (int): Frame width
        K (torch.Tensor): Camera intrinsics matrix (shape: 4 x 4)
        dist2_th (float): Squared distance threshold.
        dot_th (float): Cosine similarity threshold.

    Returns:
        proj_points (torch.Tensor): Vertices of global map's pointcloud (in 
                                    global coords) which project onto the 
                                    (height x width) camera frame.
                                    (shape: height x width x 3)
        proj_normals (torch.Tensor): Normals of points from the global
                                     pointcloud (in global coords) that
                                     project onto the (height x width)
                                     camera frame. (shape: height x width x 3)
        proj_ccounts (torch.Tensor): Confidence counter of points from the 
                                     global pointcloud that project onto the
                                     (height x width) camera frame.
                                     (shape: height x width)
        proj_colors (torch.Tensor): RGB color of points from the global 
                                    pointcloud that project onto the 
                                    (height x width) camera frame.
                                    (shape: height x width x 3)
        valid_inds (torch.Tensor): Indices of the points from global map which
                                   correspond to points from the new frame.
                                   (shape: num_corresponding_points)
    """
    device = pc_points.device
    proj_points = torch.zeros([h, w, 3]).to(device)
    proj_normals = torch.zeros([h, w, 3]).to(device)
    proj_ccounts = torch.zeros([h, w]).to(device)
    proj_colors = torch.zeros([h, w, 3]).to(device)

    # project map to frame
    frame_projected_map = project_map_to_frame(pc_points, pc_normals, pc_colors, pc_ccounts, h, w, K, transform)
    valid_points, valid_normals, valid_ccounts = frame_projected_map[:3]
    valid_colors, valid_inds, valid_img_inds = frame_projected_map[3:]

    # filter1: remove map points with dissimilar vertices or normals
    mask = find_similar_points(
        valid_points, valid_normals, valid_img_inds, input_points, input_normals, dist2_th, dot_th,
    )

    if torch.sum(mask) == 0:
        tqdm.write("Warning: all map projections were filtered out")
        return (
            proj_points,
            proj_normals,
            proj_ccounts,
            proj_colors,
            valid_inds,
        )

    valid_points = valid_points[mask]
    valid_normals = valid_normals[mask]
    valid_ccounts = valid_ccounts[mask]
    valid_colors = valid_colors[mask]
    valid_inds = valid_inds[mask]
    valid_img_inds = valid_img_inds[mask]

    # filter2: if multiple points project to same pixel, keep the best one only
    best_inds = find_best_unique_correspondences(valid_img_inds, valid_ccounts, valid_points)
    valid_points = valid_points[best_inds]
    valid_normals = valid_normals[best_inds]
    valid_ccounts = valid_ccounts[best_inds]
    valid_colors = valid_colors[best_inds]
    valid_inds = valid_inds[best_inds]
    valid_img_inds = valid_img_inds[best_inds]

    # place corresponding points in frame
    proj_points[valid_img_inds[:, 1], valid_img_inds[:, 0], :] = valid_points
    proj_normals[valid_img_inds[:, 1], valid_img_inds[:, 0], :] = valid_normals
    proj_ccounts[valid_img_inds[:, 1], valid_img_inds[:, 0]] = valid_ccounts
    proj_colors[valid_img_inds[:, 1], valid_img_inds[:, 0]] = valid_colors

    return (proj_points, proj_normals, proj_ccounts, proj_colors, valid_inds)


def average_maps(
    proj_points: torch.Tensor,
    proj_normals: torch.Tensor,
    proj_ccounts: torch.Tensor,
    proj_colors: torch.Tensor,
    input_points: torch.Tensor,
    input_normals: torch.Tensor,
    input_colors: torch.Tensor,
    alpha: torch.Tensor,
):
    r"""Computes weighted average of new frame data with the global map
    (See section 4.2 of PointFusion paper)

    Args:
        proj_points (torch.Tensor): Vertices of global map's pointcloud (in 
                                    global coords) which project onto the 
                                    (height x width) camera frame.
                                    (shape: height x width x 3)
        proj_normals (torch.Tensor): Normals of points from the global
                                     pointcloud (in global coords) that
                                     project onto the (height x width)
                                     camera frame. (shape: height x width x 3)
        proj_ccounts (torch.Tensor): Confidence counter of points from the 
                                     global pointcloud that project onto the
                                     (height x width) camera frame.
                                     (shape: height x width)
        proj_colors (torch.Tensor): RGB color of points from the global 
                                    pointcloud that project onto the 
                                    (height x width) camera frame.
                                    (shape: height x width x 3)
        input_points (torch.Tensor): Vertices of the new frame in global 
                                     coords (shape: height x width x 3)
        input_normals (torch.Tensor): Normals of new frame's points in 
                                      global coords (shape: height x width x 3)
        input_colors (torch.Tensor): RGB color of new frame's points
                                     (shape: height x width x 3)
        alpha (torch.Tensor): Sample confidences of new frame's points
                              (shape: height x width)

    Returns:
        updated_points (torch.Tensor): Weighted average of every vertex
                                       in new frame with the corresponding 
                                       vertex in global map if correspondance
                                       is found, else return new frame's vertex
                                       (shape: height x width x 3)
        updated_normals (torch.Tensor): Weighted average of every normal
                                        in new frame with the corresponding
                                        normal in the global map if
                                        correspondance is found, else return 
                                        new frame's normals
                                        (shape: height x width x 3)
        updated_ccounts (torch.Tensor): Sum of every new frame's alpha with the
                                        ccount of the corresponding vertex in
                                        the global map if correspondance is
                                        found, else return new frame's alpha
                                        (shape: height x width)
        updated_colors (torch.Tensor): Weighted average of every vertex's color
                                       in new frame with the color of the
                                       corresponding vertex in the global map 
                                       if correspondance is found, else return
                                       new frame's colors
                                       (shape: height x width x 3)
    """
    # TODO: Add the condition for radius of points before applying averaging
    updated_ccounts = proj_ccounts + alpha

    proj_ccounts = proj_ccounts.unsqueeze(2)
    updated_ccounts = updated_ccounts.unsqueeze(2)
    alpha = alpha.unsqueeze(2)

    updated_points = (proj_ccounts * proj_points) + (alpha * input_points)
    updated_points /= updated_ccounts
    updated_normals = (proj_ccounts * proj_normals) + (alpha * input_normals)
    updated_normals /= updated_ccounts
    updated_colors = (proj_ccounts * proj_colors) + (alpha * input_colors)
    updated_colors /= updated_ccounts
    updated_ccounts = updated_ccounts[:, :, 0]

    # alpha.retain_grad() # Set torch.autograd.set_detect_anomaly to False to use
    # alpha.register_hook(lambda grad: print("Gradients alpha:", torch.max(grad)))
    return updated_points, updated_normals, updated_ccounts, updated_colors


def project_map_to_frame(
    pc_points: torch.Tensor,
    pc_normals: torch.Tensor,
    pc_colors: torch.Tensor,
    pc_ccounts: torch.Tensor,
    h: torch.Tensor,
    w: torch.Tensor,
    K: torch.Tensor,
    transform: torch.Tensor,
):
    r"""Projects global model map provided in global coordinates onto the 
    image frame. 

    Args:
        pc_points (torch.Tensor): Global map's pointcloud in global 
                                  coords (shape: num_points x 3)
        pc_normals (torch.Tensor): Global map's pointcloud normals in 
                                   global coords (shape: num_points x 3)
        pc_colors (torch.Tensor): Global map's pointcloud colors
                                  (shape: num_points x 3)
        pc_ccounts (torch.Tensor): Global map's pointcloud confidence counter
                                   (shape: num_points)
        h (int): Frame height
        w (int): Frame width
        K (torch.Tensor): Camera intrinsics matrix (shape: 4 x 4)
        transform (torch.Tensor): The global camera pose (shape: 4 x 4)

    Returns:
        valid_points (torch.Tensor): Vertices of global map's pointcloud (in 
                                     global coords) which project onto the 
                                     (height x width) camera frame.
                                     (shape: P x 3)
        valid_normals (torch.Tensor): Normals of points from the global
                                      pointcloud (in global coords) that
                                      project onto the (height x width)
                                      camera frame. (shape: P x 3)
        valid_ccounts (torch.Tensor): Confidence counter of points from the 
                                      global pointcloud that project onto the
                                      (height x width) camera frame.
                                      (shape: P)
        valid_colors (torch.Tensor): RGB color of points from the global 
                                     pointcloud that project onto the 
                                     (height x width) camera frame.
                                     (shape: P x 3)
        valid_indices (torch.Tensor): Indices of points from pointcloud which
                                      project onto the (height x width) camera
                                      frame. (shape: P)
        valid_image_indices (torch.Tensor): Pixel positions of where each 
                                            point from pointcloud would be
                                            in rasterized frame.
                                            (shape: P x 2)
    """
    device = pc_points.device
    indices = torch.arange(pc_points.shape[0]).long()

    # get homogenuous coordinates
    ones_concat = torch.ones([1, pc_points.shape[0]]).to(device)
    pc_points_homog = torch.cat([pc_points.t(), ones_concat], 0)

    # project pointcloud to image plane
    tinv = torch.inverse(transform)
    # tinv = geomtery_utils.inverse_transfom_3d(transform)
    transformed_pc_points = torch.mm(tinv[0:3, :], pc_points_homog)
    img_plane_points = torch.mm(K[:3, :3], transformed_pc_points)
    img_plane_points = img_plane_points / img_plane_points[2, :]
    img_plane_points = img_plane_points.t()  # [num_points 3]
    image_indices = torch.round(img_plane_points[:, :2]).long()  # [num_points 2]

    # find points projecting inside the frame
    is_inframe = (
        (img_plane_points[:, 0] >= 0)
        & (img_plane_points[:, 0] <= w - 1)
        & (img_plane_points[:, 1] >= 0)
        & (img_plane_points[:, 1] <= h - 1)
        & (transformed_pc_points[2, :] > 0)
    )

    # remove map points projecting to outside of the frame
    valid_points = pc_points[is_inframe]
    valid_normals = pc_normals[is_inframe]
    valid_ccounts = pc_ccounts[is_inframe]
    valid_colors = pc_colors[is_inframe]
    valid_indices = indices[is_inframe]
    valid_image_indices = image_indices[is_inframe]

    return (
        valid_points,
        valid_normals,
        valid_ccounts,
        valid_colors,
        valid_indices,
        valid_image_indices,
    )


def find_similar_points(
    valid_points: torch.Tensor,
    valid_normals: torch.Tensor,
    valid_img_inds: torch.Tensor,
    input_points: torch.Tensor,
    input_normals: torch.Tensor,
    dist2_th: float,
    dot_th: float,
):
    r"""Finds points from the global map that have a small distance and a
    similar normal to the new frame point which occupies the same frame pixel
    as their projection.

    Args:
        valid_points (torch.Tensor): Vertices of global map's pointcloud (in 
                                     global coords) which project onto the 
                                     (height x width) camera frame.
                                     (shape: P x 3)
        valid_normals (torch.Tensor): Normals of points from the global
                                      pointcloud (in global coords) that
                                      project onto the (height x width)
                                      camera frame. (shape: P x 3)
        valid_img_inds (torch.Tensor): Pixel positions of where each
                                       point from pointcloud would be
                                       in rasterized frame. (shape: P x 2)
        input_points (torch.Tensor): Vertices of the new frame in global 
                                     coords (shape: height x width x 3)
        input_normals (torch.Tensor): Normals of new frame's points in 
                                      global coords (shape: height x width x 3)
        dist2_th (float): Squared distance threshold.
        dot_th (float): Cosine similarity threshold.

    Returns:
        mask (torch.Tensor): 1D tensor of booleans (shape: P)
    """
    frame_points = input_points[valid_img_inds[:, 1], valid_img_inds[:, 0]]
    frame_normals = input_normals[valid_img_inds[:, 1], valid_img_inds[:, 0]]
    is_close = are_points_close(frame_points, valid_points, dist2_th)
    is_similar = are_normals_similar(frame_normals, valid_normals, dot_th)
    mask = is_close & is_similar
    return mask


def are_points_close(frame_points: torch.Tensor, map_points: torch.Tensor, dist2_th: float):
    r"""Computes 1D boolean tensor indicating closeness between corresponding
    new frame vertices and global map vertices.
    (See section 4.1 of PointFusion paper)

    Args:
        frame_points (torch.Tensor): Vertices of new frame in global coords.
                                     (shape: P x 3)
        map_points (torch.Tensor): Vertices of global map in global coords.
                                   (shape: P x 3)
        dist2_th (float): Squared distance threshold.

    Returns:
        is_close (torch.Tensor): 1D tensor of booleans (shape: P)
    """

    diff_pts = frame_points - map_points
    dist2_pts = torch.sum(diff_pts ** 2, 1)
    is_close = dist2_pts < dist2_th
    return is_close


def are_normals_similar(frame_normals: torch.Tensor, map_normals: torch.Tensor, dot_th: float):
    r"""Computes 1D boolean tensor indicating similarity between corresponding
    new frame's normals and global map's normals.
    (See section 4.1 of PointFusion paper)

    Args:
        frame_normals (torch.Tensor): Normals of new frame in global coords.
                                      (shape: P x 3)
        map_normals (torch.Tensor): Normals of global map in global coords.
                                    (shape: P x 3)
        dot_th (float): Cosine similarity threshold.

    Returns:
        is_similar (torch.Tensor): 1D tensor of booleans (shape: P)
    """
    normal_dots = torch.sum(frame_normals * map_normals, 1)
    is_similar = normal_dots > dot_th

    return is_similar


def find_best_unique_correspondences(
    valid_img_inds: torch.Tensor, valid_ccounts: torch.Tensor, valid_points: torch.Tensor,
):
    r"""Amongst global map points which project to the same frame pixel,
    find the ones which have the highest confidence counter (and if 
    confidence counter is equal then find the closest one to viewing ray).

    Args:
        valid_img_inds (torch.Tensor): Pixel positions of where each
                                       point from pointcloud would be
                                       in rasterized frame. (shape: P x 2)
        valid_ccounts (torch.Tensor): Confidence counter of points from the 
                                      global pointcloud that project onto the
                                      (height x width) camera frame.
                                      (shape: P)
        valid_points (torch.Tensor): Vertices of global map's pointcloud (in 
                                     global coords) which project onto the 
                                     (height x width) camera frame.
                                     (shape: P x 3)

    Returns:
        max_ccount_inds (torch.Tensor): 1D tensor of booleans (shape: P)
    """
    device = valid_points.device

    # argsort s.t. duplicate indices end next to each other, s.t. first
    # duplicate has higher ccount (& if ccount equal -> smaller ray dist first)
    inv_ccounts = (1 / valid_ccounts).unsqueeze(1)  # shape: [P 1]
    ray_dists = torch.sum(valid_points ** 2, 1).unsqueeze(1)
    inds = torch.arange(valid_points.shape[0]).float().unsqueeze(1).to(device)
    unique_criteria = [valid_img_inds.float(), inv_ccounts, ray_dists, inds]
    unique_criteria = torch.cat(unique_criteria, 1).to(device)
    sorted_criteria = torch.unique(unique_criteria, dim=0)
    indices = sorted_criteria[:, -1].long()
    # used torch.unique to sort rows (rows are unique): works as if we stable
    # sorted rows ascendingly based on every column going from right to left.
    # TODO: rough idea for (probably) faster way:
    # argsort(1e10*valid_img_inds[:,0] + 1e8*valid_img_inds[:,1] + 1e6*inv_ccounts + ray_dists)

    # find indices of the first occurrences of (sorted) valid_img_inds
    sorted_nonunique_inds = sorted_criteria[:, :2]  # shape: [P 2]
    pad = torch.nn.ConstantPad2d((0, 0, 1, 1), value=-1)
    unique_padded = pad(sorted_nonunique_inds)
    unique_padded = unique_padded.t().unsqueeze(0)  # shape: [1 2 (P+2)]
    weights = torch.tensor([[-1.0, 1.0, 0.0]]).view(1, 1, 3)
    weights = weights.repeat(1, 2, 1).to(device)  # shape: [1, 2, 3]
    first_unique_mask = F.conv1d(unique_padded, weights) != 0
    first_unique_mask = first_unique_mask.squeeze(0).squeeze(0)  # shape: [P]

    max_ccount_inds = indices[first_unique_mask]  # shape: [<=P]

    return max_ccount_inds


def get_alpha(
    points: torch.Tensor, sigma: Union[int, float, torch.Tensor], dim: int = 2, eps: float = 1e-7,
):
    r"""Computes sample confidence alpha.
    (See section 4.1 of PointFusion paper)

    Args:
        points (torch.Tensor): Tensor can be of any shape, as long as the
                               "dim"-th dimension contains (x, y, z) of 
                               the vertices.
        sigma (int or float or tensor): The width of the gaussian bell.
                                        Original paper uses constant 0.6
                                        emperically.
        dim (int, optional): Dimension along which (x, y, z) of vertices is 
                             stored. Defaults to 2.
        eps (float, optional): Minimum value for alpha (to avoid numerical
                               instability)

    Returns:
        alpha (torch.Tensor): Sample confidence. Same shape as input points
                              without the "dim"-th dimension.
    """
    alpha = torch.exp(-torch.sum(points ** 2, dim) / (2 * (sigma ** 2)))
    alpha = torch.clamp(alpha, min=eps, max=1.01)  # make sure alpha is non-zero

    return alpha
