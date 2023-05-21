import warnings
from typing import Optional, Union

import torch
import torch.nn as nn
from kornia.geometry.linalg import compose_transformations

from ..odometry.gradicp import GradICPOdometryProvider
from ..odometry.icp import ICPOdometryProvider
from ..odometry.icputils import downsample_pointclouds, downsample_rgbdimages
from ..structures.pointclouds import Pointclouds
from ..structures.rgbdimages import RGBDImages
from .fusionutils import find_active_map_points, update_map_aggregate

__all__ = ["ICPSLAM"]


class ICPSLAM(nn.Module):
    r"""ICP-SLAM for batched sequences of RGB-D images.

    Args:
        odom (str): Odometry method to be used from {'gt', 'icp', 'gradicp'}. Default: 'gradicp'
        dsratio (int): Downsampling ratio to apply to input frames before ICP. Only used if `odom` is
            'icp' or 'gradicp'. Default: 4
        numiters (int): Number of iterations to run the optimization for. Only used if `odom` is
            'icp' or 'gradicp'. Default: 20
        damp (float or torch.Tensor): Damping coefficient for nonlinear least-squares. Only used if `odom` is
            'icp' or 'gradicp'. Default: 1e-8
        dist_thresh (float or int or None): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
                Only used if `odom` is 'icp' or 'gradicp'. Default: None
        lambda_max (float or int): Maximum value the damping function can assume (`lambda_min` will be
            :math:`\frac{1}{\text{lambda_max}}`). Only used if `odom` is 'gradicp'.
        B (float or int): gradLM falloff control parameter (see GradICPOdometryProvider description).
            Only used if `odom` is 'gradicp'.
        B2 (float or int): gradLM control parameter (see GradICPOdometryProvider description).
            Only used if `odom` is 'gradicp'.
        nu (float or int): gradLM control parameter (see GradICPOdometryProvider description).
            Only used if `odom` is 'gradicp'.
        device (torch.device or str or None): The desired device of internal tensors. If None, sets device to be
            the CPU. Default: None


    Examples::

        >>> rgbdimages = RGBDImages(colors, depths, intrinsics, poses)
        >>> slam = ICPSLAM(odom='gt')
        >>> pointclouds, poses = slam(rgbdimages)
        >>> o3d.visualization.draw_geometries([pointclouds.o3d(0)])

        >>> rgbdimages = RGBDImages(colors, depths, intrinsics, poses)
        >>> slam = ICPSLAM(odom='gt')
        >>> pointclouds = Pointclouds()
        >>> pointclouds, new_poses = self.step(pointclouds, frames[:, 0], None)
        >>> frames.poses[:, :1] = new_poses
        >>> pointclouds, new_poses = self.step(pointclouds, frames[:, 1], frames[:, 0])

        >>> rgbdimages = RGBDImages(colors, depths, intrinsics, poses)
        >>> slam = ICPSLAM(odom='gradicp')
        >>> pointclouds = Pointclouds()
        >>> pointclouds, new_poses = self.step(pointclouds, frames[:, 0], None)
        >>> frames.poses[:, :1] = new_poses
        >>> pointclouds, new_poses = self.step(pointclouds, frames[:, 1], frames[:, 0])
    """
    # TODO: Try to have nn.Module features supported
    def __init__(
        self,
        *,
        odom: str = "gradicp",
        dsratio: int = 4,
        numiters: int = 20,
        damp: float = 1e-8,
        dist_thresh: Union[float, int, None] = None,
        lambda_max: Union[float, int] = 2.0,
        B: Union[float, int] = 1.0,
        B2: Union[float, int] = 1.0,
        nu: Union[float, int] = 200.0,
        device: Union[torch.device, str, None] = None,
        use_embeddings: bool = False,  # KM
    ):
        super().__init__()
        if odom not in ["gt", "icp", "gradicp"]:
            msg = "odometry method ({}) not supported for PointFusion. ".format(odom)
            msg += "Currently supported odometry modules for PointFusion are: 'gt', 'icp', 'gradicp'"
            raise ValueError(msg)

        odomprov = None
        if odom == "icp":
            odomprov = ICPOdometryProvider(numiters, damp, dist_thresh)
        elif odom == "gradicp":
            odomprov = GradICPOdometryProvider(
                numiters, damp, dist_thresh, lambda_max, B, B2, nu
            )

        self.odom = odom
        self.odomprov = odomprov
        self.dsratio = dsratio
        device = torch.device(device) if device is not None else torch.device("cpu")
        self.device = torch.Tensor().to(device).device

        self.use_embeddings = use_embeddings  # KM

    def forward(self, frames: RGBDImages):
        r"""Builds global map pointclouds from a batch of input RGBDImages with a batch size
        of :math:`B` and sequence length of :math:`L`.

        Args:
            frames (gradslam.RGBDImages): Input batch of frames with a sequence length of `L`.

        Returns:
            tuple: tuple containing:

            - pointclouds (gradslam.Pointclouds): Pointclouds object containing :math:`B` global maps
            - poses (torch.Tensor): Poses computed by the odometry method

        Shape:
            - poses: :math:`(B, L, 4, 4)`
        """
        if not isinstance(frames, RGBDImages):
            raise TypeError(
                "Expected frames to be of type gradslam.RGBDImages. Got {0}.".format(
                    type(frames)
                )
            )
        pointclouds = Pointclouds(device=self.device)
        batch_size, seq_len = frames.shape[:2]
        recovered_poses = torch.empty(batch_size, seq_len, 4, 4).to(self.device)
        prev_frame = None
        for s in range(seq_len):
            live_frame = frames[:, s].to(self.device)
            if s == 0 and live_frame.poses is None:
                live_frame.poses = (
                    torch.eye(4, dtype=torch.float, device=self.device)
                    .view(1, 1, 4, 4)
                    .repeat(batch_size, 1, 1, 1)
                )
            pointclouds, live_frame.poses = self.step(
                pointclouds, live_frame, prev_frame, inplace=True
            )
            prev_frame = live_frame if self.odom != "gt" else None
            recovered_poses[:, s] = live_frame.poses[:, 0]
        return pointclouds, recovered_poses

    def step(
        self,
        pointclouds: Pointclouds,
        live_frame: RGBDImages,
        prev_frame: Optional[RGBDImages] = None,
        inplace: bool = False,
    ):
        r"""Updates global map pointclouds with a SLAM step on `live_frame`.
        If `prev_frame` is not None, computes the relative transformation between `live_frame`
        and `prev_frame` using the selected odometry provider. If `prev_frame` is None,
        use the pose from `live_frame`.

        Args:
            pointclouds (gradslam.Pointclouds): Input batch of pointcloud global maps
            live_frame (gradslam.RGBDImages): Input batch of live frames (at time step :math:`t`). Must have sequence
                length of 1.
            prev_frame (gradslam.RGBDImages or None): Input batch of previous frames (at time step :math:`t-1`).
                Must have sequence length of 1. If None, will (skip calling odometry provider and) use the pose
                from `live_frame`. Default: None
            inplace (bool): Can optionally update the pointclouds and live_frame poses in-place. Default: False

        Returns:
            tuple: tuple containing:

            - pointclouds (gradslam.Pointclouds): Updated :math:`B` global maps
            - poses (torch.Tensor): Poses for the live_frame batch

        Shape:
            - poses: :math:`(B, 1, 4, 4)`
        """
        if not isinstance(live_frame, RGBDImages):
            raise TypeError(
                "Expected live_frame to be of type gradslam.RGBDImages. Got {0}.".format(
                    type(live_frame)
                )
            )
        live_frame.poses = self._localize(pointclouds, live_frame, prev_frame)
        pointclouds = self._map(pointclouds, live_frame, inplace)
        return pointclouds, live_frame.poses

    def _localize(
        self, pointclouds: Pointclouds, live_frame: RGBDImages, prev_frame: RGBDImages
    ):
        r"""Compute the poses for `live_frame`. If `prev_frame` is not None, computes the relative
        transformation between `live_frame` and `prev_frame` using the selected odometry provider.
        If `prev_frame` is None, use the pose from `live_frame`.

        Args:
            pointclouds (gradslam.Pointclouds): Input batch of pointcloud global maps
            live_frame (gradslam.RGBDImages): Input batch of live frames (at time step :math:`t`). Must have sequence
                length of 1.
            prev_frame (gradslam.RGBDImages or None): Input batch of previous frames (at time step :math:`t-1`).
                Must have sequence length of 1. If None, will (skip calling odometry provider and) use the pose
                from `live_frame`. Default: None

        Returns:
            torch.Tensor: Poses for the live_frame batch

        Shape:
            - Output: :math:`(B, 1, 4, 4)`
        """
        if not isinstance(pointclouds, Pointclouds):
            raise TypeError(
                "Expected pointclouds to be of type gradslam.Pointclouds. Got {0}.".format(
                    type(pointclouds)
                )
            )
        if not isinstance(live_frame, RGBDImages):
            raise TypeError(
                "Expected live_frame to be of type gradslam.RGBDImages. Got {0}.".format(
                    type(live_frame)
                )
            )
        if not isinstance(prev_frame, (RGBDImages, type(None))):
            raise TypeError(
                "Expected prev_frame to be of type gradslam.RGBDImages or None. Got {0}.".format(
                    type(prev_frame)
                )
            )
        if prev_frame is not None:
            if self.odom == "gt":
                warnings.warn(
                    "`prev_frame` is not used when using `odom='gt'` (should be None)"
                )
            elif not prev_frame.has_poses:
                raise ValueError("`prev_frame` should have poses, but did not.")
        if prev_frame is None and pointclouds.has_points and self.odom != "gt":
            msg = "`prev_frame` was None despite `{}` odometry method. Using `live_frame` poses.".format(
                self.odom
            )
            warnings.warn(msg)
        if prev_frame is None or self.odom == "gt":
            if not live_frame.has_poses:
                raise ValueError(
                    "`live_frame` must have poses when `prev_frame` is None or `odom='gt'`."
                )
            return live_frame.poses

        if self.odom in ["icp", "gradicp"]:
            live_frame.poses = prev_frame.poses
            frames_pc = downsample_rgbdimages(live_frame, self.dsratio)
            pc2im_bnhw = find_active_map_points(pointclouds, prev_frame)
            maps_pc = downsample_pointclouds(pointclouds, pc2im_bnhw, self.dsratio)
            transform = self.odomprov.provide(maps_pc, frames_pc)

            return compose_transformations(
                transform.squeeze(1), prev_frame.poses.squeeze(1)
            ).unsqueeze(1)

    def _map(
        self, pointclouds: Pointclouds, live_frame: RGBDImages, inplace: bool = False
    ):
        r"""Updates global map pointclouds by aggregating them with points from `live_frame`.

        Args:
            pointclouds (gradslam.Pointclouds): Input batch of pointcloud global maps
            live_frame (gradslam.RGBDImages): Input batch of live frames (at time step :math:`t`). Must have sequence
                length of 1.
            inplace (bool): Can optionally update the pointclouds in-place. Default: False

        Returns:
            gradslam.Pointclouds: Updated :math:`B` global maps

        """
        return update_map_aggregate(pointclouds, live_frame, inplace)
