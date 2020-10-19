import math
from typing import Union

import torch
import torch.nn as nn
from kornia.geometry.linalg import compose_transformations

from ..odometry.base import OdometryProvider
from ..odometry.groundtruth import GroundTruthOdometryProvider
from ..odometry.icp import ICPOdometryProvider
from ..odometry.gradicp import GradICPOdometryProvider
from ..odometry.icputils import downsample_pointclouds, downsample_rgbdimages
from ..structures.pointclouds import Pointclouds
from ..structures.rgbdimages import RGBDImages
from .fusionutils import find_active_map_points, update_map_fusion

__all__ = ["PointFusion"]
POINTFUSION_ODOMS = [GroundTruthOdometryProvider, ICPOdometryProvider, GradICPOdometryProvider]

class PointFusion(nn.Module):
    r"""Performs Point-based Fusion (PointFusion for short) on a batched sequence of RGB-D images
    given the odometry provider.
    (Point-based Fusion paper: http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf )
    """
    # TODO: Try to have nn.Module features supported
    def __init__(
        self,
        odom,
        dist_th: Union[float, int] = 0.05,
        angle_th: Union[float, int] = 20,
        sigma: Union[float, int] = 0.6,
    ):
        r"""Initializes Point-based fusion.

        Args:
            odom (gradslam.OdomteryProvider): Odometry provider to be used.
            dist_th_sq (float or int): Squared distance threshold.
            dot_th (float or int): Dot product threshold.
            sigma (torch.Tensor or float or int): Width of the gaussian bell. Original paper uses 0.6 emperically.
        """
        super(PointFusion, self).__init__()

        if not isinstance(odom, OdometryProvider):
            raise TypeError(
                "odometry module (odom) should be subclass of OdometryProvider, but was not ({}).".format(
                    type(odom)
                )
            )
        if not any([isinstance(odom, odom_class) for odom_class in POINTFUSION_ODOMS]):
            msg = 'Provided odometry module "{}" not supported for PointFusion. '.format(odom.__class__.__name__)
            msg += "Currently supported odometry modules for PointFusion are: "
            msg += ", ".join(['"' + odom.__name__ + '"' for odom in POINTFUSION_ODOMS])
            raise NotImplementedError(msg)

        if not (isinstance(dist_th, float) or isinstance(dist_th, int)):
            raise TypeError(
                "Distance threshold must be of type float, or int; but was of type {}.".format(
                    type(dist_th)
                )
            )
        if not (isinstance(angle_th, float) or isinstance(angle_th, int)):
            raise TypeError(
                "Angle threshold must be of type float; but was of type {}.".format(
                    type(angle_th)
                )
            )

        if torch.is_tensor(dist_th) and not (dist_th.ndim == 0):
            raise ValueError(
                "Distance threshold must be scalar but had ndim ({}).".format(
                    dist_th.ndim
                )
            )
        if torch.is_tensor(angle_th) and not (angle_th.ndim == 0):
            raise ValueError(
                "Angle threshold must be scalar but had ndim ({}).".format(
                    angle_th.ndim
                )
            )

        # can remove below checks, just ensures user knows what's happening
        if dist_th < 0:
            raise ValueError(
                "Distance threshold ({}) must be non-negative.".format(dist_th)
            )
        if not ((0 <= angle_th) and (angle_th <= 90)):
            raise ValueError(
                "Angle threshold ({}) must be non-negative and <=90.".format(angle_th)
            )

        self.odom = odom
        self.dist_th = dist_th
        rad_th = (angle_th * math.pi) / 180
        self.dot_th = torch.cos(rad_th) if torch.is_tensor(rad_th) else math.cos(rad_th)
        self.sigma = sigma

    def forward(
        self, pointclouds: Pointclouds, live_frame: RGBDImages, prev_frame: RGBDImages
    ):
        r"""Updates global map pointclouds using PointFusion based on the live frame and the previous frame.

        Args:
            pointclouds (gradslam.Pointclouds): Input batch of sequences of pointcloud global maps
            live_frame (gradslam.RGBDImages): Input batch of live frames (at time step :math:`t`). Must have sequence
                length of 1.
            prev_frame (gradslam.RGBDImages): Input batch of previous frames (at time step :math:`t-1`). Must have
                sequence length of 1.

        Returns:
            pointclouds (gradslam.Pointclouds): Pointclouds object containing :math:`B` global maps
            poses (torch.Tensor): Poses for the live_frame batch

        Shape:
            poses: :math:`(B, 1, 4, 4)`
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
        if not isinstance(prev_frame, RGBDImages):
            raise TypeError(
                "Expected prev_frame to be of type gradslam.RGBDImages. Got {0}.".format(
                    type(prev_frame)
                )
            )
        if isinstance(self.odom, GroundTruthOdometryProvider):
            transform = self.odom.provide(prev_frame, live_frame)
        elif isinstance(self.odom, ICPOdometryProvider) or isinstance(
            self.odom, GradICPOdometryProvider
        ):
            live_frame.poses = prev_frame.poses
            frames_pc = downsample_rgbdimages(live_frame, self.odom.downsample_ratio)
            pc2im_bnhw = find_active_map_points(pointclouds, prev_frame)
            maps_pc = downsample_pointclouds(
                pointclouds, pc2im_bnhw, self.odom.downsample_ratio
            )
            transform = self.odom.provide(maps_pc, frames_pc)

        live_frame.poses[:, 0] = compose_transformations(
            transform.squeeze(1), prev_frame.poses.squeeze(1)
        )
        update_map_fusion(
            pointclouds, live_frame, self.dist_th, self.dot_th, self.sigma
        )

        return pointclouds, live_frame.poses
