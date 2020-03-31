import math
from typing import Union

import torch
import torch.nn as nn
from kornia.geometry.linalg import compose_transformations

from ..odometry.base import OdometryProvider
from ..structures.rgbdimages import RGBDImages
from .fusionutils import rgbdimages_to_pointclouds, update_map_fusion

__all__ = ["PointFusion"]


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

    def forward(self, rgb_batch, depth_batch, intrinsics_batch, poses_batch=None):
        r"""Builds global map pointclouds using PointFusion from input RGB-D images, intrinsics and poses.

        Args:
            rgb_batch (torch.Tensor): Input batch of sequences of rgb images
            depth_batch (torch.Tensor): Input batch of sequences of depth images
            intrinsics_batch (torch.Tensor): Input batch of intrinsics
            poses_batch (torch.Tensor): Optional input batch of sequences of poses. It is only used if using
                `GroundTruthOdometryProvider`. Default: None

        Returns:
            pointclouds (gradslam.Pointclouds): Pointclouds object containing :math:`B` global maps

        Shape:
            rgb_batch: :math:`(B, L, H, W, 3)`
            depth_batch: :math:`(B, L, H, W, 1)`
            intrinsics_batch: :math:`(B, 1, 4, 4)`
            poses_batch: :math:`(B, L, 4, 4)`
        """
        # TODO: Support channels first representation
        batch_size, seq_len = rgb_batch.shape[:2]
        rgbdimages = RGBDImages(
            rgb_batch, depth_batch, intrinsics_batch, poses_batch, channels_first=False
        )
        pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma=self.sigma)

        for s in range(1, seq_len):
            transform = self.odom.provide(rgbdimages[:, s - 1], rgbdimages[:, s])
            rgbdimages[:, s].poses = compose_transformations(
                transform.squeeze(1), rgbdimages[:, s - 1].poses.squeeze(1)
            )
            update_map_fusion(
                pointclouds, rgbdimages[:, s], self.dist_th, self.dot_th, self.sigma
            )

        return pointclouds
