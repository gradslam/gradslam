from typing import Union

import torch

from ..structures.pointclouds import Pointclouds
from .base import OdometryProvider
from .icputils import point_to_plane_ICP

__all__ = ["ICPOdometryProvider"]


class ICPOdometryProvider(OdometryProvider):
    r"""ICP odometry provider using a point-to-plane error metric. Computes the relative transformation between
    a pair of `gradslam.Pointclouds` objects using ICP (Iterative Closest Point). Uses LM (Levenberg-Marquardt) solver.
    """

    def __init__(
        self,
        numiters: int = 20,
        damp: float = 1e-8,
        dist_thresh: Union[float, int, None] = None,
    ):
        r"""Initializes internal ICPOdometryProvider state.

        Args:
            numiters (int): Number of iterations to run the optimization for. Default: 20
            damp (float or torch.Tensor): Damping coefficient for nonlinear least-squares. Default: 1e-8
            dist_thresh (float or int or None): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
                Default: None

        """
        self.numiters = numiters
        self.damp = damp
        self.dist_thresh = dist_thresh

    def provide(
        self,
        maps_pointclouds: Pointclouds,
        frames_pointclouds: Pointclouds,
    ) -> torch.Tensor:
        r"""Uses ICP to compute the relative homogenous transformation that, when applied to `frames_pointclouds`,
        would cause the points to align with points of `maps_pointclouds`.

        Args:
            maps_pointclouds (gradslam.Pointclouds): Object containing batch of map pointclouds of batch size
                :math:`(B)`
            frames_pointclouds (gradslam.Pointclouds): Object containing batch of live frame pointclouds of batch size
                :math:`(B)`

        Returns:
            torch.Tensor: The relative transformation that would align `maps_pointclouds` with `frames_pointclouds`

        Shape:
            - Output: :math:`(B, 1, 4, 4)`

        """
        if not isinstance(maps_pointclouds, Pointclouds):
            raise TypeError(
                "Expected maps_pointclouds to be of type gradslam.Pointclouds. Got {0}.".format(
                    type(maps_pointclouds)
                )
            )
        if not isinstance(frames_pointclouds, Pointclouds):
            raise TypeError(
                "Expected frames_pointclouds to be of type gradslam.Pointclouds. Got {0}.".format(
                    type(frames_pointclouds)
                )
            )
        if maps_pointclouds.normals_list is None:
            raise ValueError(
                "maps_pointclouds missing normals. Map normals must be provided if using ICPOdometryProvider"
            )
        if len(maps_pointclouds) != len(frames_pointclouds):
            raise ValueError(
                "Batch size of maps_pointclouds and frames_pointclouds should be equal ({0} != {1})".format(
                    len(maps_pointclouds), len(frames_pointclouds)
                )
            )

        device = maps_pointclouds.device
        initial_transform = torch.eye(4, device=device)

        transforms = []
        for b in range(len(maps_pointclouds)):
            transform, _ = point_to_plane_ICP(
                frames_pointclouds.points_list[b].unsqueeze(0),
                maps_pointclouds.points_list[b].unsqueeze(0),
                maps_pointclouds.normals_list[b].unsqueeze(0),
                initial_transform,
                numiters=self.numiters,
                damp=self.damp,
                dist_thresh=self.dist_thresh,
            )

            transforms.append(transform)

        return torch.stack(transforms).unsqueeze(1)
