from typing import Union

import torch

from ..structures.pointclouds import Pointclouds
from .base import OdometryProvider
from .icputils import point_to_plane_gradICP

__all__ = ["GradICPOdometryProvider"]


class GradICPOdometryProvider(OdometryProvider):
    r"""An odometry provider that uses the (differentiable) gradICP technique presented in the gradSLAM paper.
    Computes the relative transformation between a pair of `gradslam.Pointclouds` objects using GradICP which
    uses gradLM (:math:`\nabla LM`) solver (See gradLM section of 
    `the gradSLAM paper <https://arxiv.org/abs/1910.10672>`__). The iterate and damping coefficient are updated by:

    .. math::
        lambda_1 = Q_\lambda(r_0, r_1) & = \lambda_{min} + \frac{\lambda_{max} -
        \lambda_{min}}{1 + e^{-B (r_1 - r_0)}} \\
        Q_x(r_0, r_1) & = x_0 + \frac{\delta x_0}{\sqrt[nu]{1 + e^{-B2*(r_1 - r_0)}}}`

    """

    def __init__(
        self,
        numiters: int = 20,
        damp: float = 1e-8,
        dist_thresh: Union[float, int, None] = None,
        lambda_max: Union[float, int] = 2.0,
        B: Union[float, int] = 1.0,
        B2: Union[float, int] = 1.0,
        nu: Union[float, int] = 200.0,
    ):
        r"""Initializes internal GradICPOdometryProvider state.

        Args:
            numiters (int): Number of iterations to run the optimization for. Default: 20
            damp (float or torch.Tensor): Damping coefficient for nonlinear least-squares. Default: 1e-8
            dist_thresh (float or int or None): Distance threshold for removing `src_pc` points distant from `tgt_pc`.
                Default: None
            lambda_max (float or int): Maximum value the damping function can assume (`lambda_min` will be
                :math:`\frac{1}{\text{lambda_max}}`)
            B (float or int): gradLM falloff control parameter (see GradICPOdometryProvider description)
            B2 (float or int): gradLM control parameter (see GradICPOdometryProvider description)
            nu (float or int): gradLM control parameter (see GradICPOdometryProvider description)

        """
        self.numiters = numiters
        self.damp = damp
        self.dist_thresh = dist_thresh
        self.lambda_max = lambda_max
        self.B = B
        self.B2 = B2
        self.nu = nu

    def provide(
        self,
        maps_pointclouds: Pointclouds,
        frames_pointclouds: Pointclouds,
    ) -> torch.Tensor:
        r"""Uses gradICP to compute the relative homogenous transformation that, when applied to `frames_pointclouds`,
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
                "maps_pointclouds missing normals. Map normals must be provided if using GradICPOdometryProvider"
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
            transform, _ = point_to_plane_gradICP(
                frames_pointclouds.points_list[b].unsqueeze(0),
                maps_pointclouds.points_list[b].unsqueeze(0),
                maps_pointclouds.normals_list[b].unsqueeze(0),
                initial_transform,
                numiters=self.numiters,
                damp=self.damp,
                dist_thresh=self.dist_thresh,
                lambda_max=self.lambda_max,
                B=self.B,
                B2=self.B2,
                nu=self.nu,
            )

            transforms.append(transform)

        return torch.stack(transforms).unsqueeze(1)
