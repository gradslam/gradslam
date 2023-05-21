import math
import warnings
from typing import Union

import torch

from ..structures.pointclouds import Pointclouds
from ..structures.rgbdimages import RGBDImages
from .fusionutils import update_map_fusion
from .icpslam import ICPSLAM

__all__ = ["PointFusion"]


class PointFusion(ICPSLAM):
    r"""Point-based Fusion (PointFusion for short) SLAM for batched sequences of RGB-D images
    (See Point-based Fusion `paper <http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.pdf>`__).

    Args:
        odom (str): Odometry method to be used from {'gt', 'icp', 'gradicp'}. Default: 'gradicp'
        dist_th (float or int): Distance threshold.
        dot_th (float or int): Dot product threshold.
        sigma (torch.Tensor or float or int): Width of the gaussian bell. Original paper uses 0.6 emperically.
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
    >>> slam = PointFusion(odom='gt')
    >>> pointclouds, poses = slam(rgbdimages)
    >>> o3d.visualization.draw_geometries([pointclouds.o3d(0)])
    """

    def __init__(
        self,
        *,
        odom: str = "gradicp",
        dist_th: Union[float, int] = 0.05,
        angle_th: Union[float, int] = 20,
        sigma: Union[float, int] = 0.6,
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
        embedding_fusion_method: str = "slam",
    ):
        super().__init__(
            odom=odom,
            dsratio=dsratio,
            numiters=numiters,
            damp=damp,
            dist_thresh=dist_thresh,
            lambda_max=lambda_max,
            B=B,
            B2=B2,
            nu=nu,
            device=device,
            use_embeddings=use_embeddings,  # KM
        )
        if not (isinstance(dist_th, float) or isinstance(dist_th, int)):
            raise TypeError(
                "Distance threshold must be of type float or int; but was of type {}.".format(
                    type(dist_th)
                )
            )
        if not (isinstance(angle_th, float) or isinstance(angle_th, int)):
            raise TypeError(
                "Angle threshold must be of type float or int; but was of type {}.".format(
                    type(angle_th)
                )
            )
        if dist_th < 0:
            warnings.warn(
                "Distance threshold ({}) should be non-negative.".format(dist_th)
            )
        if not ((0 <= angle_th) and (angle_th <= 90)):
            warnings.warn(
                "Angle threshold ({}) should be non-negative and <=90.".format(angle_th)
            )
        self.dist_th = dist_th
        rad_th = (angle_th * math.pi) / 180
        self.dot_th = torch.cos(rad_th) if torch.is_tensor(rad_th) else math.cos(rad_th)
        self.sigma = sigma

        self.use_embeddings = use_embeddings  # KM
        self.embedding_fusion_method = embedding_fusion_method

    def _map(
        self, pointclouds: Pointclouds, live_frame: RGBDImages, inplace: bool = False
    ):
        return update_map_fusion(
            pointclouds,
            live_frame,
            self.dist_th,
            self.dot_th,
            self.sigma,
            inplace,
            use_embeddings=self.use_embeddings,  # KM
            embedding_fusion_method=self.embedding_fusion_method,
        )
