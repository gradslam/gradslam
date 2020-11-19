import torch
from ..geometry.geometryutils import relative_transformation

from ..structures.rgbdimages import RGBDImages
from .base import OdometryProvider

__all__ = ["GroundTruthOdometryProvider"]


class GroundTruthOdometryProvider(OdometryProvider):
    r"""Ground truth odometry provider. Computes the relative transformation between a pair of `gradslam.RGBDImages`
    objects. Both objects must contain `poses` attributes.
    """

    def provide(self, rgbdimages1: RGBDImages, rgbdimages2: RGBDImages) -> torch.Tensor:
        r"""Computes the relative homogenous transformation between poses of `rgbdimages2` and `rgbdimages1`.
        The relative transformation is computed as :math:`T = (T_1)^{-1} \cdot T_2`.

        Args:
            rgbdimages1 (gradslam.RGBDImages): Object containing batch of reference poses of shape :math:`(B, 1, 4, 4)`
            rgbdimages2 (gradslam.RGBDImages): Object containing batch of destination poses of shape
                :math:`(B, 1, 4, 4)`

        Returns:
            torch.Tensor: The relative transformation between the poses of `rgbdimages1` and `rgbdimages2`
            (:math:`T = (T_1)^{-1} \cdot T_2`).

        Shape:
            - Output: :math:`(B, 1, 4, 4)`
        """
        if not isinstance(rgbdimages1, RGBDImages):
            raise TypeError(
                "Expected input 1 (rgbdimages1) to be of type gradslam.RGBDImages. Got {0}.".format(
                    type(rgbdimages1)
                )
            )
        if not isinstance(rgbdimages2, RGBDImages):
            raise TypeError(
                "Expected input 2 (rgbdimages2) to be of type gradslam.RGBDImages. Got {0}.".format(
                    type(rgbdimages2)
                )
            )
        if rgbdimages1.poses is None:
            raise ValueError(
                "Input 1 (rgbdimages1) missing poses. Poses must be provided if using GroundTruthOdometryProvider"
            )
        if rgbdimages2.poses is None:
            raise ValueError(
                "Input 2 (rgbdimages2) missing poses. Poses must be provided if using GroundTruthOdometryProvider"
            )
        if rgbdimages1.shape[1] != 1:
            raise ValueError(
                "Sequence length of rgbdimages1 must be 1, but was {0}.".format(
                    rgbdimages1.shape[1]
                )
            )
        if rgbdimages2.shape[1] != 1:
            raise ValueError(
                "Sequence length of rgbdimages2 must be 1, but was {0}.".format(
                    rgbdimages2.shape[1]
                )
            )
        if rgbdimages1.shape[0] != rgbdimages2.shape[0]:
            raise ValueError(
                "Batch size of rgbdimages1 and rgbdimages2 should be equal ({0} != {1})".format(
                    rgbdimages1.shape[0], rgbdimages2.shape[0]
                )
            )

        B, L = rgbdimages1.shape[:2]
        return relative_transformation(
            rgbdimages1.poses.view(-1, 4, 4),
            rgbdimages2.poses.view(-1, 4, 4),
            orthogonal_rotations=False,
        ).view(B, L, 4, 4)
