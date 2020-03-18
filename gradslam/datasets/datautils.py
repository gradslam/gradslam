import copy
import logging
from collections import OrderedDict
from typing import List, Union

import numpy as np
import torch

__all__ = [
    "normalize_image",
    "channels_first",
    "scale_intrinsics",
    "poses_to_transforms",
    "create_label_image",
]


def normalize_image(rgb: Union[torch.Tensor, np.ndarray]):
    r"""Normalizes RGB image values from :math:`[0 255]` range to :math:`[0, 1]` range.

    Args:
        rgb (torch.Tensor or np.ndarray): RGB image in range :math:`[0, 255]`

    Returns:
        torch.Tensor or np.ndarray: Normalized RGB image in range :math:`[0, 1]`

    Shape:
        - rgb: Any shape
        - Output: Same shape as input
    """
    if torch.is_tensor(rgb):
        return rgb.float() / 255
    elif isinstance(rgb, np.ndarray):
        return rgb.astype(float) / 255
    else:
        raise TypeError("Unsupported input rgb type: %r" % type(rgb))


def channels_first(rgb: Union[torch.Tensor, np.ndarray]):
    r"""Converts from channels last representation :math:`(*, H, W, C)` to channels first representation
        :math:`(*, C, H, W)`

        Args:
            rgb (torch.Tensor or np.ndarray): :math:`(*, H, W, C)` ordering `(*, height, width, channels)`

        Returns:
            torch.Tensor or np.ndarray: :math:`(*, C, H, W)` ordering

        Shape:
            - rgb: :math:`(*, H, W, C)`
            - Output: :math:`(*, C, H, W)`
        """
    if not (isinstance(rgb, np.ndarray) or torch.is_tensor(rgb)):
        raise TypeError("Unsupported input rgb type {}".format(type(rgb)))

    if rgb.ndim < 3:
        raise ValueError(
            "Input rgb must contain atleast 3 dims, but had {} dims.".format(rgb.ndim)
        )
    if rgb.shape[-3] < rgb.shape[-1]:
        msg = "Are you sure that the input is correct? Number of channels exceeds height of image: %r > %r"
        logging.warning(msg % (rgb.shape[-3], rgb.shape[-1]))
    ordering = list(range(rgb.ndim))
    ordering[-2], ordering[-1], ordering[-3] = ordering[-3], ordering[-2], ordering[-1]

    if isinstance(rgb, np.ndarray):
        return np.ascontiguousarray(rgb.transpose(*ordering))
    elif torch.is_tensor(rgb):
        return rgb.permute(*ordering).contiguous()


def scale_intrinsics(
    intrinsics: Union[np.ndarray, torch.Tensor],
    h_ratio: Union[float, int],
    w_ratio: Union[float, int],
):
    r"""Scales the intrinsics appropriately for resized frames where 
     and :math:`w_ratio = w_\text{new} / w_\text{old}` 

    Args:
        intrinsics (np.ndarray or torch.Tensor): Intrinsics matrix of original frame
        h_ratio (float or int): Ratio of new frame's height to old frame's height
            :math:`h_ratio = h_\text{new} / h_\text{old}`
        w_ratio (float or int): Ratio of new frame's width to old frame's width
            :math:`w_ratio = w_\text{new} / w_\text{old}`

    Returns:
        scaled_intrinsics (np.ndarray or torch.Tensor): Intrinsics matrix scaled approprately for new frame size

    Shape:
        - intrinsics: :math:`(*, 3, 3)` or :math:`(*, 4, 4)`
        - scaled_intrinsics: Matches `intrinsics` shape, :math:`(*, 3, 3)` or :math:`(*, 4, 4)`

    """
    if isinstance(intrinsics, np.ndarray):
        scaled_intrinsics = intrinsics.astype(float).copy()
    elif torch.is_tensor(intrinsics):
        scaled_intrinsics = intrinsics.to(torch.float).clone()
    else:
        raise TypeError("Unsupported input rgb type {}".format(type(rgb)))
    if not (intrinsics.shape[-2:] == (3, 3) or intrinsics.shape[-2:] == (4, 4)):
        raise ValueError(
            "intrinsics should have shape (*, 3, 3) or (*, 4, 4), but had shape {} instead".format(
                intrinsics.shape
            )
        )
    if (intrinsics[..., -1, -1] != 1).any():
        logging.warn("Incorrect intrinsics: intrinsics[..., -1, -1] should be 1.")

    scaled_intrinsics[..., 0, 0] *= w_ratio  # fx
    scaled_intrinsics[..., 1, 1] *= h_ratio  # fy
    scaled_intrinsics[..., 0, 2] *= w_ratio  # cx
    scaled_intrinsics[..., 1, 2] *= h_ratio  # cy
    return scaled_intrinsics


def poses_to_transforms(poses: List[np.ndarray]):
    r"""Converts poses to transformations w.r.t. the first frame in the sequence having identity pose

    Args:
        poses (list of np.ndarray): List of ground truth poses in `np.ndarray` format.

    Returns:
        transformations (list of np.ndarray): List of ground truth frame to frame transformations where initial
            frame is transformed to have identity pose.

    Shape:
        - poses: List of `np.ndarray` homogeneous poses, each of shape :math:`[4, 4]`
        - transformations:  List of `np.ndarray` homogeneous transformations, each of shape :math:`[4, 4]`
    """
    transformations = copy.deepcopy(poses)
    for i in range(len(poses)):
        if i == 0:
            transformations[i] = np.eye(4)
        else:
            transformations[i] = np.linalg.inv(poses[i - 1]).dot(poses[i])
    return transformations


def create_label_image(prediction: np.ndarray, color_palette: OrderedDict):
    r"""Creates a label image, given a network prediction (each pixel contains class index) and a color palette.

    Args:
        prediction (np.ndarray): Output image. Each pixel contains an integer, corresponding to its class label.
        color_palette (OrderedDict: Contains :math:`(R, G, B)` colors (`uint8`) for each class.

    Returns:
        Output (np.ndarray): Label image with the given color palette

    Shape:
        - prediction: :math:`(H, W)`
        - Output: :math:`(H, W)`
    """

    label_image = np.zeros(
        (prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8
    )
    for idx, color in enumerate(color_palette):
        label_image[prediction == idx] = color
    return label_image
