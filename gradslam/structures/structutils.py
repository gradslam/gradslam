# BSD License

# For PyTorch3d software

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

#  * Neither the name Facebook nor the names of its contributors may be used to
#    endorse or promote products derived from this software without specific
#    prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Util functions used in structures. `list_to_padded` and `padded_to_list` functions are borrowed from pytorch3d:
https://github.com/facebookresearch/pytorch3d
"""
import base64
import cv2
import numpy as np
import plotly.graph_objects as go
from typing import List, Union

import torch

__all__ = []


def list_to_padded(
    x: List[torch.Tensor],
    pad_size: Union[list, tuple, None] = None,
    pad_value: float = 0.0,
    equisized: bool = False,
) -> torch.Tensor:
    r"""Transforms a list of B tensors each of shape :math:`(N_b, C_b)` into a single tensor of shape
    :math:`(N, pad_size(0), pad_size(1))`, or :math:`(N, max(N_b), max(C_b))` if pad_size is None.

    Args:
        x: list of Tensors
        pad_size: list(int) specifying the size of the padded tensor
        pad_value: float value to be used to fill the padded tensor
        equisized: bool indicating whether the items in x are of equal size
        (sometimes this is known and if provided saves computation)

    Returns:
        x_padded: tensor consisting of padded input tensors
    """
    if equisized:
        return torch.stack(x, 0)

    if pad_size is None:
        pad_dim0 = max(y.shape[0] for y in x if len(y) > 0)
        pad_dim1 = max(y.shape[1] for y in x if len(y) > 0)
    else:
        if len(pad_size) != 2:
            raise ValueError("Pad size must contain target size for 1st and 2nd dim")
        pad_dim0, pad_dim1 = pad_size

    N = len(x)
    x_padded = torch.full(
        (N, pad_dim0, pad_dim1), pad_value, dtype=x[0].dtype, device=x[0].device
    )
    for i, y in enumerate(x):
        if len(y) > 0:
            if y.ndim != 2:
                raise ValueError("Supports only 2-dimensional tensor items")
            x_padded[i, : y.shape[0], : y.shape[1]] = y
    return x_padded


def padded_to_list(x: torch.Tensor, split_size: Union[list, tuple, None] = None):
    r"""Transforms a padded tensor of shape :math:`(B, N, C)` into a list of :math:`B` tensors of shape
    :math:`(N_b, C_b)` where :math:`(N_b, C_b)` is specified in split_size(b), or of shape :math:`(N, C)` if
    split_size is None. split_size support only for 3-dimensional input tensor.

    Args:
        x: tensor consisting of padded input tensors
        split_size: the shape of the final tensor to be returned (of length N).

    Returns:
        x_list: list of Tensors

    Shape:
        - x: :math:`(B, N, C)`
    """
    if x.ndim != 3:
        raise ValueError("Supports only 3-dimensional input tensors")
    x_list = list(x.unbind(0))
    if split_size is None:
        return x_list

    N = len(split_size)
    if x.shape[0] != N:
        raise ValueError("Split size must be of same length as inputs first dimension")

    for i in range(N):
        if isinstance(split_size[i], int):
            x_list[i] = x_list[i][: split_size[i]]
        elif len(split_size[i]) == 2:
            x_list[i] = x_list[i][: split_size[i][0], : split_size[i][1]]
        else:
            raise ValueError(
                "Support only for 2-dimensional unbinded tensor. \
                    Split size for more dimensions provided"
            )
    return x_list


def numpy_to_plotly_image(img, name=None, is_depth=False, scale=None, quality=95):
    r"""Converts a numpy array img to a `plotly.graph_objects.Image` object.

    Args
        img (np.ndarray): RGB image array
        name (str): Name for the returned `plotly.graph_objects.Image` object
        is_depth (bool): Bool indicating whether input `img` is depth image. Default: False
        scale (int or None): Scale factor to display on hover. If None, will not display `scale: ...`. Default: None
        quality (int): Image quality from 0 to 100 (the higher is the better). Default: 95

    Returns:
        `plotly.graph_objects.Image`
    """
    img_str = img_to_b64str(img, quality)
    hovertemplate = "x: %%{x}<br>y: %%{y}<br>%s: %s"
    if not is_depth:
        hover_name = "[%{z[0]}, %{z[1]}, %{z[2]}]"
        hovertemplate = hovertemplate % ("color", hover_name)
    else:
        hover_name = "%{z[0]}"
        hovertemplate = hovertemplate % ("depth", hover_name)
    if scale is not None:
        scale = int(scale) if int(scale) == scale else scale
        hovertemplate += f"<br>scale: x{scale}<br>"
    hovertemplate += "<extra></extra>"

    return go.Image(source=img_str, hovertemplate=hovertemplate, name=name)


def img_to_b64str(img, quality=95):
    r"""Converts a numpy array of uint8 into a base64 jpeg string.

    Args
        img (np.ndarray): RGB or greyscale image array
        quality (int): Image quality from 0 to 100 (the higher is the better). Default: 95

    Returns:
        str: base64 jpeg string
    """
    # Can also use px._imshow._array_to_b64str:
    # https://github.com/plotly/plotly.py/blob/63f20ee08d2b83075d3749ec5d85f7909401a0ef/packages/python/plotly/plotly/express/_imshow.py#L27
    if not isinstance(img, np.ndarray):
        raise TypeError(f"img must be of type np.ndarray, but was {type(img)}")
    if img.ndim != 2 and img.ndim != 3:
        raise ValueError(f"img.ndim must be 2 or 3, but was {img.ndim}")
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.ndim == 3 else img
    retval, buffer = cv2.imencode(".jpg", img, encode_param)
    imstr = base64.b64encode(buffer).decode("utf-8")
    prefix = "data:image/jpeg;base64,"
    base64_string = prefix + imstr
    return base64_string
