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
Util functions containing representation transforms for points/normals/colors/features.
"""

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
