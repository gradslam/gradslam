from typing import Optional, Union

from plotly.subplots import make_subplots
import torch

from .structutils import numpy_to_plotly_image
from ..geometry.geometryutils import create_meshgrid
from ..geometry.projutils import inverse_intrinsics

__all__ = ["RGBDImages"]


class RGBDImages(object):
    r"""Initializes an RGBDImage object consisting of a batch of a sequence of rgb images, depth maps,
    camera intrinsics, and (optionally) poses.

    Args:
        rgb_image (torch.Tensor): 3-channel rgb image
        depth_image (torch.Tensor): 1-channel depth map
        intrinsics (torch.Tensor): camera intrinsics
        poses (torch.Tensor or None): camera extrinsics. Default: None
        channels_first(bool): indicates whether `rgb_image` and `depth_image` have channels first or channels last
            representation (i.e. rgb_image.shape is :math:`(B, L, H, W, 3)` or :math:`(B, L, 3, H, W)`.
            Default: False
        device (torch.device or str or None): The desired device of internal tensors. If None, sets device to be
            same as `rgb_image` device. Default: None
        pixel_pos (torch.Tensor or None): Similar to meshgrid but with extra channel of 1s at the end. If provided,
            can save computations when computing vertex maps. Default: None

    Shape:
        - rgb_image: :math:`(B, L, H, W, 3)` if `channels_first` is False, else :math:`(B, L, 3, H, W)`
        - depth_image: :math:`(B, L, H, W, 1)` if `channels_first` is False, else :math:`(B, L, 1, H, W)`
        - intrinsics: :math:`(B, 1, 4, 4)`
        - poses: :math:`(B, L, 4, 4)`
        - pixel_pos: :math:`(B, L, H, W, 3)` if `channels_first` is False, else :math:`(B, L, 3, H, W)`

    Examples::

        >>> colors = torch.rand([2, 8, 32, 32, 3])
        >>> depths = torch.rand([2, 8, 32, 32, 1])
        >>> intrinsics = torch.rand([2, 1, 4, 4])
        >>> poses = torch.rand([2, 8, 4, 4])
        >>> rgbdimages = gradslam.RGBDImages(colors, depths, intrinsics, poses)
        >>> print(rgbdimages.shape)
        (2, 8, 32, 32)
        >>> rgbd_select = rgbd_frame[1, 4:8]
        >>> print(rgbd_select.shape)
        (1, 4, 32, 32)
        >>> print(rgbdimages.vertex_map.shape)
        (2, 8, 32, 32, 3)
        >>> print(rgbdimages.normal_map.shape)
        (2, 8, 32, 32, 3)
    """

    _INTERNAL_TENSORS = [
        "_rgb_image",
        "_depth_image",
        "_intrinsics",
        "_poses",
        "_pixel_pos",
        "_vertex_map",
        "_normal_map",
        "_global_vertex_map",
        "_global_normal_map",
    ]

    def __init__(
        self,
        rgb_image: torch.Tensor,
        depth_image: torch.Tensor,
        intrinsics: torch.Tensor,
        poses: Optional[torch.Tensor] = None,
        channels_first: bool = False,
        device: Union[torch.device, str, None] = None,
        *,
        pixel_pos: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # input type checks
        if not torch.is_tensor(rgb_image):
            msg = "Expected rgb_image to be of type tensor; got {}"
            raise TypeError(msg.format(type(rgb_image)))
        if not torch.is_tensor(depth_image):
            msg = "Expected depth_image to be of type tensor; got {}"
            raise TypeError(msg.format(type(depth_image)))
        if not torch.is_tensor(intrinsics):
            msg = "Expected intrinsics to be of type tensor; got {}"
            raise TypeError(msg.format(type(intrinsics)))
        if not (poses is None or torch.is_tensor(poses)):
            msg = "Expected poses to be of type tensor or None; got {}"
            raise TypeError(msg.format(type(poses)))
        if not isinstance(channels_first, bool):
            msg = "Expected channels_first to be of type bool; got {}"
            raise TypeError(msg.format(type(channels_first)))
        if not (pixel_pos is None or torch.is_tensor(pixel_pos)):
            msg = "Expected pixel_pos to be of type tensor or None; got {}"
            raise TypeError(msg.format(type(pixel_pos)))

        self._channels_first = channels_first

        # input ndim checks
        if rgb_image.ndim != 5:
            msg = "rgb_image should have ndim=5, but had ndim={}".format(rgb_image.ndim)
            raise ValueError(msg)
        if depth_image.ndim != 5:
            msg = "depth_image should have ndim=5, but had ndim={}".format(
                depth_image.ndim
            )
            raise ValueError(msg)
        if intrinsics.ndim != 4:
            msg = "intrinsics should have ndim=4, but had ndim={}".format(
                intrinsics.ndim
            )
            raise ValueError(msg)
        if poses is not None and poses.ndim != 4:
            msg = "poses should have ndim=4, but had ndim={}".format(poses.ndim)
            raise ValueError(msg)

        self._rgb_image_shape = rgb_image.shape
        self._depth_shape = tuple(
            v if i != self.cdim else 1 for i, v in enumerate(rgb_image.shape)
        )
        self._intrinsics_shape = (rgb_image.shape[0], 1, 4, 4)
        self._poses_shape = (*rgb_image.shape[:2], 4, 4)
        self._pixel_pos_shape = (
            *rgb_image.shape[: self.cdim],
            *rgb_image.shape[self.cdim + 1 :],
            3,
        )

        # input shape checks
        if rgb_image.shape[self.cdim] != 3:
            msg = "Expected rgb_image to have 3 channels on dimension {0}. Got {1} instead"
            raise ValueError(msg.format(self.cdim, rgb_image.shape[self.cdim]))
        if depth_image.shape != self._depth_shape:
            msg = "Expected depth_image to have shape {0}. Got {1} instead"
            raise ValueError(msg.format(self._depth_shape, depth_image.shape))
        if intrinsics.shape != self._intrinsics_shape:
            msg = "Expected intrinsics to have shape {0}. Got {1} instead"
            raise ValueError(msg.format(self._intrinsics_shape, intrinsics.shape))
        if poses is not None and (poses.shape != self._poses_shape):
            msg = "Expected poses to have shape {0}. Got {1} instead"
            raise ValueError(msg.format(self._poses_shape, poses.shape))
        if pixel_pos is not None and (pixel_pos.shape != self._pixel_pos_shape):
            msg = "Expected pixel_pos to have shape {0}. Got {1} instead"
            raise ValueError(msg.format(self._pixel_pos_shape, pixel_pos.shape))

        # assert device type
        inputs = [rgb_image, depth_image, intrinsics, poses, pixel_pos]
        devices = [x.device for x in inputs if x is not None]
        if len(set(devices)) != 1:
            raise ValueError(
                "All inputs must be on same device, but got more than 1 device: {}".format(
                    set(devices)
                )
            )

        self._rgb_image = rgb_image if device is None else rgb_image.to(device)
        self.device = self._rgb_image.device
        self._depth_image = depth_image.to(self.device)
        self._intrinsics = intrinsics.to(self.device)
        self._poses = poses.to(self.device) if poses is not None else None
        self._pixel_pos = pixel_pos.to(self.device) if pixel_pos is not None else None

        self._vertex_map = None
        self._global_vertex_map = None
        self._normal_map = None
        self._global_normal_map = None
        self._valid_depth_mask = None

        self._B, self._L = self._rgb_image.shape[:2]
        self.h = (
            self._rgb_image.shape[3]
            if self._channels_first
            else self._rgb_image.shape[2]
        )
        self.w = (
            self._rgb_image.shape[4]
            if self._channels_first
            else self._rgb_image.shape[3]
        )
        self.shape = (self._B, self._L, self.h, self.w)

    def __getitem__(self, index):
        r"""
        Args:
            index (int or slice or list of int): Specifying the index of the rgbdimages to retrieve.
                Can be an int, slice, list of ints or a boolean tensor.

        Returns:
            gradslam.RGBDImages: Selected rgbdimages. The rgbdimages tensors are not cloned.
        """
        if isinstance(index, tuple) or isinstance(index, int):
            _index_slices = ()
            if isinstance(index, int):
                _index_slices += (slice(index, index + 1),) + (slice(None, None),)
            elif len(index) > 2:
                raise IndexError("Only batch and sequences can be indexed")
            elif isinstance(index, tuple):
                for x in index:
                    if isinstance(x, int):
                        _index_slices += (slice(x, x + 1),)
                    else:
                        _index_slices += (x,)

            new_rgb = self._rgb_image[_index_slices[0], _index_slices[1]]
            if new_rgb.shape[0] == 0:
                raise IndexError(
                    "Incorrect indexing at dimension 0, make sure range is within 0 and {0}".format(
                        self._B
                    )
                )
            if new_rgb.shape[1] == 0:
                raise IndexError(
                    "Incorrect indexing at dimension 1, make sure range is within 0 and {0}".format(
                        self._L
                    )
                )
            new_depth = self._depth_image[_index_slices[0], _index_slices[1]]
            new_intrinsics = self._intrinsics[_index_slices[0], :]
            other = RGBDImages(
                new_rgb,
                new_depth,
                new_intrinsics,
                channels_first=self.channels_first,
            )
            for k in self._INTERNAL_TENSORS:
                if k in ["_rgb_image", "_depth_image", "_intrinsics"]:
                    continue
                v = getattr(self, k)
                if torch.is_tensor(v):
                    setattr(other, k, v[_index_slices[0], _index_slices[1]])
            return other
        else:
            raise IndexError(index)

    def __len__(self):
        return self._B

    @property
    def channels_first(self):
        r"""Gets bool indicating whether RGBDImages representation is channels first or not

        Returns:
            bool: True if RGBDImages representation is channels first, else False.
        """
        return self._channels_first

    @property
    def cdim(self):
        r"""Gets the channel dimension

        Returns:
            int: :math:`2` if self.channels_first is True, else :math:`4`.
        """
        return 2 if self.channels_first else 4

    @property
    def rgb_image(self):
        r"""Gets the rgb image

        Returns:
            torch.Tensor: tensor representation of `rgb_image`

        Shape:
            - Output: :math:`(B, L, H, W, 3)` if self.channels_first is False, else :math:`(B, L, 3, H, W)`
        """
        return self._rgb_image

    @property
    def depth_image(self):
        r"""Gets the depth image

        Returns:
            torch.Tensor: tensor representation of `depth_image`

        Shape:
            - Output: :math:`(B, L, H, W, 1)` if self.channels_first is False, else :math:`(B, L, 1, H, W)`
        """
        return self._depth_image

    @property
    def intrinsics(self):
        r"""Gets the `intrinsics`

        Returns:
            torch.Tensor: tensor representation of `intrinsics`

        Shape:
            - Output: :math:`(B, 1, 4, 4)`
        """
        return self._intrinsics

    @property
    def poses(self):
        r"""Gets the `poses`

        Returns:
            torch.Tensor: tensor representation of `poses`

        Shape:
            - Output: :math:`(B, L, 4, 4)`
        """
        return self._poses

    @property
    def pixel_pos(self):
        r"""Gets the `pixel_pos`

        Returns:
            torch.Tensor: tensor representation of `pixel_pos`

        Shape:
            - Output: :math:`(B, L, H, W, 3)` if self.channels_first is False, else :math:`(B, L, 3, H, W)`
        """
        return self._pixel_pos

    @property
    def valid_depth_mask(self):
        r"""Gets a mask which is True wherever `self.dept_image` is :math:`>0`

        Returns:
            torch.Tensor: Tensor of dtype bool with same shape as `self.depth_image`. Tensor is True wherever
            `self.depth_image` > 0, and False otherwise.

        Shape:
            - Output: :math:`(B, L, H, W, 1)` if self.channels_first is False, else :math:`(B, L, 1, H, W)`
        """
        if self._valid_depth_mask is None:
            self._valid_depth_mask = self._depth_image > 0
        return self._valid_depth_mask

    @property
    def has_poses(self):
        r"""Determines whether self has `poses` or not

        Returns:
            bool
        """
        return self._poses is not None

    @property
    def vertex_map(self):
        r"""Gets the local vertex maps

        Returns:
            torch.Tensor: tensor representation of local coordinated vertex maps

        Shape:
            - Output: :math:`(B, L, H, W, 3)` if self.channels_first is False, else :math:`(B, L, 3, H, W)`
        """
        if self._vertex_map is None:
            self._compute_vertex_map()
        return self._vertex_map

    @property
    def normal_map(self):
        r"""Gets the local normal maps

        Returns:
            torch.Tensor: tensor representation of local coordinated normal maps

        Shape:
            - Output: :math:`(B, L, H, W, 3)` if self.channels_first is False, else :math:`(B, L, 3, H, W)`
        """
        if self._normal_map is None:
            self._compute_normal_map()
        return self._normal_map

    @property
    def global_vertex_map(self):
        r"""Gets the global vertex maps

        Returns:
            torch.Tensor: tensor representation of global coordinated vertex maps

        Shape:
            - Output: :math:`(B, L, H, W, 3)` if self.channels_first is False, else :math:`(B, L, 3, H, W)`
        """
        if self._global_vertex_map is None:
            self._compute_global_vertex_map()
        return self._global_vertex_map

    @property
    def global_normal_map(self):
        r"""Gets the global normal maps

        Returns:
            torch.Tensor: tensor representation of global coordinated normal maps

        Shape:
            - Output: :math:`(B, L, H, W, 3)` if self.channels_first is False, else :math:`(B, L, 3, H, W)`
        """
        if self._global_normal_map is None:
            self._compute_global_normal_map()
        return self._global_normal_map

    @rgb_image.setter
    def rgb_image(self, value):
        r"""Updates `rgb_image` of self.

        Args:
            value (torch.Tensor): New rgb image values

        Shape:
            - value: :math:`(B, L, H, W, 3)` if self.channels_first is False, else :math:`(B, L, 3, H, W)`
        """
        if value is not None:
            self._assert_shape(value, self._rgb_image_shape)
        self._rgb_image = value

    @depth_image.setter
    def depth_image(self, value):
        r"""Updates `depth_image` of self.

        Args:
            value (torch.Tensor): New depth image values

        Shape:
            - value: :math:`(B, L, H, W, 1)` if self.channels_first is False, else :math:`(B, L, 1, H, W)`
        """
        if value is not None:
            self._assert_shape(value, self._depth_image_shape)
        self._depth_image = value
        self._vertex_map = None
        self._normal_map = None
        self._global_vertex_map = None
        self._global_normal_map = None

    @intrinsics.setter
    def intrinsics(self, value):
        r"""Updates `intrinsics` of self.

        Args:
            value (torch.Tensor): New intrinsics values

        Shape:
            - value: :math:`(B, 1, 4, 4)`
        """
        if value is not None:
            self._assert_shape(value, self._intrinsics_shape)
        self._intrinsics = value
        self._vertex_map = None
        self._normal_map = None
        self._global_vertex_map = None
        self._global_normal_map = None

    @poses.setter
    def poses(self, value):
        r"""Updates `poses` of self.

        Args:
            value (torch.Tensor): New pose values

        Shape:
            - value: :math:`(B, L, 4, 4)`
        """
        if value is not None:
            self._assert_shape(value, self._poses_shape)
        self._poses = value
        self._global_vertex_map = None
        self._global_normal_map = None

    def detach(self):
        r"""Detachs RGBDImages object. All internal tensors are detached individually.

        Returns:
            gradslam.RGBDImages: detached gradslam.RGBDImages object
        """
        other = self.clone()
        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.detach())

        return other

    def clone(self):
        r"""Returns deep copy of RGBDImages object. All internal tensors are cloned individually.

        Returns:
            gradslam.RGBDImages: cloned gradslam.RGBDImages object
        """
        other = RGBDImages(
            rgb_image=self._rgb_image.clone(),
            depth_image=self._depth_image.clone(),
            intrinsics=self._intrinsics.clone(),
            channels_first=self.channels_first,
        )
        for k in self._INTERNAL_TENSORS:
            if k in ["_rgb_image", "_depth_image", "_intrinsics"]:
                continue
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
        return other

    def to(self, device: Union[torch.device, str], copy: bool = False):
        r"""Match functionality of torch.Tensor.to(device)
        If copy = True or the self Tensor is on a different device, the returned tensor is a copy of self with the
        desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device, then self is returned.

        Args:
            device (torch.device or str): Device id for the new tensor.
            copy (bool): Boolean indicator whether or not to clone self. Default False.

        Returns:
            gradslam.RGBDImages
        """
        # hack to know which gpu is used when device("cuda")
        device = torch.Tensor().to(device).device
        if not copy and self.device == device:
            return self

        other = self.clone()
        other.device = device

        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.to(device))

        return other

    def cpu(self):
        r"""Match functionality of torch.Tensor.cpu()

        Returns:
            gradslam.RGBDImages
        """
        return self.to(torch.device("cpu"))

    def cuda(self):
        r"""Match functionality of torch.Tensor.cuda()

        Returns:
            gradslam.RGBDImages
        """
        return self.to(torch.device("cuda"))

    def to_channels_last(self, copy: bool = False):
        r"""Converts to channels last representation
        If copy = True or self channels_first is True, the returned RGBDImages object is a copy of self with
        channels last representation.
        If copy = False and self channels_first is already False, then self is returned.

        Args:
            copy (bool): Boolean indicator whether or not to clone self. Default False.

        Returns:
            gradslam.RGBDImages
        """
        if not (copy or self.channels_first):
            return self
        return self.clone().to_channels_last_()

    def to_channels_first(self, copy: bool = False):
        r"""Converts to channels first representation
        If copy = True or self channels_first is False, the returned RGBDImages object is a copy of self with
        channels first representation.
        If copy = False and self channels_first is already True, then self is returned.

        Args:
            copy (bool): Boolean indicator whether or not to clone self. Default False.

        Returns:
            gradslam.RGBDImages
        """
        if not copy and self.channels_first:
            return self
        return self.clone().to_channels_first_()

    def to_channels_last_(self):
        r"""Converts to channels last representation. In place operation.

        Returns:
            gradslam.RGBDImages
        """
        if not self.channels_first:
            return self
        ordering = (0, 1, 3, 4, 2)  # B L C H W -> B L H W C
        permute = RGBDImages._permute_if_not_None
        self._rgb_image = permute(self._rgb_image, ordering)
        self._depth_image = permute(self._depth_image, ordering)
        self._vertex_map = permute(self._vertex_map, ordering)
        self._global_vertex_map = permute(self._global_vertex_map, ordering)
        self._normal_map = permute(self._normal_map, ordering)
        self._global_normal_map = permute(self._global_normal_map, ordering)

        self._channels_first = False
        self._rgb_image_shape = tuple(self._rgb_image.shape)
        self._depth_image_shape = tuple(self._depth_image.shape)
        return self

    def to_channels_first_(self):
        r"""Converts to channels first representation. In place operation.

        Returns:
            gradslam.RGBDImages
        """
        if self.channels_first:
            return self
        ordering = (0, 1, 4, 2, 3)  # B L H W C -> B L C H W
        permute = RGBDImages._permute_if_not_None
        self._rgb_image = permute(self._rgb_image, ordering)
        self._depth_image = permute(self._depth_image, ordering)
        self._vertex_map = permute(self._vertex_map, ordering)
        self._global_vertex_map = permute(self._global_vertex_map, ordering)
        self._normal_map = permute(self._normal_map, ordering)
        self._global_normal_map = permute(self._global_normal_map, ordering)

        self._channels_first = True
        self._rgb_image_shape = tuple(self._rgb_image.shape)
        self._depth_image_shape = tuple(self._depth_image.shape)
        return self

    @staticmethod
    def _permute_if_not_None(
        tensor: Optional[torch.Tensor], ordering: tuple, contiguous: bool = True
    ):
        r"""Permutes input if it is not None based on given ordering

        Args:
            tensor (torch.Tensor or None): Tensor to be permuted, or None
            ordering (tuple): The desired ordering of dimensions
            contiguous (bool): Whether to call `.contiguous()` on permuted tensor before returning.
                Default: True

        Returns:
            torch.Tensor or None: Permuted tensor or None
        """
        if tensor is None:
            return None
        assert torch.is_tensor(tensor)
        return (
            tensor.permute(*ordering).contiguous()
            if contiguous
            else tensor.permute(*ordering)
        )

    def _compute_vertex_map(self):
        r"""Coverts a batch of depth images into a batch of vertex maps."""
        B, L = self.shape[:2]
        device = self._depth_image.device
        if self._pixel_pos is None:
            meshgrid = (
                create_meshgrid(self.h, self.w, normalized_coords=False)
                .view(1, 1, self.h, self.w, 2)
                .repeat(B, L, 1, 1, 1)
                .to(device)
            )
            self._pixel_pos = torch.cat(
                [
                    meshgrid[..., 1:],
                    meshgrid[..., 0:1],
                    torch.ones_like(meshgrid[..., 0].unsqueeze(-1)),
                ],
                -1,
            )
        Kinv = inverse_intrinsics(self._intrinsics)[..., :3, :3]
        # TODO: Time tests for all einsums. Might not be efficient (especially on cpu).
        Kinv = Kinv.repeat(1, L, 1, 1)
        # Add an extra channel of ones to meshgrid for z values
        if self.channels_first:
            self._vertex_map = (
                torch.einsum("bsjc,bshwc->bsjhw", Kinv, self._pixel_pos)
                * self._depth_image
            )
        else:
            self._vertex_map = (
                torch.einsum("bsjc,bshwc->bshwj", Kinv, self._pixel_pos)
                * self._depth_image
            )
        # zero out missing depth values
        self._vertex_map = self._vertex_map * self.valid_depth_mask.to(
            self._vertex_map.dtype
        )

    def _compute_global_vertex_map(self):
        r"""Coverts a batch of local vertex maps into a batch of global vertex maps."""
        if self._poses is None:
            self._global_vertex_map = self.vertex_map.clone()
            return

        local_vertex_map = self.vertex_map

        B, L = self.shape[:2]
        rmat = self._poses[..., :3, :3]
        tvec = self._poses[..., :3, 3]
        # TODO: Time tests for all einsums. Might not be efficient (especially on cpu).
        # Add an extra channel of ones to meshgrid for z values
        if self.channels_first:
            self._global_vertex_map = torch.einsum(
                "bsjc,bschw->bsjhw", rmat, local_vertex_map
            )
            self._global_vertex_map = self._global_vertex_map + tvec.view(B, L, 3, 1, 1)
        else:
            self._global_vertex_map = torch.einsum(
                "bsjc,bshwc->bshwj", rmat, local_vertex_map
            )
            self._global_vertex_map = self._global_vertex_map + tvec.view(B, L, 1, 1, 3)

        # zero out missing depth values
        self._global_vertex_map = self._global_vertex_map * self.valid_depth_mask.to(
            self._global_vertex_map.dtype
        )

    def _compute_normal_map(self):
        r"""Converts a batch of vertex maps to a batch of normal maps."""
        dhoriz: torch.Tensor = torch.zeros_like(self.vertex_map)
        dverti: torch.Tensor = torch.zeros_like(self.vertex_map)

        if self.channels_first:
            dhoriz[..., :-1] = self.vertex_map[..., 1:] - self.vertex_map[..., :-1]
            dverti[..., :-1, :] = (
                self.vertex_map[..., 1:, :] - self.vertex_map[..., :-1, :]
            )
            dhoriz[..., -1] = dhoriz[..., -2]
            dverti[..., -1, :] = dverti[..., -2, :]
            dim = 2
        else:
            dhoriz[..., :-1, :] = (
                self.vertex_map[..., 1:, :] - self.vertex_map[..., :-1, :]
            )
            dverti[..., :-1, :, :] = (
                self.vertex_map[..., 1:, :, :] - self.vertex_map[..., :-1, :, :]
            )
            dhoriz[..., -1, :] = dhoriz[..., -2, :]
            dverti[..., -1, :, :] = dverti[..., -2, :, :]
            dim = -1

        normal_map: torch.Tensor = torch.cross(dhoriz, dverti, dim=dim)
        norm: torch.Tensor = normal_map.norm(dim=dim).unsqueeze(dim)

        self._normal_map: torch.Tensor = normal_map / torch.where(
            norm == 0, torch.ones_like(norm), norm
        )
        # zero out missing depth values
        self._normal_map = self._normal_map * self.valid_depth_mask.to(
            self._normal_map.dtype
        )

    def _compute_global_normal_map(self):
        r"""Coverts a batch of local noraml maps into a batch of global normal maps."""
        if self._poses is None:
            self._global_normal_map = self.normal_map.clone()
            return

        local_normal_map = self.normal_map

        B, L = self.shape[:2]
        rmat = self._poses[..., :3, :3]
        if self.channels_first:
            self._global_normal_map = torch.einsum(
                "bsjc,bschw->bsjhw", rmat, local_normal_map
            )
        else:
            self._global_normal_map = torch.einsum(
                "bsjc,bshwc->bshwj", rmat, local_normal_map
            )

    def plotly(
        self,
        index: int,
        include_depth: bool = True,
        as_figure: bool = True,
        ms_per_frame: int = 50,
    ):
        r"""Converts `index`-th sequence of rgbd images to either a `plotly.graph_objects.Figure` or a
        list of dicts containing `plotly.graph_objects.Image` objects of rgb and (optionally) depth images:

        .. code-block:: python


            frames = [
                {'name': 0, 'data': [rgbImage0, depthImage0], 'traces': [0, 1]},
                {'name': 1, 'data': [rgbImage1, depthImage1], 'traces': [0, 1]},
                {'name': 2, 'data': [rgbImage2, depthImage2], 'traces': [0, 1]},
                ...
            ]

        Returned `frames` can be passed to `go.Figure(frames=frames)`.

        Args:
            index (int): Index of which rgbd image (from the batch of rgbd images) to convert to plotly
                representation.
            include_depth (bool): If True, will include depth images in the returned object. Default: True
            as_figure (bool): If True, returns a `plotly.graph_objects.Figure` object which can easily
                be visualized by calling `.show()` on. Otherwise, returns a list of dicts (`frames`)
                which can be passed to `go.Figure(frames=frames)`. Default: True
            ms_per_frame (int): Milliseconds per frame when play button is hit. Only applicable if `as_figure=True`.
                Default: 50

        Returns:
            plotly.graph_objects.Figure or list of dict: If `as_figure` is True, will return
            `plotly.graph_objects.Figure` object from the `index`-th sequence of rgbd images. Else,
            returns a list of dicts (`frames`).
        """
        if not isinstance(index, int):
            raise TypeError("Index should be int, but was {}.".format(type(index)))

        def frame_args(duration):
            return {
                "frame": {"duration": duration, "redraw": True},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

        torch_rgb = self.rgb_image[index]
        if (torch_rgb.max() < 1.1).item():
            torch_rgb = torch_rgb * 255
        torch_rgb = torch.clamp(torch_rgb, min=0.0, max=255.0)
        numpy_rgb = torch_rgb.detach().cpu().numpy().astype("uint8")
        Image_rgb = [numpy_to_plotly_image(rgb, i) for i, rgb in enumerate(numpy_rgb)]

        if not include_depth:
            frames = [{"data": [frame], "name": i} for i, frame in enumerate(Image_rgb)]
        else:
            torch_depth = self.depth_image[index, ..., 0]
            scale = 10 ** torch.log10(255.0 / torch_depth.detach().max()).floor().item()
            numpy_depth = (torch_depth * scale).detach().cpu().numpy().astype("uint8")
            Image_depth = [
                numpy_to_plotly_image(d, i, True, scale)
                for i, d in enumerate(numpy_depth)
            ]
            frames = [
                {"name": i, "data": list(frame), "traces": [0, 1]}
                for i, frame in enumerate(zip(Image_rgb, Image_depth))
            ]

        if not as_figure:
            return frames

        steps = [
            {"args": [[i], frame_args(0)], "label": i, "method": "animate"}
            for i in range(self._L)
        ]
        sliders = [
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {"prefix": "Frame: "},
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": steps,
            }
        ]
        updatemenus = [
            {
                "buttons": [
                    {
                        "args": [None, frame_args(ms_per_frame)],
                        "label": "&#9654;",
                        "method": "animate",
                    },
                    {
                        "args": [[None], frame_args(0)],
                        "label": "&#9724;",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ]

        if not include_depth:
            fig = make_subplots(rows=1, cols=1, subplot_titles=("RGB",))
            fig.add_traces(frames[0]["data"][0])
        else:
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=("RGB", "Depth"),
                shared_xaxes=True,
                shared_yaxes=False,
                vertical_spacing=0.1,
            )
            fig.add_trace(frames[0]["data"][0], row=1, col=1)  # initial rgb frame
            fig.add_trace(frames[0]["data"][1], row=2, col=1)  # initial depth frame
            fig.update_layout(scene=dict(aspectmode="data"))
            fig.update_layout(
                autosize=False, height=1080
            )  # autosize is not perfect with subplots

        fig.update(frames=frames)
        fig.update_layout(updatemenus=updatemenus, sliders=sliders)
        return fig

    # TODO: rotation + transformation: keep in mind to apply to vertices, normals *and* poses

    def _assert_shape(self, value: torch.Tensor, shape: tuple):
        r"""Asserts if value is a tensor with same shape as `shape`

        Args:
            value (torch.Tensor): Tensor to check shape of
            shape (tuple): Expected shape of value
        """
        if not isinstance(value, torch.Tensor):
            raise TypeError("value must be torch.Tensor. Got {}".format(type(value)))
        if value.shape != shape:
            msg = "Expected value to have shape {0}. Got {1} instead"
            raise ValueError(msg.format(shape, value.shape))
