from typing import Optional

import torch

from ..geometry.geometryutils import create_meshgrid
from ..geometry.projutils import inverse_intrinsics

__all__ = ["RGBDImages"]


class RGBDImages(object):
    r"""Initializes an RGBDImage object consisting  of the rgb images, depth maps, intrinsics.
    Optionally allows for poses, vertex maps, and normal maps.

    Args:
        rgb_image (torch.Tensor): 3-channel RGB Image input
        depth_image (torch.Tensor): 1-channel depth map input
        intrinsics (torch.Tensor): camera intrinsics
        poses (Optional[torch.Tensor]): camera extrinsics. Default: None
        channels_first(Optional[bool]): indicates whether channel dimension first
                                        i.e. :math:`(B \times L \times H \times W \ times 3)`
                                        or :math:`(B \times L \times 3 \times H \times W)`. Default : True
        pixel_pos (torch.Tensor): Similar to meshgrid but with extra channel of 1s at the end. If provided, can
            save computations when computing vertex maps. Default: None

    Shape:
        - rgb_image: :math:`(B \times L \times H \times W \ times 3)` or `(B \times L \times 3 \times H \times W)`, where
            :math:`B` denotes the batch size
            :math:`L` denotes the sequence length
            :math:`H` denotes the height of the image
            :math:`W` denotes the width of the image.
        - depth_image: :math:`(B \times L \times H \times W \ times 1)` or `(B \times L \times 1 \times H \times W)`
        - intrinsics: :math:`(B \times 1 \times 4 \times 4)`
        - poses: :math:`(B \times L \times 4 \times 4)`
        - channels_first: None

    Examples::
        >>> rgb = torch.rand([3,16,3,32,32])
        >>> depth = torch.rand([3,16,1,32,32])
        >>> intrinsics = torch.rand([3,1,4,4])
        >>> poses = torch.rand([3,16,4,4])
        >>> rgbd_frame = gs.structures.RGBDImages(rgb, depth, intrinsics, poses, channels_first=False)
        >>> print (rgbd_frame.shape)
        (3, 16, 32, 32)
        >>> rgbd_select = rgbd_frame[2,5:10]
        >>> print (rgbd_select.shape)
        (1, 5, 32, 32)

    """

    _INTERNAL_TENSORS = [
        "_vertex_map",
        "_normal_map",
        "_poses",
    ]

    def __init__(
        self,
        rgb_image: torch.Tensor,
        depth_image: torch.Tensor,
        intrinsics: torch.Tensor,
        poses: Optional[torch.Tensor] = None,
        channels_first: Optional[bool] = True,
        *,
        pixel_pos=None,
    ):
        super(RGBDImages, self).__init__()

        self._assert_is_bool(channels_first)

        self.channels_first = channels_first
        self._pixel_pos = pixel_pos
        self.device = depth_image.device

        # check rgb image properties
        self._assert_tensor(rgb_image)
        self._assert_image_shape(rgb_image)
        self._assert_rgb_channels(rgb_image)

        # check depth image properties
        self._assert_tensor(depth_image)
        self._assert_image_shape(depth_image)
        self._assert_depth_channels(depth_image)

        # check common properties of rgb and depth image
        self._assert_rgb_depth_properties(rgb_image, depth_image)

        # checking intrinsics properties
        self._assert_tensor(intrinsics)
        self._assert_intrinsics_shape(intrinsics)

        # checking common properties rgb and intrinsics
        self._assert_rgb_intrinsics_properties(rgb_image, intrinsics)

        # assert device type
        self._assert_device(depth_image)
        self._assert_device(intrinsics)

        self.rgb_image = rgb_image
        self.depth_image = depth_image
        self.intrinsics = intrinsics
        self.channels_first = channels_first

        self._vertex_map = None
        self._normal_map = None
        self._valid_depth_mask = None
        self._poses = None

        self.batch_size = self.rgb_image.shape[0]
        self.sequence_length = self.rgb_image.shape[1]

        if self.channels_first:
            self.h = self.rgb_image.shape[3]
            self.w = self.rgb_image.shape[4]
            self.cdim = 2
        else:
            self.h = self.rgb_image.shape[2]
            self.w = self.rgb_image.shape[3]
            self.cdim = 4

        if poses is not None:
            self._assert_tensor(poses)
            self._assert_device(poses)
            self._assert_extrinsics_shape(poses)
            self._poses = poses

        self.shape = (self.batch_size, self.sequence_length, self.h, self.w)

    def __getitem__(self, index):
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

            new_rgb = self.rgb_image[_index_slices[0], _index_slices[1]]
            if new_rgb.shape[0] == 0:
                raise IndexError(
                    "Incorrect indexing at dimension 0, make sure range is within 0 and {0}".format(
                        self.batch_size
                    )
                )
            if new_rgb.shape[1] == 0:
                raise IndexError(
                    "Incorrect indexing at dimension 1, make sure range is within 0 and {0}".format(
                        self.sequence_length
                    )
                )
            new_depth = self.depth_image[_index_slices[0], _index_slices[1]]
            new_intrinsics = self.intrinsics[_index_slices[0], :]
            new_poses = (
                self._poses[_index_slices[0], _index_slices[1]]
                if self._poses is not None
                else None
            )
            return RGBDImages(
                new_rgb,
                new_depth,
                new_intrinsics,
                new_poses,
                self.channels_first,
            )
        else:
            raise IndexError(index)

    def __len__(self):
        return self.batch_size

    @property
    def valid_depth_mask(self):
        if self._valid_depth_mask is None:
            self._valid_depth_mask = self.depth_image > 0
        return self._valid_depth_mask

    @property
    def vertex_map(self):
        if self._vertex_map is None:
            self._compute_vertex_map()
        return self._vertex_map

    @property
    def normal_map(self):
        if self._normal_map is None:
            self._compute_normal_map()
        return self._normal_map

    @property
    def poses(self):
        return self._poses

    @poses.setter
    def poses(self, value):
        self._poses = value
        self._vertex_map = None
        self._normal_map = None
        return self._poses

    @staticmethod
    def _assert_is_bool(inp):
        """Asserts that the input is a bool

        Arguments:
            inp ([type]): any input type, expected bool

        Raises:
            TypeError: Error if types don't match
        """
        if not isinstance(inp, bool):
            raise TypeError(
                "Expected input to be of type bool."
                " Got {0} instead".format(type(inp))
            )

    @staticmethod
    def _assert_tensor(inp):
        """Asserts that the input is a torch.Tensor

        Arguments:
            inp ([type]): any input type, expected torch.Tensor

        Raises:
            TypeError: Error if types don't match
        """
        if not torch.is_tensor(inp):
            raise TypeError(
                "Expected input to be of type torch.Tensor. Got {0} instead".format(
                    type(inp)
                )
            )

    def _assert_device(self, inp):
        """Asserts that the input device is the same

        Arguments:
            inp ([type]): any input type, expected torch.Tensor

        Raises:
            TypeError: Error if types don't match
        """
        if inp.device != self.device:
            raise TypeError(
                "Expected input to be of device type {0}. Got {1} instead".format(
                    self.device, inp.device
                )
            )

    @staticmethod
    def _assert_image_shape(inp: torch.Tensor):
        """Asserts that the input has shape of length 5

        Arguments:
            inp ([torch.Tensor]): input tensor

        Raises:
            ValueError: Error if shape incorrect
        """
        if len(inp.shape) != 5:
            raise ValueError(
                "Expected input to have 5 dimensional tensor of shape (B x L x H x W x 3) or (B x L x 3 x H x W). Got {0} dimensional shape instead".format(
                    len(inp.shape)
                )
            )

    @staticmethod
    def _assert_intrinsics_shape(inp: torch.Tensor):
        """Asserts that the input has dimensions :math: `(B \times 1 \times 4 \times 4)`

        Arguments:
            inp ([torch.Tensor]): input tensor

        Raises:
            ValueError: Error if shape incorrect
        """
        if len(inp.shape) != 4:
            raise ValueError(
                "Expected input to have 4 dimensional tensor of shape (B x 1 x 4 x 4). Got B = {0} instead".format(
                    len(inp.shape)
                )
            )

        if inp.shape[1:] != (1, 4, 4):
            raise ValueError(
                "Expected input to have 4 dimensional tensor of shape (B x 1 x 4 x 4). Got {0} shape instead".format(
                    (inp.shape)
                )
            )

    def _assert_extrinsics_shape(self, inp: torch.Tensor):
        """Asserts that the input has dimensions :math: `(B \times L \times 4 \times 4)`

        Arguments:
            inp ([torch.Tensor]): input tensor

        Raises:
            ValueError: Error if shape incorrect
        """
        if len(inp.shape) != 4:
            raise ValueError(
                "Expected input to have 4 dimensional tensor of shape (B x L x 4 x 4). Got {0} dimensional instead".format(
                    len(inp.shape)
                )
            )

        if inp.shape[0] != self.batch_size:
            raise ValueError(
                "Expected input to have the batch size : {0}. Got {1} instead".format(
                    self.batch_size, inp.shape[0]
                )
            )

        if inp.shape[1] != self.sequence_length:
            raise ValueError(
                "Expected input to have the sequence length : {0}. Got {1} instead".format(
                    self.sequence_length, inp.shape[1]
                )
            )

        if inp.shape[2:] != torch.Size([4, 4]):
            raise ValueError(
                "Matrix Dimension mismatch. Expected torch.Size([4,4]). Got {0} instead".format(
                    inp.shape[2:]
                )
            )

    def _assert_rgb_channels(self, inp: torch.Tensor):
        """Asserts that an rgb image has 3 channels

        Arguments:
            inp ([torch.Tensor]): input tensor

        Raises:
            ValueError: Error if number of channels incorrect
        """
        _ch_check = 2
        if not self.channels_first:
            _ch_check = 4

        if inp.shape[_ch_check] != 3:
            raise ValueError(
                "Expected input to have 3 channel RGB images. Got {0} instead".format(
                    inp.shape[3]
                )
            )

    def _assert_depth_channels(self, inp: torch.Tensor):
        """Asserts that a depth map has 1 channel

        Arguments:
            inp ([torch.Tensor]): input tensor

        Raises:
            ValueError: Error if number of channels incorrect
        """
        _ch_check = 2
        if not self.channels_first:
            _ch_check = 4

        if inp.shape[_ch_check] != 1:
            raise ValueError(
                "Expected input to have 1 channel depth maps. Got {0} instead".format(
                    inp.shape[3]
                )
            )

    def _assert_rgb_depth_properties(self, inp1: torch.Tensor, inp2: torch.Tensor):
        """Asserts that rgb and depth channels have similar properties

        Arguments:
            inp1 (torch.Tensor): rgb tensor
            inp2 (torch.Tensor): depth tensor
        """

        if inp1.shape[0] != inp2.shape[0]:
            raise ValueError("Batch size of inputs should match")

        if inp1.shape[1] != inp2.shape[1]:
            raise ValueError("Sequence length of inputs should match")

        if self.channels_first:
            if inp1.shape[3:] != inp2.shape[3:]:
                raise ValueError(
                    "RGB height and width should match depth map height and width"
                )
        else:
            if inp1.shape[2:4] != inp2.shape[2:4]:
                raise ValueError(
                    "RGB height and width should match depth map height and width"
                )

    @staticmethod
    def _assert_rgb_intrinsics_properties(inp1: torch.Tensor, inp2: torch.Tensor):
        """Asserts that rgb and instrinsics channels have similar properties

        Arguments:
            inp1 (torch.Tensor): rgb tensor
            inp2 (torch.Tensor): intrinsics tensor
        """
        if inp1.shape[0] != inp2.shape[0]:
            raise ValueError("Batch size of image tensors and intrinsics should match")

    def detach(self):
        r"""Detachs RGBDImages object. All internal tensors are detached individually.

        Returns:
            gradslam.RGBDImages: detached gradslam.RGBDImages object
        """
        if self.device == device:
            return self

        other = self.clone()
        other.rgb_image = other.rgb_image.detach()
        other.depth_image = other.depth_image.detach()
        other.intrinsics = other.intrinsics.detach()

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
        poses = None if self._poses is None else self._poses.clone()

        other = RGBDImages(
            rgb_image=self.rgb_image,
            depth_image=self.depth_image,
            intrinsics=self.intrinsics,
            poses=poses,
            channels_first=self.channels_first,
        )

        for k in self._INTERNAL_TENSORS:
            v = getattr(self, k)
            if torch.is_tensor(v):
                setattr(other, k, v.clone())
        return other

    def to(self, device: str):
        r"""Match functionality of torch.Tensor.to(device)
        If copy = True or the self Tensor is on a different device, the returned tensor is a copy of self with the
        desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device, then self is returned.

        Args:
            device (str): Device id for the new tensor.
            copy (bool): Boolean indicator whether or not to clone self. Default False.

        Returns:
            gradslam.RGBDImages
        """
        if self.device == device:
            return self

        other = self.clone()
        # hack to know which gpu is used when device("cuda")
        other.device = torch.Tensor().to(device).device
        other.rgb_image = other.rgb_image.to(device)
        other.depth_image = other.depth_image.to(device)
        other.intrinsics = other.intrinsics.to(device)

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

    def _compute_vertex_map(self):
        r"""Coverts a batch of depth images into a batch of vertex maps."""
        B, L = self.shape[:2]
        device = self.depth_image.device
        if self._pixel_pos is None:
            meshgrid = (
                create_meshgrid(self.h, self.w, normalized_coords=False)
                .view(1, 1, self.h, self.w, 2)
                .repeat(B, L, 1, 1, 1)
                .to(device)
            )
            self._pixel_pos = torch.cat(
                [meshgrid, torch.ones_like(meshgrid[..., 0].unsqueeze(-1))], -1
            )
        Kinv = inverse_intrinsics(self.intrinsics)[..., :3, :3]
        # Assume identity extrinsics (pose) if no pose is provided, else compose pose transformation and inverse
        # intrinsics into one.
        rmat = None if self._poses is None else self._poses[..., :3, :3]
        tvec = None if self._poses is None else self._poses[..., :3, 3]
        # TODO: Time tests for all 3 einsums. Might not be efficient (especially on cpu).
        TKinv = (
            Kinv.repeat(1, L, 1, 1)
            if self._poses is None
            else torch.einsum("bsjk,bkm->bsjm", rmat, Kinv.squeeze(1))
        )
        # Add an extra channel of ones to meshgrid for z values
        if self.channels_first:
            self._vertex_map = (
                torch.einsum("bsjc,bshwc->bsjhw", TKinv, self._pixel_pos)
                * self.depth_image
            )
            self._vertex_map = (
                self._vertex_map
                if tvec is None
                else self._vertex_map + tvec.view(B, L, 3, 1, 1)
            )
        else:
            self._vertex_map = (
                torch.einsum("bsjc,bshwc->bshwj", TKinv, self._pixel_pos)
                * self.depth_image
            )
            self._vertex_map = (
                self._vertex_map
                if tvec is None
                else self._vertex_map + tvec.view(B, L, 1, 1, 3)
            )
        # zero out missing depth values
        self._vertex_map = self._vertex_map * self.valid_depth_mask.to(
            self._vertex_map.dtype
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
            dim = 2
        else:
            dhoriz[..., :-1, :] = (
                self.vertex_map[..., 1:, :] - self.vertex_map[..., :-1, :]
            )
            dverti[..., :-1, :, :] = (
                self.vertex_map[..., 1:, :, :] - self.vertex_map[..., :-1, :, :]
            )
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

    # TODO: rotation + transformation: keep in mind to apply to vertices, normals *and* poses
