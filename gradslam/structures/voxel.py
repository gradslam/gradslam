import torch

from ..geometry import homogenize_points
from ..geometry import transform_pts_nd_KF


class VoxelGrid(object):
    r"""Class to store a voxel grid. """

    def __init__(
        self,
        origin=torch.FloatTensor([0, 0, 0]),
        extent=torch.FloatTensor([1, 1, 1]),
        res=torch.LongTensor([8, 8, 8]),
        device="cpu",
    ):
        r"""Initializes a `VoxelGrid` object with the extents and
        resolution specified.

        Args:
            origin (torch.FloatTensor): Origin of the voxel grid (assumed to
                be at the center of the grid) in (x, y, z) coordinates.
            extent (torch.FloatTensor): Extent of each dimension of the voxel
                grid (x, y, and z respectively).
            res (torch.LongTensor): Number of voxels to be used to represent
                the x, y, and z axes respectively.
            device (str, optional): Device to store the voxel grid on (default:
                'cpu') (choices: ['cpu', 'gpu'])

        """

        super(VoxelGrid, self).__init__()

        # Initialize class parameters
        self.device = device
        self.origin = origin.to(self.device)
        self.extent = extent.to(self.device)
        self.res = res.to(self.device).long()

        # Compute the size of each voxel (x, y, z directions)
        self.voxelsize = self.extent / self.res.float()

        # Voxel coordinates
        self.voxel_coords = None
        # World coordinates
        self.world_coords = None

    def get_min_bound(self):
        r"""Returns the minimum coordinate of the voxel grid. """
        return self.origin - 0.5 * self.extent
        # return self.origin

    def get_max_bound(self):
        r"""Returns the maximum coordinate of the voxel grid. """
        return self.origin + 0.5 * self.extent

    def get_center(self):
        r"""Returns the center of the voxel grid. """
        # return 0.5 * (self.get_min_bound + self.get_max_bound)
        return self.origin

    def get_voxel_coords(self):
        r"""Returns a meshgrid of voxel coordinates. 

        Returns:
            (torch.Tensor): Voxel coordinates (shape: 1 x self.res[0]
                x self.res[1] x self.res[2] x 3)
        """
        minbound = self.get_min_bound()
        maxbound = self.get_max_bound()
        xs = torch.linspace(minbound[0], maxbound[0], self.res[0] + 1)
        ys = torch.linspace(minbound[1], maxbound[1], self.res[1] + 1)
        zs = torch.linspace(minbound[2], maxbound[2], self.res[2] + 1)
        # import numpy as np
        # xx, yy, zz = np.meshgrid(xs.cpu().numpy(), ys.cpu().numpy(), zs.cpu().numpy(),
        #     sparse=False, indexing='xy')
        # print(xx.shape)
        # grid = torch.from_numpy(grid).to(self.device)
        grid = torch.stack((torch.meshgrid([xs[:-1], ys[:-1], zs[:-1]])))
        grid = grid + 0.5 * self.voxelsize[0]
        return grid.permute(1, 2, 3, 0).unsqueeze(0).to(self.device)
        # return grid.permute(1, 2, 3, 0).unsqueeze(0).to(self.device)

    def voxel_to_world_coords(self):
        r"""Converts voxel coordinates to world coords. """
        if self.voxel_coords is None:
            self.voxel_coords = self.get_voxel_coords()
        return self.origin + (self.voxel_coords + 0.5) * self.voxelsize.unsqueeze(0)
        # return self.transform_voxel_coords(self.voxel_coords, pose)

    def voxel_to_world_coords_selected(self, selected_voxels):
        r"""Converts only a subset of selected voxels to world coordinates. 

        Args:
            selected_voxels (np.ndarray or torch.Tensor): Selected voxels
                to convert to world coordinates (shape: (N, 3))

        Returns:
            world_coords (np.ndarray or torch.Tensor): World coordinates
                of selected voxels (shape: (N, 3))

        """
        import numpy as np

        # Whether we need to map selected_voxels back to numpy.
        back_to_numpy = False
        # If selected_voxels is numpy, convert to torch tensor.
        if isinstance(selected_voxels, np.ndarray):
            selected_voxels = torch.from_numpy(selected_voxels).float()
            back_to_numpy = True
        # Device to restore selected_voxels to.
        restore_device = selected_voxels.device
        # Map it to the same device the other TSDFVolume data is on.
        selected_voxels = selected_voxels.to(self.device)
        # Perform the conversion
        world_coords = self.origin + selected_voxels * self.voxelsize.unsqueeze(0)
        # Restore to original device. (This is an optional step; we just
        # use this to preserve device attributes for each tensor).
        selected_voxels = selected_voxels.to(restore_device)
        world_coords = world_coords.to(restore_device)
        # If the input array was in numpy, map the output as well as the
        # input arrays back to numpy.
        if back_to_numpy:
            selected_voxels = selected_voxels.detach().cpu().numpy()
            world_coords = world_coords.detach().cpu().numpy()
        return world_coords

    @staticmethod
    def transform_voxel_coords(voxels, tform):
        r"""Transforms a tensor containing voxel coordinates by a 4 x 4
        transformation matrix.

        Args:
            voxels (torch.Tensor): Voxel coordinates (shape: ... x 3).
            tform (torch.Tensor): Transform(s) (shape: ... x 4 x 4).

        NOTE: If batchsize of `tform` is > 1, then it is expected that
        the tform broadcasts across voxels (this usually means that
        the first dimensions of `voxels` and `tform` match).

        Returns:
            (torch.Tensor): Transformed voxel coordinates (same shape
            as input `voxels`).

        """

        # voxels should be at least 4D tensor.
        # voxels should have size of last dim equal to 3.
        # tform should have at least 2 dims.
        # tform should have size of last 2 dims equal to 4.

        # If dim 0 of tform is greater than size 1 (and tform
        # has > 2 dims) then voxels should have the same size
        # for dim 0.

        # Homogenize
        voxels_homo = homogenize_points(voxels)

        # Transform
        return transform_pts_nd_KF(voxels, tform)


# TODO: This should go away; into tests
if __name__ == "__main__":

    origin = torch.FloatTensor([2.0, 2.0, 2.0])
    extent = torch.FloatTensor([10.0, 10.0, 10.0])
    grid = VoxelGrid(res=torch.LongTensor([2, 2, 2]), origin=origin, extent=extent)
    print(grid.origin, grid.extent, grid.res, grid.voxelsize)
    print(grid.get_min_bound(), grid.get_max_bound(), grid.get_center())
    voxels = grid.get_voxel_coords()[0]
    # print(voxels)
    # print(grid.voxel_to_world_coords())
