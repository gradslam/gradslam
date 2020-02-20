import torch

from .voxel import VoxelGrid


class TSDFVolume(VoxelGrid):
    r"""Class to store a TSDF volume. Builds on the VoxelGrid class. """

    def __init__(self, trunc_dist=None, **kwargs):
        r"""Initializes a `TSDFVolume` object with the extents and
        resolution specified.

        Args:
            trunc_dist (float): Truncation value to use, to cap the signed
                distance function (SDF). (default: None)

        **kwargs:
            origin (torch.FloatTensor): Origin of the voxel grid (assumed to
                be at the center of the grid) in (x, y, z) coordinates.
            extent (torch.FloatTensor): Extent of each dimension of the voxel
                grid (x, y, and z respectively).
            res (torch.LongTensor): Number of voxels to be used to represent
                the x, y, and z axes respectively.
            device (str, optional): Device to store the voxel grid on (default:
                'cpu') (choices: ['cpu', 'gpu'])

        """

        super(TSDFVolume, self).__init__(**kwargs)

        # Set truncation distance
        if trunc_dist is not None:
            self.trunc_dist = trunc_dist
        else:
            # Set default truncation distance to 5 * voxelsize[0].
            # NOTE: We're only using voxelsize[0], which implies that
            # we assume an isotropic voxelsize across all dims.
            self.trunc_dist = 5 * self.voxelsize[0]

        # Tensors to store TSDF value and voxel weights (confidences).
        # Ensure that self.voxel_coords is initialized first
        if not self.voxel_coords:
            self.voxel_coords = self.get_voxel_coords()
        # Voxel coords is of shape (1, self.res[0], self.res[1], 
        # self.res[2], 3). TSDF volume and weights are of shape
        # (1, self.res[0], self.res[1], self.res[2])
        self.tsdf = self.trunc_dist * torch.ones(*self.voxel_coords.shape[:-1]).to(self.device)
        # self.tsdf = self.trunc_dist * torch.ones(*self.voxel_coords.shape[:-1]).to(self.device)
        self.weights = torch.zeros(*self.voxel_coords.shape[:-1]).to(
            self.device)

        # Tensor to store validity mask (indicates whether or not a
        # voxel gridcell is valid).
        self.valid = torch.zeros_like(self.voxel_coords).byte()

        # Flatten the TSDF (for (un)ease of indexing)
        self.tsdf = self.tsdf.reshape(-1, 1)
        # Flatten the weights (for (un)ease of indexing)
        self.weights = self.weights.reshape(-1, 1)

        # Tensor to store color information at each voxel
        # Flatten the colors (for (un)ease of indexing).
        self.r = torch.zeros_like(self.tsdf)
        self.g = torch.zeros_like(self.tsdf)
        self.b = torch.zeros_like(self.tsdf)

        # Bit to store whether TSDF has been initialized
        self.initialized = False


    def reset(self):
        r"""Resets the TSDFVolume, weights, and colors.

        """
        self.tsdf = self.trunc_dist * self.tsdf.fill_(1.)
        self.weights = self.weights.fill_(0.)
        self.r = self.r.fill_(0)
        self.g = self.g.fill_(0)
        self.b = self.b.fill_(0)


if __name__ == '__main__':

    vol = TSDFVolume()
    print(vol.valid)
