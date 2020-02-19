import cv2
import glob
import imageio
import os
from natsort import natsorted
import numpy as np
import torch
from torch.utils import data
from typing import Optional


class ScannetDataset(data.Dataset):
    r"""Dataset class for the ScanNet dataset.

    Args:
        - basedir (str): Path to the base directory containing a sequence
            from ScanNet. This directory is assumed to contain the unzipped
            rgb and depth images, and the camera calibration parameters.
        - start (int, Optional): Index of the frame from which to start
            (default: 0).
        - end (int, Optional): Index of the frame at which to end
            (default: -1)

    """ 

    def __init__(self, basedir: str, start: int = 0, end: int = -1):
        super(ScannetDataset, self).__init__()
        
        # Lists to store paths to color images, depth images, and IMU poses
        self.path_color = []
        self.path_depth = []
        self.poses = []

        # Start and end frames. Used to determine sequence length.
        self.start = start
        self.end = end
        assert start >= 0, 'Start frame cannot be less than 0.'
        if end != -1:
            assert end > start, 'Invalid start, end'
        self.seqlen = self.end - self.start

        # Get a list of all color, depth, and pose files.
        colorfiles = glob.glob(os.path.join(basedir, 'color', '*.jpg'))
        depthfiles = glob.glob(os.path.join(basedir, 'depth', '*.png'))
        posefiles = glob.glob(os.path.join(basedir, 'pose', '*.txt'))
        colorfiles = natsorted(colorfiles)
        depthfiles = natsorted(depthfiles)
        posefiles = natsorted(posefiles)

        # Ensure that seqlen is less than the number of total files.
        assert self.seqlen <= len(colorfiles), 'Invalid sequence length.'

        # Class members to store the list of valid filepaths.
        self.colorfiles = colorfiles[self.start:self.end+1]
        self.depthfiles = depthfiles[self.start:self.end+1]
        self.posefiles = posefiles[self.start:self.end+1]

        # Open the metadata (scene****_**.txt) file to read camera intrinsics.
        f = open(glob.glob(os.path.join(basedir, 'scene*.txt'))[0], 'r')
        lines = f.readlines()
        f.close()

        # ScanNet color images are 968 x 1296. Compute scale factors sx, sy
        # that map the number of rows and columns of these color images to
        # 640, 480 respectively.
        sx = 640 / 1296.
        sy = 480 / 968.
        # Use these scale factors to update the intrinsics.
        self.fx = sx * float(lines[6].strip().split()[2])
        self.fy = sy * float(lines[8].strip().split()[2])
        self.cx = sx * float(lines[10].strip().split()[2])
        self.cy = sy * float(lines[12].strip().split()[2])
        # Cache the scale factors, for later use.
        self.sx = sx
        self.sy = sy

        # Scaling factor for depth images
        self.scaling_factor = 1000.

    def __len__(self):
        r"""Return the length of the dataset. """
        return self.seqlen

    def _preprocess_color(self, color):
        r"""Preprocesses the color image by resizing to target size (640, 480).

        Args:
            color (np.array): Color image to be preprocessed

        """
        color = cv2.resize(color, (640, 480), interpolation=cv2.INTER_NEAREST)
        color = ScannetDataset._normalize_image_intensities(color)
        color = ScannetDataset._swap_axes(
            ScannetDataset._channels_first(color), 1, 2)
        return color

    def _preprocess_depth(self, depth):
        r"""Preprocesses the depth image by scaling depth values to metres. 
        
        Args:
            depth (np.array): Depth image to be preprocessed.

        """
        return depth.astype(float) / self.scaling_factor

    @staticmethod
    def _channels_first(rgb):
        """Brings channels from the last dim of the array/tensor to the first.

        Args:
            rgb (torch.Tensor or np.ndarray): W x H x C ordering (width, height, channels)

        Returns:
            (torch.Tensor or np.ndarray): C x W x H ordering

        """
        if type(rgb) == np.ndarray:
            assert rgb.ndim == 3, 'Input array must contain exactly 3 dimensions.'
            if rgb.shape[0] < rgb.shape[2]:
                print('Are you sure you are passing the right input? Number of \
                    channels seem to exceed the height of the image.')
            return np.swapaxes(rgb, 0, 2)
        elif type(rgb) == torch.tensor:
            assert rgb.ndim() == 3, 'Input tensor must contain exactly 3 dimensions.'
            if rgb.shape[0] < rgb.shape[2]:
                print('Are you sure you are passing the right input? Number of \
                    channels seem to exceed the height of the image.')
            return rgb.transpose(0, 2)

    @staticmethod
    def _normalize_image_intensities(rgb):
        r"""Normalizes the RGB values of an image so that they lie in the range [0, 1]
        as opposed to [0, 255].

        Args:
            rgb (torch.Tensor or np.ndarray): RGB image (3 channels, C x W x H ordering)

        Returns:
            (torch.Tensor or np.ndarray): RGB image (normalized such that values are in
                the range [0, 1])
        """
        if torch.is_tensor(rgb):
            return rgb.float() / 255
        else:
            return rgb.astype(float) / 255

    @staticmethod
    def _swap_axes(img, ax1=0, ax2=1):
        """Swaps the axes ax1 and ax2 of img. """
        if type(img) == np.ndarray:
            return np.swapaxes(img, ax1, ax2)
        elif type(img) == torch.tensor:
            return img.transpose(ax1, ax2)

    def __getitem__(self, idx):
        r"""Returns the item at index idx. """

        # Read in the color, depth, and pose info.
        color1 = np.asarray(imageio.imread(self.colorfiles[idx]), dtype=float)
        color1 = self._preprocess_color(color1)
        color1 = torch.from_numpy(color1)
        color2 = np.asarray(imageio.imread(self.colorfiles[idx+1]),
            dtype=float)
        color2 = self._preprocess_color(color2)
        color2 = torch.from_numpy(color2)
        depth1 = np.asarray(imageio.imread(self.depthfiles[idx]),
            dtype=np.int64)
        depth1 = self._preprocess_depth(depth1)
        depth1 = torch.from_numpy(depth1)
        depth2 = np.asarray(imageio.imread(self.depthfiles[idx+1]),
            dtype=np.int64)
        depth2 = self._preprocess_depth(depth2)
        depth2 = torch.from_numpy(depth2)
        pose = np.loadtxt(self.posefiles[idx])
        pose = torch.from_numpy(pose)
        
        return color1, depth1, color2, depth2, pose


if __name__ == '__main__':

    basedir = '/home/jatavalk/data/scannet/scene0000_00'
    data = ScannetDataset(basedir, 0, 2)
    print(len(data))
