import glob
import os
from collections import OrderedDict
from typing import Optional, Union

import cv2
import imageio
import numpy as np
import torch
from ..geometry.geometryutils import relative_transformation
from natsort import natsorted
from torch.utils import data

from . import datautils

__all__ = ["Scannet"]


class Scannet(data.Dataset):
    r"""A torch Dataset for loading in `the Scannet dataset <http://www.scan-net.org/>`_. Will fetch sequences of
    rgb images, depth maps, intrinsics matrices, poses, frame to frame relative transformations (with first frame's
    pose as the reference transformation), names of sequences, and semantic segmentation labels.
    Args:
        basedir (str): Path to the base directory containing the `sceneXXXX_XX/` directories from ScanNet. Each scene
            subdirectory is assumed to contain `color/`, `depth/`, `intrinsic/`, `label-filt/` and `pose/` directories.
        seqmetadir (str): Path to directory containing sequence associations. Directory is assumed to contain
            metadata `.txt` files (one metadata per sequence): e.g. `sceneXXXX_XX-seq_Y.txt` .
        scenes (str or tuple of str): Scenes to use from sequences (used for creating train/val/test splits). Can
            be path to a `.txt` file where each line is a scene name (`sceneXXXX_XX`), a tuple of scene names, or None
            to use all scenes.
        start (int): Index of the frame from which to start for every sequence. Default: 0
        end (int): Index of the frame at which to end for every sequence. Default: -1
        height (int): Spatial height to resize frames to. Default: 480
        width (int): Spatial width to resize frames to. Default: 640
        seg_classes (str): The palette of classes that the network should learn. Either `"nyu40"` or `"scannet20"`.
            Default: `"scannet20"`
        channels_first (bool): If True, will use channels first representation :math:`(B, L, C, H, W)` for images
            `(batchsize, sequencelength, channels, height, width)`. If False, will use channels last representation
            :math:`(B, L, H, W, C)`. Default: False
        normalize_color (bool): Normalize color to range :math:`[0, 1]` or leave it at range :math:`[0, 255]`.
            Default: False
        return_depth (bool): Determines whether to return depths. Default: True
        return_intrinsics (bool): Determines whether to return intrinsics. Default: True
        return_pose (bool): Determines whether to return poses. Default: True
        absolute_pose (bool): Determines whether to return absolute poses or inverse transformed poses. Default: True
        return_transform (bool): Determines whether to return transforms w.r.t. initial pose being transformed to be
            identity. Default: True
        return_names (bool): Determines whether to return sequence names. Default: True
        return_labels (bool): Determines whether to return segmentation labels. Default: True
    Examples::
        >>> dataset = Scannet(
            basedir="ScanNet-gradSLAM/extractions/scans/",
            seqmetadir="ScanNet-gradSLAM/extractions/sequence_associations/",
            scenes=("scene0000_00", "scene0001_00")
            )
        >>> loader = data.DataLoader(dataset=dataset, batch_size=4)
        >>> colors, depths, intrinsics, poses, transforms, names, labels = next(iter(loader))
    """

    def __init__(
        self,
        basedir: str,
        seqmetadir: str,
        scenes: Union[tuple, str, None],
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        height: int = 480,
        width: int = 640,
        seg_classes: str = "scannet20",
        channels_first: bool = False,
        normalize_color: bool = False,
        *,
        return_depth: bool = True,
        return_intrinsics: bool = True,
        return_pose: bool = True,
        absolute_pose: bool = True,
        return_transform: bool = True,
        return_names: bool = True,
        return_labels: bool = True,
    ):
        super(Scannet, self).__init__()

        basedir = os.path.normpath(basedir)
        self.height = height
        self.width = width
        self.height_downsample_ratio = float(height) / 480
        self.width_downsample_ratio = float(width) / 640
        self.seg_classes = seg_classes
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.return_depth = return_depth
        self.return_intrinsics = return_intrinsics
        self.return_pose = return_pose
        self.absolute_pose = absolute_pose
        self.return_transform = return_transform
        self.return_names = return_names
        self.return_labels = return_labels

        self.color_encoding = get_color_encoding(self.seg_classes)

        # Start and end frames. Used to determine sequence length.
        self.start = start
        self.end = end
        full_sequence = self.end == -1
        if start < 0:
            raise ValueError("Start frame cannot be less than 0.")
        if not (end == -1 or end > start):
            raise ValueError(
                "End frame ({}) should be equal to -1 or greater than start ({})".format(
                    end, start
                )
            )
        self.seqlen = self.end - self.start

        # scenes should be a tuple
        if isinstance(scenes, str):
            if os.path.isfile(scenes):
                with open(scenes, "r") as f:
                    scenes = tuple(f.read().split("\n"))
            else:
                raise ValueError("incorrect filename: {} doesn't exist".format(scenes))
        elif not (scenes is None or isinstance(scenes, tuple)):
            msg = "scenes should either be path to split.txt or tuple of scenes or None, but was of type %r instead"
            raise TypeError(msg % type(scenes))

        # Get a list of all color, depth, pose, label and intrinsics files.
        colorfiles, depthfiles, posefiles = [], [], []
        labelfiles, intrinsicsfiles, seqnames = [], [], []
        seqmetapaths = natsorted(glob.glob(os.path.join(seqmetadir, "*.txt")))
        for seqmetapath in seqmetapaths:
            scene_name = os.path.basename(seqmetapath).split("-")[0]

            if scenes is not None:
                if scene_name not in scenes:
                    continue

            seq_colorfiles, seq_depthfiles, seq_posefiles = [], [], []
            seq_labelfiles, seq_intrinsicsfiles = [], []
            with open(seqmetapath, "r") as f:
                lines = f.readlines()
                if full_sequence:
                    self.end = len(lines)
                    self.seqlen = self.end - self.start
                if self.seqlen > len(lines):
                    msg = "sequence length can't be larger than dataset sequence length but it was: %r > %r"
                    raise ValueError(msg % (self.seqlen, len(lines)))
                lines = lines[self.start : self.end]

            for line in lines:
                line = line.strip().split()
                msg = "incorrect reading from scannet metadata"
                if line[0] != "color":
                    raise ValueError(msg)
                seq_colorfiles.append(os.path.join(basedir, line[1]))
                if line[2] != "depth":
                    raise ValueError(msg)
                seq_depthfiles.append(os.path.join(basedir, line[3]))
                if line[4] != "pose":
                    raise ValueError(msg)
                seq_posefiles.append(os.path.join(basedir, line[5]))
                if line[6] != "label-filt":
                    raise ValueError(msg)
                seq_labelfiles.append(os.path.join(basedir, line[7]))
                if line[14] != "intrinsic_depth":
                    raise ValueError(msg)
                seq_intrinsicsfiles.append(os.path.join(basedir, line[15]))

            colorfiles.append(seq_colorfiles)
            depthfiles.append(seq_depthfiles)
            posefiles.append(seq_posefiles)
            labelfiles.append(seq_labelfiles)
            intrinsicsfiles.append(seq_intrinsicsfiles[0])
            seqnames.append(os.path.basename(seqmetapath).split(".")[0])

        self.num_sequences = len(colorfiles)

        # Class members to store the list of valid filepaths.
        self.colorfiles = colorfiles
        self.depthfiles = depthfiles
        self.posefiles = posefiles
        self.labelfiles = labelfiles
        self.intrinsicsfiles = intrinsicsfiles
        self.seqnames = seqnames

        # Scaling factor for depth images
        self.scaling_factor = 1000.0

    def __len__(self):
        r"""Returns the length of the dataset. """
        return self.num_sequences

    def __getitem__(self, idx: int):
        r"""Returns the data from the sequence at index idx.
        Returns:
            color_seq (torch.Tensor): Sequence of rgb images of each frame
            depth_seq (torch.Tensor): Sequence of depths of each frame
            pose_seq (torch.Tensor): Sequence of poses of each frame
            transform_seq (torch.Tensor): Sequence of transformations between each frame in the sequence and the
                previous frame. Transformations are w.r.t. the first frame in the sequence having identity pose
                (relative transformations with first frame's pose as the reference transformation). First
                transformation in the sequence will always be `torch.eye(4)`.
            label_seq (torch.Tensor): Sequence of semantic segmentation labels
            intrinsics (torch.Tensor): Intrinsics for the current sequence
            seqname (str): Name of the sequence
        Shape:
            - color_seq: :math:`(L, H, W, 3)` if `channels_first` is False, else :math:`(L, 3, H, W)`. `L` denotes
                sequence length.
            - depth_seq: :math:`(L, H, W, 1)` if `channels_first` is False, else :math:`(L, 1, H, W)`. `L` denotes
                sequence length.
            - pose_seq: :math:`(L, 4, 4)` where `L` denotes sequence length.
            - transform_seq: :math:`(L, 4, 4)` where `L` denotes sequence length.
            - label_seq: :math:`(L, H, W)` where `L` denotes sequence length.
            - intrinsics: :math:`(1, 4, 4)`
        """

        # Read in the color, depth, pose, label and intrinstics info.
        color_seq_path = self.colorfiles[idx]
        depth_seq_path = self.depthfiles[idx]
        pose_seq_path = self.posefiles[idx]
        label_seq_path = self.labelfiles[idx]
        intrinsics_path = self.intrinsicsfiles[idx]
        seqname = self.seqnames[idx]

        color_seq, depth_seq, pose_seq, label_seq = [], [], [], []
        poses = []
        for i in range(self.seqlen):
            color = np.asarray(imageio.imread(color_seq_path[i]), dtype=float)
            color = self._preprocess_color(color)
            color = torch.from_numpy(color)
            color_seq.append(color)

            if self.return_depth:
                depth = np.asarray(imageio.imread(depth_seq_path[i]), dtype=np.int64)
                depth = self._preprocess_depth(depth)
                depth = torch.from_numpy(depth)
                depth_seq.append(depth)

            if self.return_pose or self.return_transform:
                pose = np.loadtxt(pose_seq_path[i]).astype(float)
                poses.append(pose)
                pose = torch.from_numpy(pose)
                pose_seq.append(pose)

            if self.return_labels:
                label = np.asarray(imageio.imread(label_seq_path[i]), dtype=np.uint8)
                label = self._preprocess_label(label)
                label = torch.from_numpy(label)
                label_seq.append(label)

        output = []
        color_seq = torch.stack(color_seq, 0).float()
        output.append(color_seq)

        if self.return_depth:
            depth_seq = torch.stack(depth_seq, 0).float()
            output.append(depth_seq)

        if self.return_intrinsics:
            intrinsics = np.loadtxt(intrinsics_path).astype(float)
            intrinsics = self._preprocess_intrinsics(intrinsics)
            intrinsics = torch.from_numpy(intrinsics).float()
            output.append(intrinsics)

        if self.return_pose:
            pose_seq = torch.stack(pose_seq, 0).float()
            if not self.absolute_pose:
                pose_seq = self._preprocess_poses(pose_seq)
            output.append(pose_seq)

        if self.return_transform:
            transform_seq = datautils.poses_to_transforms(poses)
            transform_seq = [torch.from_numpy(x).float() for x in transform_seq]
            transform_seq = torch.stack(transform_seq, 0).float()
            output.append(transform_seq)

        if self.return_names:
            output.append(seqname)

        if self.return_labels:
            label_seq = torch.stack(label_seq, 0).float()
            output.append(label_seq)

        return tuple(output)

    def _preprocess_color(self, color: np.ndarray):
        r"""Preprocesses the color image by resizing to :math:`(H, W, C)`, (optionally) normalizing values to
        :math:`[0, 1]`, and (optionally) using channels first :math:`(C, H, W)` representation.
        Args:
            color (np.ndarray): Raw input rgb image
        Retruns:
            np.ndarray: Preprocessed rgb image
        Shape:
            - Input: :math:`(H_\text{old}, W_\text{old}, C)`
            - Output: :math:`(H, W, C)` if `self.channels_first == False`, else :math:`(C, H, W)`.
        """
        color = cv2.resize(
            color, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        )
        if self.normalize_color:
            color = datautils.normalize_image(color)
        if self.channels_first:
            color = datautils.channels_first(color)
        return color

    def _preprocess_depth(self, depth: np.ndarray):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.
        Args:
            depth (np.ndarray): Raw depth image
        Returns:
            np.ndarray: Preprocessed depth
        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (self.width, self.height),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, -1)
        if self.channels_first:
            depth = datautils.channels_first(depth)
        return depth / self.scaling_factor

    def _preprocess_intrinsics(self, intrinsics: Union[torch.Tensor, np.ndarray]):
        r"""Preprocesses the intrinsics by scaling `fx`, `fy`, `cx`, `cy` based on new frame size and expanding the
        0-th dimension.
        Args:
            intrinsics (torch.Tensor or np.ndarray): Intrinsics matrix to be preprocessed
        Returns:
            Output (torch.Tensor or np.ndarray): Preprocessed intrinsics
        Shape:
            - intrinsics: :math:`(4, 4)`
            - Output: :math:`(1, 4, 4)`
        """
        scaled_intrinsics = datautils.scale_intrinsics(
            intrinsics, self.height_downsample_ratio, self.width_downsample_ratio
        )
        if torch.is_tensor(scaled_intrinsics):
            return scaled_intrinsics.unsqueeze(0)
        elif isinstance(scaled_intrinsics, np.ndarray):
            return np.expand_dims(scaled_intrinsics, 0)

    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by transforming all of them such that the initial pose will be identity.
        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed
        Returns:
            Output (torch.Tensor): Poses relative to the initial frame
        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1), poses
        )

    def _preprocess_label(self, label: np.ndarray):
        r"""Preprocesses the "nyu40" label image by resizing it and (optionally) converting to "scannet20" labels
        Args:
            label (np.ndarray): "nyu40" label image with `uint8` values
        Returns:
            np.ndarray: Preprocessed labels
        Shape:
            - label: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W)`
        """
        label = cv2.resize(
            label, (self.width, self.height), interpolation=cv2.INTER_NEAREST
        )
        if self.seg_classes.lower() == "scannet20":
            label = nyu40_to_scannet20(label)
        label = np.expand_dims(label, -1)
        return label


def get_color_encoding(seg_classes):
    r"""Gets the color palette for different sets of labels (`"nyu40"` or `"scannet20"`)
    Args:
        seg_classes (str): Determines whether to use `"nyu40"` labels or `"scannet20"`
    Returns:
        Output (OrderedDict): Label names as keys and color palettes as values.
    """
    if seg_classes.lower() == "nyu40":
        # Color palette for nyu40 labels
        return OrderedDict(
            [
                ("unlabeled", (0, 0, 0)),
                ("wall", (174, 199, 232)),
                ("floor", (152, 223, 138)),
                ("cabinet", (31, 119, 180)),
                ("bed", (255, 187, 120)),
                ("chair", (188, 189, 34)),
                ("sofa", (140, 86, 75)),
                ("table", (255, 152, 150)),
                ("door", (214, 39, 40)),
                ("window", (197, 176, 213)),
                ("bookshelf", (148, 103, 189)),
                ("picture", (196, 156, 148)),
                ("counter", (23, 190, 207)),
                ("blinds", (178, 76, 76)),
                ("desk", (247, 182, 210)),
                ("shelves", (66, 188, 102)),
                ("curtain", (219, 219, 141)),
                ("dresser", (140, 57, 197)),
                ("pillow", (202, 185, 52)),
                ("mirror", (51, 176, 203)),
                ("floormat", (200, 54, 131)),
                ("clothes", (92, 193, 61)),
                ("ceiling", (78, 71, 183)),
                ("books", (172, 114, 82)),
                ("refrigerator", (255, 127, 14)),
                ("television", (91, 163, 138)),
                ("paper", (153, 98, 156)),
                ("towel", (140, 153, 101)),
                ("showercurtain", (158, 218, 229)),
                ("box", (100, 125, 154)),
                ("whiteboard", (178, 127, 135)),
                ("person", (120, 185, 128)),
                ("nightstand", (146, 111, 194)),
                ("toilet", (44, 160, 44)),
                ("sink", (112, 128, 144)),
                ("lamp", (96, 207, 209)),
                ("bathtub", (227, 119, 194)),
                ("bag", (213, 92, 176)),
                ("otherstructure", (94, 106, 211)),
                ("otherfurniture", (82, 84, 163)),
                ("otherprop", (100, 85, 144)),
            ]
        )
    elif seg_classes.lower() == "scannet20":
        # Color palette for scannet20 labels
        return OrderedDict(
            [
                ("unlabeled", (0, 0, 0)),
                ("wall", (174, 199, 232)),
                ("floor", (152, 223, 138)),
                ("cabinet", (31, 119, 180)),
                ("bed", (255, 187, 120)),
                ("chair", (188, 189, 34)),
                ("sofa", (140, 86, 75)),
                ("table", (255, 152, 150)),
                ("door", (214, 39, 40)),
                ("window", (197, 176, 213)),
                ("bookshelf", (148, 103, 189)),
                ("picture", (196, 156, 148)),
                ("counter", (23, 190, 207)),
                ("desk", (247, 182, 210)),
                ("curtain", (219, 219, 141)),
                ("refrigerator", (255, 127, 14)),
                ("showercurtain", (158, 218, 229)),
                ("toilet", (44, 160, 44)),
                ("sink", (112, 128, 144)),
                ("bathtub", (227, 119, 194)),
                ("otherfurniture", (82, 84, 163)),
            ]
        )


def nyu40_to_scannet20(label):
    r"""Remaps a label image from the `"nyu40"` class palette to the `"scannet20"` class palette"""

    # Ignore indices 13, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26. 27. 29. 30. 31. 32, 35. 37. 38, 40
    # Because, these classes from 'nyu40' are absent from 'scannet20'. Our label files are in
    # 'nyu40' format, hence this 'hack'. To see detailed class lists visit:
    # http://kaldir.vc.in.tum.de/scannet_benchmark/labelids_all.txt ('nyu40' labels)
    # http://kaldir.vc.in.tum.de/scannet_benchmark/labelids.txt ('scannet20' labels)
    # The remaining labels are then to be mapped onto a contiguous ordering in the range [0,20]

    # The remapping array comprises tuples (src, tar), where 'src' is the 'nyu40' label, and 'tar' is the
    # corresponding target 'scannet20' label
    remapping = [
        (0, 0),
        (13, 0),
        (15, 0),
        (17, 0),
        (18, 0),
        (19, 0),
        (20, 0),
        (21, 0),
        (22, 0),
        (23, 0),
        (25, 0),
        (26, 0),
        (27, 0),
        (29, 0),
        (30, 0),
        (31, 0),
        (32, 0),
        (35, 0),
        (37, 0),
        (38, 0),
        (40, 0),
        (14, 13),
        (16, 14),
        (24, 15),
        (28, 16),
        (33, 17),
        (34, 18),
        (36, 19),
        (39, 20),
    ]
    for src, tar in remapping:
        label[np.where(label == src)] = tar
    return label
