import os
from typing import Optional, Union

import cv2
import imageio
import numpy as np
import torch
from ..geometry.geometryutils import relative_transformation
from torch.utils import data

from . import datautils
from . import tumutils

__all__ = ["TUM"]


class TUM(data.Dataset):
    r"""A torch Dataset for loading in `the TUM dataset <https://vision.in.tum.de/data/datasets/rgbd-dataset>`_.
    Will fetch sequences of rgb images, depth maps, intrinsics matrix, poses, frame to frame relative transformations
    (with first frame's pose as the reference transformation), names of frames. Uses extracted `.tgz` sequences
    downloaded from `here <https://vision.in.tum.de/data/datasets/rgbd-dataset/download>`__.
    Expects similar to the following folder structure for the TUM dataset:

    .. code-block::


        | ├── TUM
        | │   ├── rgbd_dataset_freiburg1_rpy
        | │   │   ├── depth/
        | │   │   ├── rgb/
        | │   │   ├── accelerometer.txt
        | │   │   ├── depth.txt
        | │   │   ├── groundtruth.txt
        | │   │   └── rgb.txt
        | │   ├── rgbd_dataset_freiburg1_xyz
        | │   │   ├── depth/
        | │   │   ├── rgb/
        | │   │   ├── accelerometer.txt
        | │   │   ├── depth.txt
        | │   │   ├── groundtruth.txt
        | │   │   └── rgb.txt
        | │   ├── ...
        |
        |

    Example of sequence creation from frames with `seqlen=4`, `dilation=1`, `stride=3`, and `start=2`:

    .. code-block::


                                            sequence0
                        ┎───────────────┲───────────────┲───────────────┒
                        |               |               |               |
        frame0  frame1  frame2  frame3  frame4  frame5  frame6  frame7  frame8  frame9  frame10  frame11 ...
                                                |               |               |                |
                                                └───────────────┵───────────────┵────────────────┚
                                                                    sequence1

    Args:
        basedir (str): Path to the base directory containing extracted TUM sequences in separate directories.
            Each sequence subdirectory is assumed to contain `depth/`, `rgb/`, `accelerometer.txt`, `depth.txt` and
            `groundtruth.txt` and `rgb.txt`, E.g.:

            .. code-block::


                ├── rgbd_dataset_freiburgX_NAME
                │   ├── depth/
                │   ├── rgb/
                │   ├── accelerometer.txt
                │   ├── depth.txt
                │   ├── groundtruth.txt
                │   └── rgb.txt

        sequences (str or tuple of str or None): Sequences to use from those available in `basedir`.
            Can be path to a `.txt` file where each line is a sequence name (e.g. `rgbd_dataset_freiburg1_rpy`),
            a tuple of sequence names, or None to use all sequences. Default: None
        seqlen (int): Number of frames to use for each sequence of frames. Default: 4
        dilation (int or None): Number of (original trajectory's) frames to skip between two consecutive
            frames in the extracted sequence. See above example if unsure.
            If None, will set `dilation = 0`. Default: None
        stride (int or None): Number of frames between the first frames of two consecutive extracted sequences.
            See above example if unsure. If None, will set `stride = seqlen * (dilation + 1)`
            (non-overlapping sequences). Default: None
        start (int or None): Index of the rgb frame from which to start extracting sequences for every sequence.
            If None, will start from the first frame. Default: None
        end (int): Index of the rgb frame at which to stop extracting sequences for every sequence.
            If None, will continue extracting frames until the end of the sequence. Default: None
        height (int): Spatial height to resize frames to. Default: 480
        width (int): Spatial width to resize frames to. Default: 640
        channels_first (bool): If True, will use channels first representation :math:`(B, L, C, H, W)` for images
            `(batchsize, sequencelength, channels, height, width)`. If False, will use channels last representation
            :math:`(B, L, H, W, C)`. Default: False
        normalize_color (bool): Normalize color to range :math:`[0 1]` or leave it at range :math:`[0 255]`.
            Default: False
        return_depth (bool): Determines whether to return depths. Default: True
        return_intrinsics (bool): Determines whether to return intrinsics. Default: True
        return_pose (bool): Determines whether to return poses. Default: True
        return_transform (bool): Determines whether to return transforms w.r.t. initial pose being transformed to be
            identity. Default: True
        return_names (bool): Determines whether to return sequence names. Default: True
        return_timestamps (bool): Determines whether to return rgb, depth and pose timestamps. Default: True


    Examples::

        >>> dataset = TUM(
            basedir="TUM-data/",
            sequences=("rgbd_dataset_freiburg1_rpy", "rgbd_dataset_freiburg1_xyz"))
        >>> loader = data.DataLoader(dataset=dataset, batch_size=4)
        >>> colors, depths, intrinsics, poses, transforms, names = next(iter(loader))

    """

    def __init__(
        self,
        basedir: str,
        sequences: Union[tuple, str, None] = None,
        seqlen: int = 4,
        dilation: Optional[int] = None,
        stride: Optional[int] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        height: int = 480,
        width: int = 640,
        channels_first: bool = False,
        normalize_color: bool = False,
        *,
        return_depth: bool = True,
        return_intrinsics: bool = True,
        return_pose: bool = True,
        return_transform: bool = True,
        return_names: bool = True,
        return_timestamps: bool = True,
    ):
        super(TUM, self).__init__()

        basedir = os.path.normpath(basedir)
        self.height = height
        self.width = width
        self.height_downsample_ratio = float(height) / 480
        self.width_downsample_ratio = float(width) / 640
        self.channels_first = channels_first
        self.normalize_color = normalize_color

        self.return_depth = return_depth
        self.return_intrinsics = return_intrinsics
        self.return_pose = return_pose
        self.return_transform = return_transform
        self.return_names = return_names
        self.return_timestamps = return_timestamps

        self.load_poses = self.return_pose or self.return_transform

        if not isinstance(seqlen, int):
            raise TypeError('"seqlen" must be int. Got {0}.'.format(type(seqlen)))
        if not (isinstance(stride, int) or stride is None):
            raise TypeError(
                '"stride" must be int or None. Got {0}.'.format(type(stride))
            )
        if not (isinstance(dilation, int) or dilation is None):
            raise TypeError(
                "dilation must be int or None. Got {0}.".format(type(dilation))
            )
        dilation = dilation if dilation is not None else 0
        stride = stride if stride is not None else seqlen * (dilation + 1)
        self.seqlen = seqlen
        self.stride = stride
        self.dilation = dilation
        if seqlen < 0:
            raise ValueError('"seqlen" must be positive. Got {0}.'.format(seqlen))
        if dilation < 0:
            raise ValueError('"dilation" must be positive. Got {0}.'.format(dilation))
        if stride < 0:
            raise ValueError('"stride" must be positive. Got {0}.'.format(stride))

        if not (isinstance(start, int) or start is None):
            raise TypeError('"start" must be int or None. Got {0}.'.format(type(start)))
        if not (isinstance(end, int) or end is None):
            raise TypeError('"end" must be int or None. Got {0}.'.format(type(end)))
        start = start if start is not None else 0
        self.start = start
        self.end = end
        if start is not None and start < 0:
            raise ValueError(
                '"start" must be None or positive. Got {0}.'.format(stride)
            )
        if not (end is None or end > start):
            raise ValueError(
                '"end" ({0}) must be None or greater than start ({1})'.format(
                    end, start
                )
            )

        # preprocess sequences to be a tuple or None
        if isinstance(sequences, str):
            if os.path.isfile(sequences):
                with open(sequences, "r") as f:
                    sequences = tuple(f.read().split("\n"))
            else:
                raise ValueError(
                    "incorrect filename: {} doesn't exist".format(sequences)
                )
        elif not (sequences is None or isinstance(sequences, tuple)):
            msg = '"sequences" should either be path to .txt file or tuple of sequence names or None, '
            msg += " but was of type {0} instead"
            raise TypeError(msg.format(type(sequences)))
        if isinstance(sequences, tuple):
            if len(sequences) == 0:
                raise ValueError(
                    '"sequences" must have atleast one element. Got len(sequences)=0'
                )

        # check if TUM folder structure correct: If sequences is not None, should contain all sequence paths.
        # Should also contain atleast one sequence path.
        sequence_paths = []
        dirmsg = "TUM folder should look something like:\n\n| ├── basedir\n"
        dirmsg += "| │   ├── rgbd_dataset_freiburgX_NAME\n| │   │   ├── depth/\n"
        dirmsg += "| │   │   ├── rgb/\n| │   │   ├── accelerometer.txt\n"
        dirmsg += "| │   │   └── depth.txt\n| │   │   └── groundtruth.txt\n"
        dirmsg += "| │   │   └── rgb.txt\n| │   ├── ..."
        for item in os.listdir(basedir):
            if os.path.isdir(os.path.join(basedir, item)):
                split = item.split("_")
                if (
                    split[0] != "rgbd"
                    or split[1] != "dataset"
                    or split[2][:-1] != "freiburg"
                    or len(split) < 4
                ):
                    msg = 'Incorrect folder names in "basedir" ({0}). '.format(basedir)
                    msg += "Folder names of extracted .tgz files from TUM should follow the following naming "
                    msg += (
                        'convention: "rgbd_dataset_freiburgX_NAME". Got "{0}".'.format(
                            item
                        )
                    )
                    raise ValueError(msg)
                if sequences is None or (sequences is not None and item in sequences):
                    sequence_paths.append(os.path.join(basedir, item))
        if len(sequence_paths) == 0:
            raise ValueError(
                'Incorrect folder structure in basedir ("{0}"). '.format(basedir)
                + dirmsg
            )
        if sequences is not None and len(sequence_paths) != len(sequences):
            msg = '"sequences" contains sequences not available in basedir:\n'
            msg += '"sequences" contains: ' + ", ".join(sequences) + "\n"
            msg += (
                '"basedir" contains: '
                + ", ".join(list(map(os.path.basename, sequence_paths)))
                + "\n"
            )
            raise ValueError(msg.format(basedir) + dirmsg)

        # get association and pose file paths
        rgb_text_files, depth_text_files, pose_text_files = [], [], []
        for sequence_path in sequence_paths:
            rgb_text_file = os.path.join(sequence_path, "rgb.txt")
            if not os.path.isfile(rgb_text_file):
                msg = 'Missing "rgb.txt" file in {0}. '.format(rgb_text_file)
                raise ValueError(msg + dirmsg)
            rgb_text_files.append(rgb_text_file)
            depth_text_file = os.path.join(sequence_path, "depth.txt")
            if not os.path.isfile(depth_text_file):
                msg = 'Missing "depth.txt" file in {0}. '.format(depth_text_file)
                raise ValueError(msg + dirmsg)
            depth_text_files.append(depth_text_file)
            if self.load_poses:
                pose_text_file = os.path.join(sequence_path, "groundtruth.txt")
                if not os.path.isfile(pose_text_file):
                    msg = 'Missing poses file ("groundtruth.txt") in {0}. '.format(
                        pose_text_file
                    )
                    raise ValueError(msg + dirmsg)
                pose_text_files.append(pose_text_file)

        # Get a list of all color, depth, pose, label and intrinsics files.
        colorfiles, depthfiles, poses, framenames = [], [], [], []
        timestamps = []
        idx = np.arange(seqlen) * (dilation + 1)
        for file_num, rgb_text_file in enumerate(rgb_text_files):
            depth_text_file = depth_text_files[file_num]
            pose_text_file = pose_text_files[file_num] if self.load_poses else None
            parentdir = os.path.dirname(rgb_text_file)
            splitpath = rgb_text_file.split(os.sep)
            sequence_name = splitpath[-2]
            if sequences is not None:
                if sequence_name not in sequences:
                    continue

            seq_colorfiles, seq_depthfiles = [], []
            seq_poses, seq_framenames = [], []
            associations, seq_timestamps = self._findAssociations(
                rgb_text_file, depth_text_file, pose_text_file
            )

            for frame_num, association in enumerate(associations):
                msg = "Incorrect reading from TUM associations"
                if association[0][:3] != "rgb":
                    raise ValueError(msg)
                seq_colorfiles.append(
                    os.path.normpath(os.path.join(parentdir, association[0]))
                )
                if association[1][:5] != "depth":
                    raise ValueError(msg)
                seq_depthfiles.append(
                    os.path.normpath(os.path.join(parentdir, association[1]))
                )
                if self.load_poses:
                    seq_poses.append(association[2])
                seq_framenames.append(
                    sequence_name.strip("/\\") + "/" + association[0][3:-4]
                )

            num_frames = len(seq_colorfiles)
            for start_ind in range(0, num_frames, stride):
                if (start_ind + idx[-1]) >= num_frames:
                    break
                inds = start_ind + idx
                colorfiles.append([seq_colorfiles[i] for i in inds])
                depthfiles.append([seq_depthfiles[i] for i in inds])
                framenames.append(", ".join([seq_framenames[i] for i in inds]))
                timestamps.append([seq_timestamps[i] for i in inds])
                if self.load_poses:
                    poses.append([seq_poses[i] for i in inds])

        self.num_sequences = len(colorfiles)

        # Class members to store the list of valid filepaths.
        self.colorfiles = colorfiles
        self.depthfiles = depthfiles
        self.poses = poses
        self.framenames = framenames
        self.timestamps = timestamps

        # Camera intrinsics matrix for TUM dataset
        intrinsics = torch.tensor(
            [[525.0, 0, 319.5, 0], [0, 525.0, 239.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        ).float()
        self.intrinsics = datautils.scale_intrinsics(
            intrinsics, self.height_downsample_ratio, self.width_downsample_ratio
        ).unsqueeze(0)

        # Scaling factor for depth images
        self.scaling_factor = 5000.0

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
            intrinsics (torch.Tensor): Intrinsics for the current sequence
            framename (str): Name of the frame
            timestamp_seq (str): Sequence of timestamps of matched rgb, depth and pose stored
                as "rgb rgb_timestamp depth depth_timestamp pose pose_timestamp\n".

        Shape:
            - color_seq: :math:`(L, H, W, 3)` if `channels_first` is False, else :math:`(L, 3, H, W)`. `L` denotes
                sequence length.
            - depth_seq: :math:`(L, H, W, 1)` if `channels_first` is False, else :math:`(L, 1, H, W)`. `L` denotes
                sequence length.
            - pose_seq: :math:`(L, 4, 4)` where `L` denotes sequence length.
            - transform_seq: :math:`(L, 4, 4)` where `L` denotes sequence length.
            - intrinsics: :math:`(1, 4, 4)`
        """

        # Read in the color, depth, pose, label and intrinstics info.
        color_seq_path = self.colorfiles[idx]
        depth_seq_path = self.depthfiles[idx]
        pose_pointquat_seq = self.poses[idx] if self.load_poses else None
        framename = self.framenames[idx]
        timestamp_seq = self.timestamps[idx]

        color_seq, depth_seq, pose_seq, label_seq = [], [], [], []
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

        if self.load_poses:
            poses = self._homogenPoses(pose_pointquat_seq)
            pose_seq = [torch.from_numpy(pose) for pose in poses]

        output = []
        color_seq = torch.stack(color_seq, 0).float()
        output.append(color_seq)

        if self.return_depth:
            depth_seq = torch.stack(depth_seq, 0).float()
            output.append(depth_seq)

        if self.return_intrinsics:
            intrinsics = self.intrinsics
            output.append(intrinsics)

        if self.return_pose:
            pose_seq = torch.stack(pose_seq, 0).float()
            pose_seq = self._preprocess_poses(pose_seq)
            output.append(pose_seq)

        if self.return_transform:
            transform_seq = datautils.poses_to_transforms(poses)
            transform_seq = [torch.from_numpy(x).float() for x in transform_seq]
            transform_seq = torch.stack(transform_seq, 0).float()
            output.append(transform_seq)

        if self.return_names:
            output.append(framename)

        if self.return_timestamps:
            timestamp_seq = "\n".join(
                ["rgb {} depth {} pose {}".format(*t) for t in timestamp_seq]
            )
            output.append(timestamp_seq)

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

    def _preprocess_poses(self, poses: torch.Tensor):
        r"""Preprocesses the poses by setting first pose in a sequence to identity and computing the relative
        homogenous transformation for all other poses.

        Args:
            poses (torch.Tensor): Pose matrices to be preprocessed

        Returns:
            Output (torch.Tensor): Preprocessed poses

        Shape:
            - poses: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
            - Output: :math:`(L, 4, 4)` where :math:`L` denotes sequence length.
        """
        return relative_transformation(
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1), poses
        )

    def _homogenPoses(self, poses_point_quaternion):
        r"""Converts a list of 3D point unit quaternion poses to a list of homogeneous poses

        Args:
            poses_point_quaternion (list of np.ndarray): List of np.ndarray 3D point unit quaternion
                poses, each of shape :math:`(7,)`.

        Returns:
            list of np.ndarray: List of homogeneous poses in np.ndarray format. Each np.ndarray
                has a shape of :math:`(4, 4)`.
        """
        return [
            datautils.pointquaternion_to_homogeneous(pose)
            for pose in poses_point_quaternion
        ]

    def _findAssociations(
        self,
        rgb_text_file: str,
        depth_text_file: str,
        poses_text_file: Optional[str] = None,
        max_difference: float = 0.02,
    ):
        r"""Associates TUM color images, depth images and (optionally) poses based on un-synchronized time
        stamps and returns associations as tuples.

        Args:
            rgb_text_file (str): Path to "rgb.txt"
            depth_text_file (str): Path to "depth.txt"
            poses_text_file (str or None): Path to ground truth poses ("groundtruth.txt"). Default: None
            max_difference (float): Search radius for candidate generation. Default: 0.02

        Returns:
            associations (list of tuple): List of tuples, each tuple containing rgb frame path,
                depth frame path, and an np.ndarray for 3D point unit quaternion poses of shape :math:`(7,)`
                (rgb_frame_path, depth_frame_path, point_quaternion_npndarray).
            timestamps (list of tuple of str): Timestamps of matched rgb, depth and pose.
                The first dimension corresponds to the number of matches :math:`N`, and the second dimension
                stores the associated timestamps as (rgb_timestamp, depth_timestamp, pose_timestamp).

        """
        rgb_dict = tumutils.read_file_list(rgb_text_file, self.start, self.end)
        depth_dict = tumutils.read_file_list(depth_text_file)
        matches = tumutils.associate(rgb_dict, depth_dict, 0, float(max_difference))

        if poses_text_file is not None:
            poses_dict = tumutils.read_trajectory(poses_text_file, matrix=False)
            matches_dict = {match[1]: match[0] for match in matches}
            matches = tumutils.associate(
                matches_dict, poses_dict, 0, float(max_difference)
            )
            matches = [
                (matches_dict[match[0]], match[0], match[1]) for match in matches
            ]

        if poses_text_file is None:
            associations = [(rgb_dict[m[0]][0], depth_dict[m[1]][0]) for m in matches]
            timestamps = [(m[0], m[1], None) for m in matches]
        else:
            associations = [
                (
                    rgb_dict[m[0]][0],
                    depth_dict[m[1]][0],
                    np.array(poses_dict[m[2]], dtype=np.float32),
                )
                for m in matches
            ]
            timestamps = list(matches)
        return associations, timestamps
