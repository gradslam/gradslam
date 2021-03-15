import os
import warnings
from typing import Optional, Union

import cv2
import imageio
import numpy as np
import torch
from ..geometry.geometryutils import relative_transformation
from torch.utils import data

from . import datautils

__all__ = ["ICL"]


class ICL(data.Dataset):
    r"""A torch Dataset for loading in `the ICL-NUIM dataset <https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html>`_.
    Will fetch sequences of rgb images, depth maps, intrinsics matrix, poses, frame to frame relative transformations
    (with first frame's pose as the reference transformation), names of frames. Uses the
    `TUM RGB-D Compatible PNGs` files and `Global_RT_Trajectory_GT` from
    `here <https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html}{ICL-NUIM dataset>`_. Expects the following folder
    structure for the ICL dataset:

    .. code-block::


        | ├── ICL
        | │   ├── living_room_traj0_frei_png
        | │   │   ├── depth/
        | │   │   ├── rgb/
        | │   │   ├── associations.txt
        | │   │   └── livingRoom0n.gt.sim
        | │   ├── living_room_traj1_frei_png
        | │   │   ├── depth/
        | │   │   ├── rgb/
        | │   │   ├── associations.txt
        | │   │   └── livingRoom1n.gt.sim
        | │   ├── living_room_traj2_frei_png
        | │   │   ├── depth/
        | │   │   ├── rgb/
        | │   │   ├── associations.txt
        | │   │   └── livingRoom2n.gt.sim
        | │   ├── living_room_traj3_frei_png
        | │   │   ├── depth/
        | │   │   ├── rgb/
        | │   │   ├── associations.txt
        | │   │   └── livingRoom3n.gt.sim
        | │   ├── living_room_trajX_frei_png
        | │   │   ├── depth/
        | │   │   ├── rgb/
        | │   │   ├── associations.txt
        | │   │   └── livingRoomXn.gt.sim
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
        basedir (str): Path to the base directory containing the `living_room_trajX_frei_png/` directories from
            ICL-NUIM. Each trajectory subdirectory is assumed to contain `depth/`, `rgb/`, `associations.txt` and
            `livingRoom0n.gt.sim`.

            .. code-block::


                ├── living_room_trajX_frei_png
                │   ├── depth/
                │   ├── rgb/
                │   ├── associations.txt
                │   └── livingRoomXn.gt.sim

        trajectories (str or tuple of str or None): Trajectories to use from "living_room_traj0_frei_png",
            "living_room_traj1_frei_png", "living_room_traj2_frei_png" or "living_room_traj3_frei_png".
            Can be path to a `.txt` file where each line is a trajectory name (`living_room_traj0_frei_png`),
            a tuple of trajectory names, or None to use all trajectories. Default: None
        seqlen (int): Number of frames to use for each sequence of frames. Default: 4
        dilation (int or None): Number of (original trajectory's) frames to skip between two consecutive
            frames in the extracted sequence. See above example if unsure.
            If None, will set `dilation = 0`. Default: None
        stride (int or None): Number of frames between the first frames of two consecutive extracted sequences.
            See above example if unsure. If None, will set `stride = seqlen * (dilation + 1)`
            (non-overlapping sequences). Default: None
        start (int or None): Index of the frame from which to start extracting sequences for every trajectory.
            If None, will start from the first frame. Default: None
        end (int): Index of the frame at which to stop extracting sequences for every trajectory.
            If None, will continue extracting frames until the end of the trajectory. Default: None
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


    Examples::

        >>> dataset = ICL(
            basedir="ICL-data/",
            trajectories=("living_room_traj0_frei_png", "living_room_traj1_frei_png")
            )
        >>> loader = data.DataLoader(dataset=dataset, batch_size=4)
        >>> colors, depths, intrinsics, poses, transforms, names = next(iter(loader))

    """

    def __init__(
        self,
        basedir: str,
        trajectories: Union[tuple, str, None] = None,
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
    ):
        super(ICL, self).__init__()

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

        self.load_poses = self.return_pose or self.return_transform

        if not isinstance(seqlen, int):
            raise TypeError("seqlen must be int. Got {0}.".format(type(seqlen)))
        if not (isinstance(stride, int) or stride is None):
            raise TypeError("stride must be int or None. Got {0}.".format(type(stride)))
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
            raise ValueError("seqlen must be positive. Got {0}.".format(seqlen))
        if dilation < 0:
            raise ValueError('"dilation" must be positive. Got {0}.'.format(dilation))
        if stride < 0:
            raise ValueError("stride must be positive. Got {0}.".format(stride))

        if not (isinstance(start, int) or start is None):
            raise TypeError("start must be int or None. Got {0}.".format(type(start)))
        if not (isinstance(end, int) or end is None):
            raise TypeError("end must be int or None. Got {0}.".format(type(end)))
        start = start if start is not None else 0
        self.start = start
        self.end = end
        if start < 0:
            raise ValueError("start must be positive. Got {0}.".format(stride))
        if not (end is None or end > start):
            raise ValueError(
                "end ({0}) must be None or greater than start ({1})".format(end, start)
            )

        # preprocess trajectories to be a tuple or None
        valid_trajectory_dirs = [
            f
            for f in os.listdir(basedir)
            if os.path.isdir(os.path.join(basedir, f))
            and f[:16] == "living_room_traj"
            and f[-9:] == "_frei_png"
        ]
        if len(valid_trajectory_dirs) == 0:
            msg = "basedir ({0}) should contain trajectory folders with the following naming ".format(
                basedir
            )
            msg += 'convention: "living_room_trajX_frei_png". Found 0 folders with this naming convention.'
            raise ValueError(msg)

        if isinstance(trajectories, str):
            if os.path.isfile(trajectories):
                with open(trajectories, "r") as f:
                    trajectories = tuple(f.read().split("\n"))
                valid_trajectory_dirs = list(trajectories)
            else:
                raise ValueError(
                    "incorrect filename: {} doesn't exist".format(trajectories)
                )
        elif not (trajectories is None or isinstance(trajectories, tuple)):
            msg = '"trajectories" should either be path to .txt file or tuple of trajectory names or None, '
            msg += " but was of type {0} instead"
            raise TypeError(msg.format(type(trajectories)))
        if isinstance(trajectories, tuple):
            if len(trajectories) == 0:
                raise ValueError(
                    '"trajectories" must have atleast one element. Got len(trajectories)=0'
                )
            msg = '"trajectories" should only contain trajectory folder names of the following convention: '
            msg += '"living_room_trajX_frei_png". It contained: {0}.'
            for t in trajectories:
                if not (t[:16] == "living_room_traj" and t[-9:] == "_frei_png"):
                    raise ValueError(msg.format(t))
            valid_trajectory_dirs = list(trajectories)

        # check if ICL folder structure correct: If trajectories is not None, should contain all trajectory paths.
        # Should also contain atleast one trajectory path.
        trajectory_paths = []
        dirmsg = "ICL folder should look something like:\n\n| ├── basedir\n"
        for i in range(4):
            dirmsg += (
                "| │   ├── living_room_traj{0}_frei_png\n| │   │   ├── depth/\n".format(
                    str(i)
                )
            )
            dirmsg += "| │   │   ├── rgb/\n| │   │   ├── associations.txt\n"
            dirmsg += "| │   │   └── livingRoom{0}n.gt.sim\n".format(str(i))
        for item in os.listdir(basedir):
            if (
                os.path.isdir(os.path.join(basedir, item))
                and item in valid_trajectory_dirs
            ):
                trajectory_paths.append(os.path.join(basedir, item))
        if len(trajectory_paths) == 0:
            raise ValueError(
                'Incorrect folder structure in basedir ("{0}"). '.format(basedir)
                + dirmsg
            )
        if trajectories is not None and len(trajectory_paths) != len(trajectories):
            msg = '"trajectories" contains trajectories not available in basedir:\n'
            msg += "trajectories contains: " + ", ".join(trajectories) + "\n"
            msg += (
                "basedir contains: "
                + ", ".join(list(map(os.path.basename, trajectory_paths)))
                + "\n"
            )
            raise ValueError(msg.format(basedir) + dirmsg)

        # get association and pose file paths
        associationsfiles, posesfiles = [], []
        for trajectory_path in trajectory_paths:
            associationsfile = os.path.join(trajectory_path, "associations.txt")
            if not os.path.isfile(associationsfile):
                msg = 'Missing associations file ("associations.txt") in {0}. '.format(
                    trajectory_path
                )
                raise ValueError(msg + dirmsg)
            associationsfiles.append(associationsfile)
            if self.load_poses:
                trajectory_num = trajectory_path[
                    trajectory_path.index("living_room_traj") + 16 :
                ].split("_")[0]
                posesfile = os.path.join(
                    trajectory_path, "livingRoom{0}n.gt.sim".format(trajectory_num)
                )
                if not os.path.isfile(posesfile):
                    msg = 'Missing ground truth poses file ("{0}") in {1}. '.format(
                        posesfile, basedir
                    )
                    raise ValueError(msg + dirmsg)
                posesfiles.append(posesfile)

        # Get a list of all color, depth, pose, label and intrinsics files.
        colorfiles, depthfiles, posemetas, framenames = [], [], [], []
        idx = np.arange(seqlen) * (dilation + 1)
        for file_num, associationsfile in enumerate(associationsfiles):
            parentdir = os.path.dirname(associationsfile)
            splitpath = associationsfile.split(os.sep)
            trajectory_name = splitpath[-2]
            if trajectories is not None:
                if trajectory_name not in trajectories:
                    continue

            traj_colorfiles, traj_depthfiles = [], []
            traj_poselinenums, traj_framenames = [], []
            with open(associationsfile, "r") as f:
                lines = f.readlines()
                if self.end is None:
                    end = len(lines)
                if end > len(lines):
                    msg = "end was larger than number of frames in trajectory: {0} > {1} (trajectory: {2})"
                    warnings.warn(msg.format(end, len(lines), trajectory_name))
                # traj0 is missing a pose in livingRoom0n.gt.sim, thus we remove one of the frames
                if trajectory_name == "living_room_traj0_frei_png":
                    lines = lines[:-1]
                lines = lines[start:end]

            if self.load_poses:
                posesfile = posesfiles[file_num]
                with open(posesfile, "r") as f:
                    len_posesfile = sum(1 for line in f)

            for line_num, line in enumerate(lines):
                line_num += self.start
                line = line.strip().split()
                msg = "incorrect reading from ICL associations"
                if line[3][:3] != "rgb":
                    raise ValueError(msg)
                traj_colorfiles.append(
                    os.path.normpath(os.path.join(parentdir, line[3]))
                )
                if line[1][:5] != "depth":
                    raise ValueError(msg)
                traj_depthfiles.append(
                    os.path.normpath(os.path.join(parentdir, line[1]))
                )

                if self.load_poses:
                    if line_num * 4 > len_posesfile:
                        msg = '{0}th pose should start from line {1} of file "{2}", but said file has only {3} lines.'
                        raise ValueError(
                            msg.format(
                                line_num,
                                line_num * 4,
                                os.path.join(*posesfile.split(os.sep)[-2:]),
                                len_posesfile,
                            )
                        )
                    traj_poselinenums.append(line_num * 4)

                traj_framenames.append(
                    os.path.join(trajectory_name, line[1][6:].split(".")[0])
                )

            traj_len = len(traj_colorfiles)
            for start_ind in range(0, traj_len, stride):
                if (start_ind + idx[-1]) >= traj_len:
                    break
                inds = start_ind + idx
                colorfiles.append([traj_colorfiles[i] for i in inds])
                depthfiles.append([traj_depthfiles[i] for i in inds])
                framenames.append(", ".join([traj_framenames[i] for i in inds]))
                if self.load_poses:
                    posemetas.append(
                        {
                            "file": posesfile,
                            "line_nums": [traj_poselinenums[i] for i in inds],
                        }
                    )

        self.num_sequences = len(colorfiles)

        # Class members to store the list of valid filepaths.
        self.colorfiles = colorfiles
        self.depthfiles = depthfiles
        self.posemetas = posemetas
        self.framenames = framenames

        # Camera intrinsics matrix for ICL dataset
        intrinsics = torch.tensor(
            [[481.20, 0, 319.5, 0], [0, -480.0, 239.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
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
        pose_seq_meta = self.posemetas[idx] if self.load_poses else None
        framename = self.framenames[idx]

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
            poses = self._loadPoses(pose_seq_meta["file"], pose_seq_meta["line_nums"])
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
            poses[0].unsqueeze(0).repeat(poses.shape[0], 1, 1),
            poses,
            orthogonal_rotations=False,
        )

    def _loadPoses(self, pose_path, start_lines):
        r"""Loads poses from groundtruth pose text files and returns the poses
        as a list of numpy arrays.

        Args:
            pose_path (str): The path to groundtruth pose text file.
            start_lines (list of ints):

        Returns:
            poses (list of np.array): List of ground truth poses in
                    np.array format. Each np.array has a shape of [4, 4] if
                    homogen_coord is True, or a shape of [3, 4] otherwise.
        """
        pose = []
        poses = []
        parsing_pose = False
        with open(pose_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if not (i in start_lines or parsing_pose):
                continue
            parsing_pose = True
            line = line.strip().split()
            if len(line) != 4:
                msg = "Faulty poses file: Expected line {0} of the poses file {1} to contain pose matrix values, "
                msg += 'but it didn\'t. You can download "Global_RT_Trajectory_GT" from here:\n'
                msg += "https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html"
                raise ValueError(msg)
            pose.append(line)

            if len(pose) == 3:
                pose.append([0.0, 0.0, 0.0, 1.0])
                poses.append(np.array(pose, dtype=np.float32))
                pose = []
                parsing_pose = False

        return poses
