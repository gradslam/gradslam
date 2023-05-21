import os
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import torch
import tyro
from tqdm import trange
from typing_extensions import Literal

from gradslam.datasets import (
    ICLDataset,
    ReplicaDataset,
    ScannetDataset,
    load_dataset_config,
)
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds
from gradslam.structures.rgbdimages import RGBDImages


@dataclass
class ProgramArgs:
    # Torch device to run computation on (E.g., "cpu")
    device: str = "cuda"

    # gradslam mode ("incremental" vs "batch")
    mode: Literal["incremental", "batch"] = "incremental"

    # Path to the data config (.yaml) file
    dataconfig_path: str = "dataconfigs/icl.yaml"
    # Path to the dataset directory
    data_dir: str = "/path/to/icl/base/dir"
    # Sequence from the dataset to load
    sequence: str = "living_room_traj1_frei_png"
    # Start frame index
    start_idx: int = 0
    # End frame index
    end_idx: int = 800
    # Stride (number of frames to skip between successive fusion steps)
    stride: int = 50
    # Desired image width and height
    desired_height: int = 240
    desired_width: int = 320


def get_dataset(dataconfig_path, basedir, sequence, **kwargs):
    config_dict = load_dataset_config(dataconfig_path)
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def run_batch_slam(dataset, device="cuda"):
    """
    Load and run SLAM (batch mode)
    """

    colors, depths, poses = [], [], []
    intrinsics = None
    print("Loading data...")
    for idx in trange(len(dataset)):
        _color, _depth, intrinsics, _pose, *_ = dataset[idx]
        colors.append(_color)
        depths.append(_depth)
        poses.append(_pose)

    colors = torch.stack(colors)
    depths = torch.stack(depths)
    poses = torch.stack(poses)
    colors = colors.unsqueeze(0)
    depths = depths.unsqueeze(0)
    intrinsics = intrinsics.unsqueeze(0).unsqueeze(0)
    poses = poses.unsqueeze(0)
    colors = colors.float()
    depths = depths.float()
    intrinsics = intrinsics.float()
    poses = poses.float()

    # create rgbdimages object
    rgbdimages = RGBDImages(
        colors,
        depths,
        intrinsics,
        poses,
        channels_first=False,
        # has_embeddings=False,  # KM
    )

    # SLAM
    print("Running batch SLAM...")
    slam = PointFusion(odom="gt", dsratio=1, device=device)  # , use_embeddings=False)
    pointclouds, recovered_poses = slam(rgbdimages)

    print(pointclouds.colors_padded.shape)
    pcd = pointclouds.open3d(0)
    o3d.visualization.draw_geometries([pcd])


def run_incremental_slam(dataset, device="cuda"):
    """
    Load and run SLAM (incremental mode) -- load one frame at a time
    """

    # SLAM
    slam = PointFusion(odom="gt", dsratio=1, device=device)  # , use_embeddings=False)
    pointclouds = Pointclouds(device=device)

    colors, depths, poses = [], [], []
    intrinsics = None
    frame_cur, frame_prev = None, None
    print("Running SLAM...")
    for idx in trange(len(dataset)):
        _color, _depth, intrinsics, _pose, *_ = dataset[idx]
        frame_cur = RGBDImages(
            _color.unsqueeze(0).unsqueeze(0),
            _depth.unsqueeze(0).unsqueeze(0),
            intrinsics.unsqueeze(0).unsqueeze(0).float(),
            _pose.unsqueeze(0).unsqueeze(0).float(),
        )
        pointclouds, _ = slam.step(pointclouds, frame_cur, frame_prev)

    pcd = pointclouds.open3d(0)
    o3d.visualization.draw_geometries([pcd])


def main():
    args = tyro.cli(ProgramArgs)
    # dataconfig = load_dataset_config(args.dataconfig_path)
    dataset = get_dataset(
        dataconfig_path=args.dataconfig_path,
        basedir=args.data_dir,
        sequence=args.sequence,
        start=args.start_idx,
        end=args.end_idx,
        stride=args.stride,
        desired_height=args.desired_height,
        desired_width=args.desired_width,
    )
    if args.mode == "incremental":
        run_incremental_slam(dataset, device=args.device)
    elif args.mode == "batch":
        run_batch_slam(dataset, device=args.device)
    else:
        raise ValueError(f"Invalide `mode` argument: {args.mode}")


if __name__ == "__main__":
    main()
