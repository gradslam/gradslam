"""
Run GradSLAM PointFusion on various datasets.

AzureKinect, ICL, Replica, Scannet, AI2Thor, Record3D, Realsense
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from gradslam.datasets import datautils
from gradslam.geometry.geometryutils import relative_transformation
from natsort import natsorted

from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.pointclouds import Pointclouds
from gradslam.structures.rgbdimages import RGBDImages
from tqdm import trange

from gradslam.datasets import (
    AzureKinectDataset,
    ICLDataset,
    ReplicaDataset,
    ScannetDataset,
    Ai2thorDataset,
    Record3DDataset,
    RealsenseDataset,
)
from gradslam.datasets import load_dataset_config


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run PointFusion on a dataset")
    parser.add_argument('-s', "--sequence_path", type=str, help="Path to Scan Sequence")
    parser.add_argument('-g', "--gradslam_dataconfig", type=str, help="Path to Data Config File")
    args = parser.parse_args()
    
    # Load Dataset & Sequence Information
    sequence_path = args.sequence_path
    basedir = os.path.dirname(sequence_path)
    dataconfig_path = args.gradslam_dataconfig
    cfg = load_dataset_config(dataconfig_path)
    desired_image_height = cfg["desired_image_height"]
    desired_image_width = cfg["desired_image_width"]
    dataset = get_dataset(
        config_dict=cfg,
        basedir=basedir,
        sequence=os.path.basename(sequence_path),
        start=0,
        end=-1,
        stride=cfg["stride"],
        desired_height=desired_image_height,
        desired_width=desired_image_width,
    )
    device = torch.device("cuda:0")

    # Load RGB, Depth & Pose Data
    colors, depths, poses = [], [], []
    intrinsics = None
    for idx in range(len(dataset)):
        _color, _depth, intrinsics, _pose = dataset[idx]
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

    # Create rgbdimages object
    rgbdimages = RGBDImages(
        colors,
        depths,
        intrinsics,
        poses,
        channels_first=False,
        has_embeddings=False,
    )

    # Define PointFusion SLAM Object
    slam = PointFusion(odom="gt", dsratio=1, device=device, use_embeddings=False)

    # Initialize Variables for Incremental SLAM
    frame_cur, frame_prev = None, None
    pointclouds = Pointclouds(
        device=device,
    )
    colors, depths, poses = [], [], []
    intrinsics = None
    frame_cur, frame_prev = None, None

    print("Running PointFusion (incremental mode)...")
    # Incremental PointFusion SLAM
    for idx in trange(len(dataset)):
        _color, _depth, intrinsics, _pose, *_ = dataset[idx]

        frame_cur = RGBDImages(
            _color.unsqueeze(0).unsqueeze(0),
            _depth.unsqueeze(0).unsqueeze(0),
            intrinsics.unsqueeze(0).unsqueeze(0).float(),
            _pose.unsqueeze(0).unsqueeze(0).float(),
        )
        pointclouds, _ = slam.step(pointclouds, frame_cur, frame_prev)

    ## Visualize Pointcloud
    # pcd = pointclouds.open3d(0)
    # o3d.visualization.draw_geometries([pcd])

    # Save Pointcloud
    pointclouds.save_to_h5(sequence_path, include_embeddings=False)