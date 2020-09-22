from argparse import ArgumentParser, RawTextHelpFormatter

import open3d as o3d
import torch
from torch.utils.data import DataLoader

from gradslam.datasets.scannet import Scannet
from gradslam.odometry.groundtruth import GroundTruthOdometryProvider
from gradslam.odometry.icp import ICPOdometryProvider
from gradslam.odometry.gradicp import GradICPOdometryProvider
from gradslam.slam.fusionutils import rgbdimages_to_pointclouds
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.rgbdimages import RGBDImages

parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument(
    "--scannet_path",
    type=str,
    required=True,
    help="Path to the directory containing extracted "
    "scans from Scannet dataset (e.g. should contain "
    "`scene0000_00/`, `scene0001_00/`, ...)",
)
parser.add_argument(
    "--scannet_meta_path",
    type=str,
    required=True,
    help="Path to the directory containing the meta data for Scannet sequences",
)
parser.add_argument(
    "--odometry",
    type=str,
    default="gradICP",
    choices=["GT", "ICP", "gradICP"],
    help="Odometry method to use. Supported options:\n"
    " GT = Ground Truth odometry\n"
    " ICP = Iterative Closest Point\n"
    " gradICP (*default) = Differentiable Iterative Closest Point",
)
args = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = Scannet(
        args.scannet_path,
        args.scannet_meta_path,
        ("scene0333_00", "scene0636_00",),
        start=0,
        end=4,
        height=240,
        width=320,
        channels_first=False,
    )
    loader = DataLoader(dataset=dataset, batch_size=2)
    colors, depths, intrinsics, poses, *_ = next(iter(loader))
    batch_size, seq_len = colors.shape[:2]

    if args.odometry == "GT":
        odom = GroundTruthOdometryProvider()
    elif args.odometry == "ICP":
        odom = ICPOdometryProvider(downsample_ratio=4)
        poses = torch.eye(4).float().view(1, 1, 4, 4).repeat(batch_size, seq_len, 1, 1)
    elif args.odometry == "gradICP":
        odom = GradICPOdometryProvider(downsample_ratio=4)
        poses = torch.eye(4).float().view(1, 1, 4, 4).repeat(batch_size, seq_len, 1, 1)

    slam = PointFusion(odom)
    rgbdimages = RGBDImages(colors, depths, intrinsics, poses, channels_first=False)
    pointclouds = rgbdimages_to_pointclouds(rgbdimages[:, 0], sigma=slam.sigma).to(
        device
    )
    prev_frame = rgbdimages[:, 0].to(device)
    for s in range(seq_len):
        live_frame = rgbdimages[:, s].to(device)
        pointclouds, new_poses = slam(pointclouds, live_frame, prev_frame)
        live_frame.poses[:, :1] = new_poses
        prev_frame = live_frame

    o3d.visualization.draw_geometries([pointclouds.o3d(0)])
    o3d.visualization.draw_geometries([pointclouds.o3d(1)])
