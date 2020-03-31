from argparse import ArgumentParser

import open3d as o3d
from torch.utils.data import DataLoader

from gradslam.datasets.scannet import Scannet
from gradslam.odometry.groundtruth import GroundTruthOdometryProvider
from gradslam.slam.pointfusion import PointFusion

parser = ArgumentParser()
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
args = parser.parse_args()

if __name__ == "__main__":
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

    odom = GroundTruthOdometryProvider()
    slam = PointFusion(odom)
    pointclouds = slam(colors, depths, intrinsics, poses)

    o3d.visualization.draw_geometries([pointclouds.o3d(0)])
    o3d.visualization.draw_geometries([pointclouds.o3d(1)])
