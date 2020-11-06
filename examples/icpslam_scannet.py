from argparse import ArgumentParser, RawTextHelpFormatter

import open3d as o3d
import torch
from torch.utils.data import DataLoader

from gradslam.datasets.scannet import Scannet
from gradslam.slam.icpslam import ICPSLAM
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
    default="gradicp",
    choices=["gt", "icp", "gradicp"],
    help="Odometry method to use. Supported options:\n"
    " gt = Ground Truth odometry\n"
    " icp = Iterative Closest Point\n"
    " gradicp (*default) = Differentiable Iterative Closest Point\n",
)
args = parser.parse_args()


if __name__ == "__main__":
    # select device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load dataset
    # Scannet needs to have been extracted in our format
    dataset = Scannet(
        args.scannet_path,
        args.scannet_meta_path,
        (
            "scene0333_00",
            "scene0636_00",
        ),
        start=0,
        end=4,
        height=240,
        width=320,
    )
    loader = DataLoader(dataset=dataset, batch_size=2)
    colors, depths, intrinsics, poses, *_ = next(iter(loader))

    # create rgbdimages object
    rgbdimages = RGBDImages(colors, depths, intrinsics, poses, channels_first=False)

    # SLAM
    slam = ICPSLAM(odom=args.odometry, dsratio=4, device=device)
    pointclouds, recovered_poses = slam(rgbdimages)

    # visualization
    o3d.visualization.draw_geometries([pointclouds.open3d(0)])
    o3d.visualization.draw_geometries([pointclouds.open3d(1)])
