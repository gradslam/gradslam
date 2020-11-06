from argparse import ArgumentParser, RawTextHelpFormatter

import open3d as o3d
import torch
from torch.utils.data import DataLoader

from gradslam.datasets.icl import ICL
from gradslam.datasets.tum import TUM
from gradslam.slam.pointfusion import PointFusion
from gradslam.structures.rgbdimages import RGBDImages

parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    choices=["icl", "tum"],
    help="Dataset to use. Supported options:\n"
    " icl = Ground Truth odometry\n"
    " tum = Iterative Closest Point\n",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="Path to the dataset directory",
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
    if args.dataset == "icl":
        dataset = ICL(args.dataset_path, seqlen=10, height=120, width=160)
    elif args.dataset == "tum":
        dataset = TUM(args.dataset_path, seqlen=10, height=120, width=160)
    loader = DataLoader(dataset=dataset, batch_size=2)
    colors, depths, intrinsics, poses, *_ = next(iter(loader))

    # create rgbdimages object
    rgbdimages = RGBDImages(colors, depths, intrinsics, poses, channels_first=False)

    # SLAM
    slam = PointFusion(odom=args.odometry, dsratio=4, device=device)
    pointclouds, recovered_poses = slam(rgbdimages)

    # visualization
    o3d.visualization.draw_geometries([pointclouds.open3d(0)])
    o3d.visualization.draw_geometries([pointclouds.open3d(1)])
