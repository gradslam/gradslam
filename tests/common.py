"""
Common utils for testing.
"""

import numpy as np
import os
import torch
from pathlib import Path

# TODO: Figure out why this isn't working with circleci
# DATA_DIR = Path(__file__).resolve().parent / "data" / "msrd_b2s3"
DATA_DIR = Path("tests/data/msrd_b2s3/")  # assumes tests are ran from package's base directory

def default_to_cpu_if_no_gpu(device):
    return (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device(device)
    )


def load_test_data(channels_first=False, batch_size=2):
    assert isinstance(batch_size, int) and batch_size < 3 and batch_size > 0
    data_dir_files = os.listdir(DATA_DIR)
    assert 'colors.npy' in data_dir_files, f"'colors.npy' not found (DATA_DIR='{DATA_DIR}')\n{data_dir_files}\n"
    colors = torch.from_numpy(
        np.load(DATA_DIR / "colors.npy")
    )[:batch_size]
    depths = torch.from_numpy(
        np.load(DATA_DIR / "depths.npy")
    )[:batch_size]
    intrinsics = torch.from_numpy(
        np.load(DATA_DIR / "intrinsics.npy")
    )[:batch_size]
    poses = torch.from_numpy(
        np.load(DATA_DIR / "poses.npy")
    )[:batch_size]
    if channels_first:
        colors = colors.permute(0, 1, 4, 2, 3).contiguous()
        depths = depths.permute(0, 1, 4, 2, 3).contiguous()
    return colors, depths, intrinsics, poses

if __name__ == "__main__":
    load_test_data()
