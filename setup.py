import os
import logging
from setuptools import setup, find_packages

import numpy as np
import torch

CUDA_AVAILABLE = False
if torch.cuda.is_available():
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    CUDA_AVAILABLE = True


if not torch.cuda.is_available():
    # From: https://github.com/NVIDIA/apex/blob/b66ffc1d952d0b20d6706ada783ae5b23e4ee734/setup.py
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt
    # to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during
    # an nvidia-docker build command).
    print(
        "\nWarning: Torch did not find available GPUs on this system.\n",
        "If your intention is to cross-compile, this is not an error.\n"
        "By default, gradslam will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
        "Volta (compute capability 7.0), and Turing (compute capability 7.5).\n"
        "If you wish to cross-compile for a single specific architecture,\n"
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n',
    )
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"

PACKAGE_NAME = "gradslam"
VERSION = "0.0.1"
DESCRIPTION = "gradSLAM: Dense SLAM meets Automatic Differentiation"
URL = "<url.to.go.in.here>"
AUTHOR = "MontrealRobotics"
LICENSE = "MIT (TBD)"
DOWNLOAD_URL = ""
LONG_DESCRIPTION = """
A longer description of what the library does. This will appear on pypi, and also influence
search keywords etc. (Maybe 1-2 paragraphs long)
"""
CLASSIFIERS = [
    "Intended Audience :: Developers",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Operating System :: POSIX :: Linux",
    # TODO: Add Windows OS
    "Programming Language :: Python :: 3 :: Only",
    "License :: OSI Approved :: MIT",
    "Topic :: Software Development :: Libraries",
]

# Minimum required pytorch version (TODO: check if versions below this work).
TORCH_MINIMUM_VERSION = "1.3.0"

logger = logging.getLogger()
logging.basicConfig(format="%(levelname)s - %(message)s")

# Check that PyTorch version installed meets minimum requirements.
if torch.__version__ < TORCH_MINIMUM_VERSION:
    logger.warning(
        f"gradslam has beent tested with PyTorch >= {0}. Found version {1} instead.".format(
            TORCH_MINIMUM_VERSION, torch.__version__
        )
    )


def get_requirements():
    return [
        "colorama",
        "flake8",
        "black",
        "imageio",
        "kornia",
        "matplot",
        "natsort",
        "numpy",
        "open3d==0.10.0.0",    # pinned to resolve https://github.com/intel-isl/Open3D/issues/2508
        "opencv-python-headless",
        "plotly",
        "pytest>=4.6",
        "pytest-cov>=2.7",
        "pyyaml",
        "scikit-image",
        "sphinx==2.2.0",  # pinned to resolve issue with docutils 0.16b0.dev
        "tqdm",
    ]


def get_extensions():
    if CUDA_AVAILABLE:
        return [
            CUDAExtension(
                "gradslam.chamferdistcuda",
                [
                    "cuda/chamfer_cuda.cpp",
                    "cuda/chamfer.cu",
                ],
            ),
        ]
    else:
        return []


if __name__ == "__main__":

    setup(
        # Metadata
        name=PACKAGE_NAME,
        version=VERSION,
        author=AUTHOR,
        description=DESCRIPTION,
        url=URL,
        long_description=LONG_DESCRIPTION,
        licence=LICENSE,
        python_requires=">3.6",
        # Package info
        packages=find_packages(exclude=("docs", "test", "examples")),
        install_requires=get_requirements(),
        zip_safe=True,
        include_dirs=[np.get_include()],
        classifiers=CLASSIFIERS,
        # CUDAExtensions
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension} if CUDA_AVAILABLE else {},
    )
