import logging
from setuptools import setup, find_packages

import numpy as np
import torch


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
        "chamferdist==1.0.0",
        "imageio",
        "kornia",
        "natsort",
        "numpy",
        "plotly",
        "pyyaml",
        "open3d==0.10.0.0",  # pinned to resolve https://github.com/intel-isl/Open3D/issues/2508
        "opencv-python-headless",
    ]


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
        extras_require={
            "all": ["matplot", "tqdm"],
            "dev": [
                "black",
                "flake8",
                "pytest>=4.6",
                "pytest-cov>=2.7",
                "sphinx==2.2.0",  # pinned to resolve issue with docutils 0.16b0.dev
            ],
        },
        zip_safe=True,
        include_dirs=[np.get_include()],
        classifiers=CLASSIFIERS,
    )
