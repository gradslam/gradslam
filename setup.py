import runpy

import logging
from setuptools import setup, find_packages

# Retrieve __version__ from the package.

PACKAGE_NAME = "gradslam"
VERSION = runpy.run_path("gradslam/version.py")["__version__"]
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


logger = logging.getLogger()
logging.basicConfig(format="%(levelname)s - %(message)s")


def get_requirements():
    packages = None
    with open("requirements.txt") as f:
        packages = f.read().splitlines()
    return packages


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
                "nbsphinx",
                "pytest>=4.6",
                "pytest-cov>=2.7",
                "sphinx==2.2.0",  # pinned to resolve issue with docutils 0.16b0.dev
            ],
        },
        zip_safe=True,
        include_dirs=[],
        classifiers=CLASSIFIERS,
    )
