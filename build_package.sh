#!/bin/bash -ex
.dev_env/bin/conda init /bin/bash
source ~/.bashrc
.dev_env/bin/conda activate
conda install -y pytorch torchvision -c conda
python setup.py build develop
cd docs
sphinx-build . _build
cd ..

