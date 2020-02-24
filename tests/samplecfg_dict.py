import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import cfgnode
from cfgnode import CfgNode as CN


cfg = {"TRAIN": {"HYPERPARAM_1": 0.9}}
