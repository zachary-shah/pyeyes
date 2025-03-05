"""
Test case:
- user supplies two complex images (spin echoes) of the same size
"""

import numpy as np
import torch

from pyeyes.viewers import ComparativeViewer

# Data
se_folder = "/local_mount/space/mayday/data/users/zachs/pyeyes/data/autofocus"

img_dict = {
    "af": np.load(f"{se_folder}/x_autofocus.npy"),
    "no_comp": np.load(f"{se_folder}/x_db0_no_comp.npy"),
    "smooth": np.load(f"{se_folder}/x_db0_smooth.npy"),
    "gt": np.load(f"{se_folder}/x_gt.npy"),
}

# Parameters
named_dims = ["x", "y", "z"]
vdims = ["x", "y"]

Viewer = ComparativeViewer(
    data=img_dict,
    named_dims=named_dims,
    view_dims=vdims,
    config_path="./cfgs/af_config.yaml",
)
Viewer.launch()
