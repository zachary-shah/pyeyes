"""
Test case:
- user supplies two complex images (spin echoes) of the same size
"""

import numpy as np
import torch

from pyeyes.viewers import ComparativeViewer

# Data
se_folder = "/local_mount/space/mayday/data/users/zachs/pyeyes/data/se"

img_dict = {
    "4avg": np.load(f"{se_folder}/avg_se.npy"),
    "1avg": np.load(f"{se_folder}/single_se.npy"),
}

# Parameters
named_dims = ["x", "y", "z"]
vdims = ["x", "y"]

Viewer = ComparativeViewer(
    data=img_dict,
    named_dims=named_dims,
    view_dims=vdims,
    config_path="./cfgs/cplx_config.yaml",
)
Viewer.launch()
