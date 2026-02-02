"""
Test case:
- user supplies three images of the same size with large dataset
"""

import numpy as np
from paths import cfg_path, data_path

from pyeyes import set_theme
from pyeyes.viewers import ComparativeViewer

set_theme("dark")

# Data
festive_pth = data_path / "dwi" / "festive.npy"
skope_pth = data_path / "dwi" / "skope.npy"
uncorr_pth = data_path / "dwi" / "uncorr.npy"

festive = np.load(festive_pth)  # Bdir x X x Y x Z
skope = np.load(skope_pth)
uncorr = np.load(uncorr_pth)

img_dict = {"skope": skope, "festive": festive, "uncorr": uncorr}

# Parameters
named_dims = ["Bdir", "x", "y", "z"]
vdims = ["x", "y"]

config_path = cfg_path / "cfg_diff.yaml"

Viewer = ComparativeViewer(
    data=img_dict, named_dims=named_dims, view_dims=vdims, config_path=config_path
)

Viewer.launch()
