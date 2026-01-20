"""
Test case:
- user supplies two complex images (spin echoes) of the same size
"""

import numpy as np
from paths import cfg_path, data_path

from pyeyes.viewers import ComparativeViewer

img_dict = {
    "4avg": np.load(data_path / "se" / "avg_se.npy"),
    "1avg": np.load(data_path / "se" / "single_se.npy"),
}

# Parameters
named_dims = ["x", "y", "z"]
vdims = ["x", "y"]

Viewer = ComparativeViewer(
    data=img_dict,
    named_dims=named_dims,
    view_dims=vdims,
    config_path=cfg_path / "cplx_config.yaml",
)
Viewer.launch()
