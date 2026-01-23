"""
Test case:
- user supplies two complex images (spin echoes) of the same size
"""

import numpy as np
from paths import cfg_path, data_path

from pyeyes.viewers import ComparativeViewer

small_scale = True

img_dict = {
    "4avg": np.load(data_path / "se" / "se_4avg.npy"),
    "1avg": np.load(data_path / "se" / "se_1avg.npy"),
}

if small_scale:
    for k in img_dict:
        img_dict[k] = img_dict[k] * 1e-10

# Parameters
named_dims = ["x", "y", "z"]
vdims = ["x", "y"]

Viewer = ComparativeViewer(
    data=img_dict,
    named_dims=named_dims,
    view_dims=vdims,
    config_path=cfg_path / "cfg_cplx.yaml",
)
Viewer._autoscale_clim(None)
Viewer.launch()
