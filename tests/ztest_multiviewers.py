"""
Test case:
- two different types of dataset are launched simultaneously
"""

import numpy as np
from paths import cfg_path, data_path

from pyeyes import ComparativeViewer, launch_viewers

# Data
af_folder = data_path / "af"
se_folder = data_path / "se"

img_dict_af = {
    "af": np.load(af_folder / "x_af.npy"),
    "no_comp": np.load(af_folder / "x_no_comp.npy"),
    "smooth": np.load(af_folder / "x_smooth.npy"),
    "gt": np.load(af_folder / "x_gt.npy"),
}

img_dict_se = {
    "4avg": np.load(se_folder / "se_4avg.npy"),
    "1avg": np.load(se_folder / "se_1avg.npy"),
}

Viewer_SE = ComparativeViewer(
    data=img_dict_se,
    named_dims=["x", "y", "z"],
    view_dims=["x", "y"],
    config_path=cfg_path / "cfg_cplx.yaml",
)

Viewer_AF = ComparativeViewer(
    data=img_dict_af,
    named_dims=["x", "y", "z"],
    view_dims=["x", "y"],
    config_path=cfg_path / "cfg_af.yaml",
)

launch_viewers(
    {"SpinEcho": Viewer_SE, "Autofocus": Viewer_AF},
    port=9999,
    title="Spin Echo vs Autofocus",
)
