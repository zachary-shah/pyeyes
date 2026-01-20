"""
Test case:
- user supplies single image and a config from a different dataset (different name)
"""

import numpy as np
from paths import cfg_path, data_path

from pyeyes import set_theme
from pyeyes.viewers import ComparativeViewer

set_theme("dark")

# Load Data
mrf_folder = data_path / "mrf"
llr_1min_pd = np.load(mrf_folder / "llr_1min_pd.npy")
llr_1min_t1 = np.load(mrf_folder / "llr_1min_t1.npy")
llr_1min_t2 = np.load(mrf_folder / "llr_1min_t2.npy")
mrf_1min = np.stack([llr_1min_pd, llr_1min_t1, llr_1min_t2], axis=0)

# Allow loading viewer from config
config_path = cfg_path / "cfg_mrf_single.yaml"

Viewer = ComparativeViewer(
    data={"1-Minute MRF": mrf_1min},
    named_dims=["Map Type", "x", "y", "z"],
    view_dims=["y", "z"],
    cat_dims={"Map Type": ["PD", "T1", "T2"]},
    config_path=config_path,
)

Viewer.launch()
