"""
Test case:
- user supplies single image and a config created from the same dataset (same name)
"""

import numpy as np
from paths import cfg_path, data_path

from pyeyes import set_theme
from pyeyes.viewers import ComparativeViewer

set_theme("dark")

# Load Data
mrf_folder = data_path / "mrf"
llr_2min_pd = np.load(mrf_folder / "llr_2min_pd.npy")
llr_2min_t1 = np.load(mrf_folder / "llr_2min_t1.npy")
llr_2min_t2 = np.load(mrf_folder / "llr_2min_t2.npy")
mrf_2min = np.stack([llr_2min_pd, llr_2min_t1, llr_2min_t2], axis=0)

# Allow loading viewer from config
config_path = cfg_path / "cfg_mrf_single.yaml"

Viewer = ComparativeViewer(
    data={"2-Minute MRF": mrf_2min},
    named_dims=["Map Type", "x", "y", "z"],
    view_dims=["y", "z"],
    cat_dims={"Map Type": ["PD", "T1", "T2"]},
    config_path=config_path,
)

Viewer.launch()
