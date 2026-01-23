"""
Test case:
- user supplies two images of the same size with pre-defined config
"""

import numpy as np
from paths import cfg_path, data_path

from pyeyes.viewers import ComparativeViewer, spawn_comparative_viewer_detached

spawn_detached = False

# Load Data
mrf_folder = data_path / "mrf"
llr_1min_pd = np.load(mrf_folder / "llr_1min_pd.npy")
llr_1min_t1 = np.load(mrf_folder / "llr_1min_t1.npy")
llr_1min_t2 = np.load(mrf_folder / "llr_1min_t2.npy")
llr_2min_pd = np.load(mrf_folder / "llr_2min_pd.npy")
llr_2min_t1 = np.load(mrf_folder / "llr_2min_t1.npy")
llr_2min_t2 = np.load(mrf_folder / "llr_2min_t2.npy")

mrf_1min = np.stack([llr_1min_pd, llr_1min_t1, llr_1min_t2], axis=0)
mrf_2min = np.stack([llr_2min_pd, llr_2min_t1, llr_2min_t2], axis=0)

# Make a Dictionary of volumes to compare
img_dict = {"1 Minute MRF": mrf_1min, "2 Minute MRF": mrf_2min}

# Parameters
named_dims = ["Map Type", "x", "y", "z"]
vdims = ["y", "z"]

# Allow categorial dimensions to be specified
cat_dims = {"Map Type": ["PD", "T1", "T2"]}

# Allow loading viewer from config
config_path = cfg_path / "cfg_mrf.yaml"

if spawn_detached:
    spawn_comparative_viewer_detached(
        data=img_dict,
        named_dims=named_dims,
        view_dims=vdims,
        cat_dims=cat_dims,
        config_path=config_path,
    )
    print("Process spawned in detached mode.")
else:
    Viewer = ComparativeViewer(
        data=img_dict,
        named_dims=named_dims,
        view_dims=vdims,
        cat_dims=cat_dims,
        config_path=config_path,
    )
    print("Process launching in attached mode.")
    Viewer.launch(title="MRF Viewer")
