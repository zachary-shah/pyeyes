"""
Test case:
- user supplies two images of the same size with pre-defined config
"""

import numpy as np

from pyeyes.viewers import ComparativeViewer

# Load Data
mrf_folder = "/local_mount/space/mayday/data/users/zachs/pyeyes/data/mrf"
llr_1min_pd = np.load(f"{mrf_folder}/llr_1min_pd.npy")
llr_1min_t1 = np.load(f"{mrf_folder}/llr_1min_t1.npy")
llr_1min_t2 = np.load(f"{mrf_folder}/llr_1min_t2.npy")
llr_2min_pd = np.load(f"{mrf_folder}/llr_2min_pd.npy")
llr_2min_t1 = np.load(f"{mrf_folder}/llr_2min_t1.npy")
llr_2min_t2 = np.load(f"{mrf_folder}/llr_2min_t2.npy")

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
config_path = "./cfgs/cfg_mrf_1min_vs_2min.yaml"

Viewer = ComparativeViewer(
    data=img_dict,
    named_dims=named_dims,
    view_dims=vdims,
    cat_dims=cat_dims,
    config_path=config_path,
)

Viewer.launch()
