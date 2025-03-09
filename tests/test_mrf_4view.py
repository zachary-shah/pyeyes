"""
Test case:
- user supplies four images of the same size in the most basic way, with a categorical dimension
"""

import numpy as np

from pyeyes import set_theme
from pyeyes.viewers import ComparativeViewer

set_theme("dark")

# Data
mrf_folder = "/local_mount/space/mayday/data/users/zachs/pyeyes/data/mrf"

llr_1min_pd = np.load(f"{mrf_folder}/llr_1min_pd.npy")
llr_1min_t1 = np.load(f"{mrf_folder}/llr_1min_t1.npy")
llr_1min_t2 = np.load(f"{mrf_folder}/llr_1min_t2.npy")
llr_2min_pd = np.load(f"{mrf_folder}/llr_2min_pd.npy")
llr_2min_t1 = np.load(f"{mrf_folder}/llr_2min_t1.npy")
llr_2min_t2 = np.load(f"{mrf_folder}/llr_2min_t2.npy")

mrf_1min = np.stack([llr_1min_pd, llr_1min_t1, llr_1min_t2], axis=0)
mrf_2min = np.stack([llr_2min_pd, llr_2min_t1, llr_2min_t2], axis=0)

# simulate 4min and 8min with shot noise
llr_45sec_pd = llr_1min_pd + np.random.normal(
    0, np.std(llr_1min_pd) * 0.1, llr_1min_pd.shape
)
llr_45sec_t1 = llr_1min_t1 + np.random.normal(
    0, np.std(llr_1min_t1) * 0.1, llr_1min_t1.shape
)
llr_45sec_t2 = llr_1min_t2 + np.random.normal(
    0, np.std(llr_1min_t2) * 0.1, llr_1min_t2.shape
)

llr_30sec_pd = llr_1min_pd + np.random.normal(
    0, np.std(llr_1min_pd) * 0.25, llr_1min_pd.shape
)
llr_30sec_t1 = llr_1min_t1 + np.random.normal(
    0, np.std(llr_1min_t1) * 0.25, llr_1min_t1.shape
)
llr_30sec_t2 = llr_1min_t2 + np.random.normal(
    0, np.std(llr_1min_t2) * 0.25, llr_1min_t2.shape
)

mrf_30sec = np.stack([llr_30sec_pd, llr_30sec_t1, llr_30sec_t2], axis=0)
mrf_45sec = np.stack([llr_45sec_pd, llr_45sec_t1, llr_45sec_t2], axis=0)

# mask
mrf_30sec = mrf_30sec * (np.abs(mrf_2min) > 0)
mrf_45sec = mrf_45sec * (np.abs(mrf_2min) > 0)

img_dict = {
    "30sec": mrf_30sec,
    "45sec": mrf_45sec,
    "1min": mrf_1min,
    "2min": mrf_2min,
}

# Parameters
named_dims = ["Map Type", "x", "y", "z"]
vdims = ["y", "z"]

# Allow categorial dimensions to be specified
cat_dims = {"Map Type": ["PD", "T1", "T2"]}

Viewer = ComparativeViewer(
    data=img_dict,
    named_dims=named_dims,
    view_dims=vdims,
    cat_dims=cat_dims,
)

Viewer.launch()
