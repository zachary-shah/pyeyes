import numpy as np

from pyeyes.viewer2 import ComparativeViewer2

# Data
mrf_folder = "/local_mount/space/mayday/data/users/zachs/zachplotlib/data/mrf/new"

llr_1min_pd = np.load(f"{mrf_folder}/masked_pd_filtered.npy")
llr_1min_t1 = np.load(f"{mrf_folder}/masked_t1_filtered.npy")
llr_1min_t2 = np.load(f"{mrf_folder}/masked_t2_filtered.npy")
llr_2min_pd = np.load(f"{mrf_folder}/masked_pd_filtered.npy")
llr_2min_t1 = np.load(f"{mrf_folder}/masked_t1_filtered.npy")
llr_2min_t2 = np.load(f"{mrf_folder}/masked_t2_filtered.npy")

mrf_1min = np.stack([llr_1min_pd, llr_1min_t1, llr_1min_t2], axis=0)
mrf_2min = np.stack([llr_2min_pd, llr_2min_t1, llr_2min_t2], axis=0)

img_dict = {"6 min": mrf_1min, "2min": mrf_2min}

# Parameters
named_dims = ["Map Type", "x", "y", "z"]
vdims = ["y", "z"]

# Allow categorial dimensions to be specified
cat_dims = {"Map Type": ["PD", "T1", "T2"]}

Viewer = ComparativeViewer2(
    data=img_dict,
    named_dims=named_dims,
    view_dims=vdims,
    cat_dims=cat_dims,
)

Viewer.launch()
