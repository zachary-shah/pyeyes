from pathlib import Path

import numpy as np
from paths import cfg_path, data_path

from pyeyes.viewers import ComparativeViewer

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

# Allow categorial dimensions to be specified
cat_dims = {"Map Type": ["PD", "T1", "T2"]}

Viewer = ComparativeViewer(
    data=img_dict,
    named_dims=named_dims,
    view_dims=list("xy"),
    cat_dims=cat_dims,
)

export_path = Path("./exports/reload_test/")
Viewer.export_reloadable_pyeyes(
    path=export_path / "test_export_reloadable.py", num_slices_to_keep={"z": 20}
)
