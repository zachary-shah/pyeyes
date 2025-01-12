import numpy as np

from pyeyes.viewers import ComparativeViewer

# Data
festive_pth = (
    "/local_mount/space/mayday/data/users/zachs/zachplotlib/data/dwi/recon_festive.npy"
)
skope_pth = (
    "/local_mount/space/mayday/data/users/zachs/zachplotlib/data/dwi/recon_skope.npy"
)
uncorr_pth = (
    "/local_mount/space/mayday/data/users/zachs/zachplotlib/data/dwi/recon_uncorr.npy"
)

festive = np.load(festive_pth)  # Bdir x X x Y x Z
skope = np.load(skope_pth)
uncorr = np.load(uncorr_pth)

img_dict = {"skope": skope, "festive": festive, "uncorr": uncorr}

# Parameters
named_dims = ["Bdir", "x", "y", "z"]
vdims = ["x", "y"]

Viewer = ComparativeViewer(data=img_dict, named_dims=named_dims, view_dims=vdims)

Viewer.launch()
