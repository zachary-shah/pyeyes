import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use as mpl_backend

from pyeyes.mpl.dti import plot_dti

mpl_backend("WebAgg")

# Options.
dti_type = "festive"  # 'skope', 'uncorr', or 'festive'

# TODO: allow multiple DTI to compare like with ComparativeViewer. This will need to handle color dims tho
dti_root = "/local_mount/space/mayday/data/users/zachs/pyeyes/data/dti/dtifit"
dti_folder = f"{dti_root}/{dti_type}/{dti_type}_b1000_DTI"

Viewer = plot_dti(
    dti_imgs_or_path=dti_folder,
    suptitle=f"DTI: {dti_type.capitalize()}",
)

Viewer.launch()
