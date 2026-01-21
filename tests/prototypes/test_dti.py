# Depreciated
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import use as mpl_backend
from paths import data_path

from pyeyes.prototypes.mpl.dti import plot_dti

mpl_backend("WebAgg")

# Options.
dti_type = "festive"  # 'skope', 'uncorr', or 'festive'

dti_root = data_path / "dti" / "dtifit"
dti_folder = dti_root / dti_type / f"{dti_type}_b1000_DTI"

Viewer = plot_dti(
    dti_imgs_or_path=dti_folder,
    suptitle=f"DTI: {dti_type.capitalize()}",
)

Viewer.launch()
