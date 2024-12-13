import numpy as np

from pyeyes.gradio_version.roi_gradio import RoiPlot

img_path = "/local_mount/space/mayday/data/users/zachs/zachplotlib/data/phantom/np"

gt = np.load(f"{img_path}/gt.npy")
img_titles = ["recon_gridded_nominal", "recon_gridded_skope", "recon_nufft_nominal", "recon_nufft_skope"]
img_list = [np.load(f"{img_path}/{img}.npy") for img in img_titles]

RP = RoiPlot(gt, img_list, img_titles)
RP.launch()

print("Complete!")