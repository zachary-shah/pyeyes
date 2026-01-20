import numpy as np

from pyeyes.viewers import ComparativeViewer as cv

# Flat data case, but load slice with all zeros by default.
data = np.zeros((100, 100, 3))
data[:, :, 0] = 1
data[:, :, 2] = 3
viewer = cv(data=data, named_dims=list("xyz"))
viewer.launch()
