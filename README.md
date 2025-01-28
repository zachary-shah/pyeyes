# pyeyes

Pyeyes is a n-dimensional data visualization tool for comparing images. Especially designed as an MRI visualization tool, inspired by FSLEyes. Built on top of [Holoviews](https://holoviews.org/) and [Bokeh](https://bokeh.org/) for interative plotting.

<video width="600" controls>
  <source src="./doc/demo.mov" type="video/mp4">
</video>

## Features

- **Interactive Slicing:** Seemlessly navigate through MRI volumes of arbitrary dimensionality.
- **Dynamic Contrast Adjustment:** Toggle through different color maps, color limits, and more on the fly.
- **Quantitative Imaging Views:** Support for the standard quantitative MRI colormaps.
- **Comparative Metrics:** Get quick looks at standard image-processing metrics against your gold-standard datasets.
- **Export and Repeatability:** Save viewer configurations you like and export static figures with ease.

## Installation

### Using PyPI
Install the package and all dependences from pip manager:
```
pip install pyeyes
```

### Development
Alternatively, for contributing, ou can create the relevant conda environment using mamba:
```
mamba env create -n pyeyes --file env.yml
```

Activate the installed environment:
```
mamba activate pyeyes
```

### Example Usage

```python
import numpy as np
from pyeyes import set_theme, ComparativeViewer

# Choose from 'dark', 'light', and 'soft_dark'
set_theme('dark')

# Form Dictionary of Datasets to view
img_dict = {
    "Dataset 1": np.random.randn((3, 200, 200, 200, 30)),
    "Dataset 2": np.random.randn((3, 200, 200, 200, 30)),
}

# Describe the dimensionality of the data
named_dims = ["Map Type", "x", "y", "z", "Vol"]

# Decide which dimensions to view
vdims = ["y", "z"]

# Allow categorial dimensions to be specified
cat_dims = {"Map Type": ["PD", "T1", "T2"]}

# Once launched, viewer config can be saved to config
# path for repeating view with same or different data
config_path = "./cfgs/cfg_mrf_1min_vs_2min.yaml"

# Initialize
Viewer = ComparativeViewer(
    data=img_dict,
    named_dims=named_dims,
    view_dims=vdims,
    cat_dims=cat_dims,
    config_path=config_path,
)

# Launch viewer in web browser!
Viewer.launch()
```

# Contributing

Before contributing, run
```bash
pre-commit install
```
