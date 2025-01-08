# pyeyes

Pyeyes is a n-dimensional data visualization tool for comparing images. Especially designed as an MRI visualization tool, inspired by FSLEyes.

## Installation
You can create the relevant conda environment using mamba:
```
mamba env create -n pyeyes --file env.yml
```

Activate the installed environment:
```
mamba activate pyeyes
```

Then install the pyeyes library by moving to the mr_recon directory and running
```
pip install -e ./
```

## Feature List
- [ ] select value dimensions (default x, y)
- [x] slicer dimensions
- [ ] toggle real, im, mag, phase
- [x] control contrast (vmin/vmax)
- [x] select colormap
- [ ] select width, height to crop image to
- [ ] toggle select which images to display
- [ ] toggle difference maps
    - [ ] select reference for difference map
    - [ ] select difference maps
- [ ] toggle ROI selection
    - [ ] select ROI center
    - [ ] select ROI height, width