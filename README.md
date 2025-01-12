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

## Example scripts

Under `/tests`, run `compare_diffusion.py` to see diffusion weighted image example. Run `compare_mrf.py` to see MRF example (Note: offical MRF colormap implementation still a TODO.)

# Contributing

Before contributing, run
```bash
pre-commit install
```

## Feature List
- [x] select value dimensions (default x, y)
- [x] slicer dimensions
- [ ] allow for categorical dimensions
- [x] toggle real, im, mag, phase
- [x] control contrast (vmin/vmax)
- [x] select colormap
- [x] select width, height to crop image to
- [x] toggle select which images to display
- [ ] toggle difference maps
    - [ ] select reference for difference map
    - [ ] select difference maps
- [ ] toggle ROI selection
    - [ ] select ROI center
    - [ ] select ROI height, width
- [ ] Add MRF color maps
- [ ] Export figure
- [ ] Export figure-generating config
