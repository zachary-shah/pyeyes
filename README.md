# pyeyes

Pyeyes is a n-dimensional data visualization tool for comparing images. Especially designed as an MRI visualization tool, inspired by FSLEyes.

## Installation
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

## Example scripts

Under `/tests`, run `compare_mrf.py` to see an example with quantitative data. Run `compare_cplx.py` to see examples with complex-valued data.

# Contributing

Before contributing, run
```bash
pre-commit install
```

# To-Do List

### Features
- [x] select value dimensions (default x, y)
- [x] slicer dimensions
- [x] allow for categorical dimensions
- [x] toggle real, im, mag, phase
- [x] control contrast (vmin/vmax)
- [x] select colormap
- [x] Add MRF color maps
- [x] select width, height to crop image to
- [x] toggle select which images to display
- [x] toggle ROI selection
    - [x] select ROI center
    - [x] select ROI height, width
    - [x] allow ROI to be displayed out of figure pane instead
- [x] toggle difference maps
    - [x] select reference for difference map
    - [x] select difference maps
- [x] Export figure (using bokeh built-in)
- [x] Export figure-generating config
- [ ] Image Masking
- [ ] "Auto-crop" to crop out all white-space from image
- [ ] Mouse wheel scrolling through slices
- [ ] Ortho 3D Viewer
- [ ] Multiple sliceable views
- [ ] Click and popup point value

### Formatting
- [ ] Allow renaming of image labels
- [ ] Add option to add borders for gridspec layout

### Efficiency
- [ ] Replace DynamicMap with HoloMap, or integrate LRU cache for slice options
- [ ] Replace ROI 2-click selection with BoundsXY (first attempt at this failed... will revisit later)
