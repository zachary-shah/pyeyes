# Pyeyes

## Overview of Repo

`pyeyes` is a Python repository for viewing arbitrarily high-dimensional data with ease, speficially designed with MRI data in mind, but is useful for viewing any sort of high-dimensional (especially complex) numpy data.

In `pyeyes`, a dataset is designated as a array-like input with `N` input dimensions. The dataset can be described by `N` `named_dims`, of which a subset of these designated `view_dims` (or `vdims`) can be viewed fully at a time. All other dimensions, designated `slice_dims` (or `sdims` for short), can then be sliced through using widgets like sliders for display of other 2D slices along these views in real time.

The goal of `pyeyes` as a repository is to support multiple modes of GUI-based interactive data `Viewers`, which are classes for creating various views on a given dataset. Currently, only one viewer, `ComparativeViewer`, is supported, which is meant for comparing multiple image datasets of the same dimensionality and shape.

## ComparativeViewer
The `ComparativeViewer` class allows the user to easily parse and view equally dimensioned datasets (e.g. three 4-D spatio-temporal MRI datasets) which may be complex valued, described by a set of named dimensions. The user can select a single 2D view of the datasets to view simultaneously and update in real-time. This is facilitated by a GUI built primarily with `panel` and `HoloViews` to allow interaction with the datasets via `panel` widgets. Data can be easily navigated in arbitrary dimensions, compared against a designated reference, viewed with focus on specific sub-regions of interest, and exported / configured for easy re-examination on similar tasks.

The viewer was designed specifically for MRI researchers with a focus on comparative analysis between different acquisition and reconstruction techniques for a given dataset.

The GUI currently contains 6 Tabs: "View", "Contrast", "ROI", "Analysis", "Misc", and "Export", each with interactive widgets of different functionality based on their names.

### ComparativeViewer Functionality in `pyeyes v0.3.0` and before:

#### Main "View" Functionalities
- [x] Given a dataset, select view dimensions (default x, y). This is specified as a L/R and U/D viewing dimension
- [x] Allow user to change slice along all other dimensions in GUI
- [x] Allow user to change which view dimensions are active, replacing sliced dimensions as needed
- [x] Allow for input dimensions to be categorical, in which case slicing does not use a slider but instead a drop-down selector
- [x] Allow flips along L/R and U/D through buttons
- [x] Modify size of base images through slider "Size Scale"
- [x] Modify title font size
- [x] Modify L/R and U/D display ranges
- [x] In the default mode, click names of "Displayed Images" to enable which ones are viewed
- [x] Allow "Single View" mode, which instead of showing a grid of all images, will just show one at a time, which can be toggled by clicking different image names. This is enabled by a "Single View" button.

#### "Contrast" Functionalities
- [x] For complex-valued inputs, allow toggle between viewing real, im, mag, and phase of data. For magnitude, just toggle mag / real.
- [x] Allow user to control contrast with a 'clim' slider (sets vmin/vmax of images)
- [x] Select colormap for displayed images.
    - [x] Also supports "Quantitative" color maps for MRF, if input is supplied with a categorical dimension that is deemed MRF-like
- [x] Toggle inclusion/exclusion of colorbar
- [x] Add label to colorbar
- [x] Add "Auto Scale" button, which sets the clim and color map based on heuristics of percentile and view type

#### "ROI" Functionalities
- [x] Add functionality to draw a region of interest focus. This is initialized when the user clicks a "Draw ROI" button. They will then be prompted to select two corners which define a bounding box for the ROI, and then a menu with ROI options will appear. Customization includes:
    - [x] Color map for ROI (by default, is the 'same' as the image color map)
    - [x] Scale of zoom (factor by which it is enlarged relative to base image)
    - [x] Location (corner selection like 'top right', 'bottom left', etc)
    - [x] Crops (slider to control L/R crop in terms of pixel coordinates, and similar control for U/D)
    - [x] Line color selector for bounding box of ROI
    - [x] Line width for bounding box of ROI
    - [x] Zoom order (interpolation, zero, first, cubic, etc)
    - [x] Also an option to disable the ROI overlay, in which case it is displayed below the display image rather than over it. In this case, control for 'Zoom Scale' and 'ROI Location' are disabled, as these will be automatically parameterized.

#### "Analysis" functionalities
- [x] Compute metrics relative to one dataset, specified as the "reference". This dataset is automatically displayed on the left-hand side
- [x] First, allow selection of a 'difference metric map'. This includes L1 difference between 2 images, NRMSE, SSIM maps, etc. Modifications include
    - [x] Error Map Type
    - [x] Error scale
    - [x] Error color map
    - [x] An "autoformat" button which automatically sets the scale and color depending on the type
- [x] Clicking the "Error Map" button will enable this display below all Display Images except the reference
- [x] Also "Text" metrics can be displayed depending on which ones are selected in the GUI. These by default display on the Error Map plot, unless not shown, in which case they are instead placed over the main image. Modifications for these include:
    - [x] text metrics font size
    - [x] text metrics location

#### "Export" functionalities
- [x] User can export the config used to generate the viewer to a path, which can then be loaded in as a paramter to the viewer as `config_path="/path/to/cfg/`.
- [x] User can export a static HTML of the viewer in the GUI, specifiing a reduced slice along each sliceable dimension to save for efficiency. Only saves along the current viewing dimension, as the viewer is static
- [x] Not in GUI, but viewer can also export the viewer as runnable .py file for headless scenarios using `viewer.export_reloadable_pyeyes()`

#### Other functionalities
In the backend, user choices on contrast, slicing, and cropping are cached when appropriate views are changed. For example, if the user changes between magnitude and phase views, a cache of the clim and cmap are made so that returning to the other view restores the contrast settings at the time of the view switch.

## Repo Structrue
TODO: describe gui elements in src/gui (generalized Widget interface)

TODO: explain unsupported prototyeps. there is a prototype of a 1d viewer (created with `launch_1d_viewer` in `src/pyeyes/prototypes/line.py`), as well as a matplotlib-based viewer of a specific type of MRI data (diffusion data) in `src/pyeyes/prototypes/mpl/`, but these are not to be edited, as they are currently prototypes with no declared support.

## Test Infrastructure
TODO: describe pytest, tests/ztest_* being runnable demos but not part of auto-test suite, GUI test built with playwright


## New features as of v0.4.0

### Brief overview
`pyeyes` v0.4.0 includes many backend speed improvements to `ComparativeViewer`, improved plot interactivity through mouse-wheel scrolling and pixel value inspection, and numerous new plot modification options exposed in the panel GUI.

### Exhaustive Feature Update Log

- [x] Dimension inputs made more flexible and intuitive to accept no inputs, string lists, einsum-like strings, or non-delimited character strings of length equal to the number of dimensions.
- [x] `mouse_scroll`: This feature should allow the user to scroll through a slice of images when moving the mouse scroll wheel when the mouse is over the display image.
    - this automatically scrolls along the last modified sliceable dimension (including categorial dimensions), or by default the first dimension in the list of sliceable dims.
    - Every time the view_dims is updated, this scrollable dimension updates to the first dimension in sliceable dims
    - this required disabling bokeh-default scroll-based zoom
- [x] Image normalization options now exposed on interface in `Analysis` tab.
    - [x] `Normalize Images` option: Will normalize all images to selected reference dataset, as well as display error maps after image normalization
    - [x] `Normalize for Error Metrics Only`: (previously `Normalize Error Maps`): will not normalize display images, but does display error maps computed after image normalization
- [x] Exposed more display and interaction features in `Misc` tab
    - [x] User-defined font selection
    - [x] Optional toggle of displaying image titles
    - [x] Error map titles now optional to display
    - [x] Option to enable grid outline to deliniate images
    - [x] `popup-pixel`: If the user clicks their mouse anywhere on a display image, a popup then displays the value of the pixel where they clicked their mouse, for all display images at that pixel location. This is disabled by default as it increases scroll latency. Enabling will allow highlighted pixel to be tracked as user scrolls through slices. Features for popup-pixel:
        - [x] Optional display of pixel coordinate in true array frame
        - [x] Optional toggle of dispaly on error maps
        - [x] Modification of popup location with respect to pixel
        - [x] Button to clear popup
    - [x] Editing of each image name as it is displayed on figure

### Bug Fixes
- [x] Issue with overlay behaviour related to flipping or cropping image dimensions resolved.
- [x] Issue with in-figure overlays having inconsistent locations for different image flips
- [x] Ensured widgets can't be incremented out of bounds (like incrementing sdims below 0 or above max)
- [x] Fix bug with some values of error map in MRF being NaN and nrmse being too high
- [x] Fix bug with multiple calls of figure update for certain widget events
- [x] Cbar margin overflow adjusted
- [x] Number of digits in displayed numerics made more consistent with format integration from `bokeh.models.BasicTickFormatter` and sig-fig control.

## Future Feature Log

### ComparativeViewer
- [ ] Add toggle for "Single View" mode on display images to be short-cutted by keyboard inputs (possibly up/down arrows or left-right arrows?). If the user has checked `Single View` button, then clicking the up/down buttons on the keyboard will trigger moving along the list of possible displayed images.
- [ ] On the `Analysis` Tab, add a widget which specifies to what precision the metrics should be displayed. Also add a widget that allows the metrics to instead be displayed in scientific notation, which is just a toggle (always set to 3 sig figs for scientific notation)
- [ ] Support masking, where if some pixels in the input image are "nan", then display these as transparent.
- [ ] cache clim properties for quantitative maps / categorical input
- [ ] allow clicking on any display image for popup. currently the user must click on the left-most display image, as this image is the only source connected to the callback stream.
- [ ] add global watcher for scroller to disable scrolling on the web page while scrolling images
- [ ] Make display image names update synced across other GUI elements besides figure itself (Displayed Images button list, dataset reference drop-down menu, etc.)

### MultisliceViewer

A to-be-supported viewer. This will take a single dataset with the same inputs as the `ComparativeViewer`, except that the `data` includes only one numpy array (not multiple datasets). Then by default the viewer will show just one 2D image, which is sliceable as the ComparativeViewer normally is (change viewing dims, change slice dims, change colormap limits, etc). But then, the user will be able to press a "+" button to the right of this display, which will add a second un-linked view of the data, which can also be modified with the same controls in a separate panel. this can continue up to 4 different simultaneous views.

For this viewer, instead of having panes "View", "Contrast", "ROI", "Analysis", "Export", there will be 1 pane per view, with the "Export" tab always present. The addition of each new view will also add a new pane to control that specific data view.

Additionally, when adding a new view by pressing the "+" button, the user will have an option to request which type of view, of which we will start out with 2:
1) a 2D viewer (the view we have described with ComparativeViewer)
2) a 1D viewer (which will only have 1 viewing dim rather than 2).

Each type of viewer has controls unique to it.

The 2D viewer pane should have the following widgets:
- [ ] Selection of L/R and U/D viewing dimension
- [ ] EditableIntSliders to slice through all sdims
- [ ] Flip image U/D and L/R
- [ ] Control size scale and title font, title name
- [ ] Pick mag/phase/real/imag (cplx_type) for complex data, and mag/real for real data
- [ ] Clim controls
- [ ] Color map picker
- [ ] Add colorbar
- [ ] Colorbar label
- [ ] Auto-format button (autoscale in comarative viewer)

The 1D viewer has:
- [ ] Selection of X dimension and Y dimension. All other dimensions are sliced.
- [ ] Sliders for each sdim.
- [ ] Selection of number of lines to show for Y dimension, by control of start, stop, step along this dimension. By default, pick start/stop as 0/end, and step to provide maximally 50 lines displayed at once.
- [ ] Each line is labeled as "{Y} {index}", for index being the index along dimension named Y
- [ ] Allow toggle between linear / log scale for both x and y axes
- [ ] Allow control of opacity (alpha)
- [ ] Finally, the `MultisliceViewer` should be exportable by config in the same way the `ComparativeViewer` is. When the user is happy with their data views, they can provide a path and hit "export config" button, and a config will be saved, such that if they supply the path to this config with a data input of the same size and named dimensions, it will re-load the Viewer with the same state as when they saved the config. This may require a clever paramterization to allow easy load/save IO in a way that is flexible to support new features with backwards compatibility for partially described previous versions of config files.
