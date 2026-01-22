# Pyeyes

## Overview of Repo

`pyeyes` is a Python repository for viewing arbitrarily high-dimensional data with ease, speficially designed with MRI data in mind, but is useful for viewing any sort of high-dimensional (especially complex) numpy data.

In `pyeyes`, a dataset is designated as a array-like input with `N` input dimensions. The dataset can be described by `N` `named_dims`, of which a subset of these designated `view_dims` (or `vdims`) can be viewed fully at a time. All other dimensions, designated `slice_dims` (or `sdims` for short), can then be sliced through using widgets like sliders for display of other 2D slices along these views in real time.

The goal of `pyeyes` as a repository is to support multiple modes of `Viewers`, which are classes for creating various views on a given dataset. Currently, only one viewer, `ComparativeViewer`, is supported, which is meant for comparing multiple image datasets of the same dimensionality and shape. This is functional but a bit messy and still missing some desired features. There is also a prototype of a 1d viewer (created with `launch_1d_viewer` in `src/pyeyes/prototypes/line.py`), as well as a matplotlib-based viewer of a specific type of MRI data (diffusion data) in `src/pyeyes/prototypes/mpl/`, but these are not to be edited, as they are currently prototypes with no declared support.

## ComparativeViewer Functionalities
The `ComparativeViewer` class allows the user to easily parse and view equally dimensioned datasets (e.g. three 4-D spatio-temporal MRI datasets) which may be complex valued, described by a set of named dimensions. The user can select a single 2D view of the datasets to view simultaneously and update in real-time. This is facilitated by a GUI built primarily with `panel` and `HoloViews` to allow interaction with the datasets via `panel` widgets. Data can be easily navigated in arbitrary dimensions, compared against a designated reference, viewed with focus on specific sub-regions of interest, and exported / configured for easy re-examination on similar tasks.

The viewer was designed specifically for MRI researchers with a focus on comparative analysis between different acquisition and reconstruction techniques for a given dataset.

The GUI currently contains 5 Tabs: "View", "Contrast", "ROI", "Analysis", and "Export", each with interactive widgets of different functionality based on their names.

### Main "View" Functionalities
- [ ] Given a dataset, select view dimensions (default x, y). This is specified as a L/R and U/D viewing dimension
- [ ] Allow user to change slice along all other dimensions in GUI
- [ ] Allow user to change which view dimensions are active, replacing sliced dimensions as needed
- [ ] Allow for input dimensions to be categorical, in which case slicing does not use a slider but instead a drop-down selector
- [x] Allow flips along L/R and U/D through buttons
- [x] Modify size of base images through slider "Size Scale"
- [x] Modify title font size
- [ ] Modify L/R and U/D display ranges
- [ ] In the default mode, click names of "Displayed Images" to enable which ones are viewed
- [ ] Allow "Single View" mode, which instead of showing a grid of all images, will just show one at a time, which can be toggled by clicking different image names. This is enabled by a "Single View" button.

### "Contrast" Functionalities
- [ ] For complex-valued inputs, allow toggle between viewing real, im, mag, and phase of data. For magnitude, just toggle mag / real.
- [ ] Allow user to control contrast with a 'clim' slider (sets vmin/vmax of images)
- [ ] Select colormap for displayed images.
    - [ ] Also supports "Quantitative" color maps for MRF, if input is supplied with a categorical dimension that is deemed MRF-like
- [ ] Toggle inclusion/exclusion of colorbar
- [ ] Add label to colorbar
- [ ] Add "Auto Scale" button, which sets the clim and color map based on heuristics of percentile and view type

### "ROI" Functionalities
- [ ] Add functionality to draw a region of interest focus. This is initialized when the user clicks a "Draw ROI" button. They will then be prompted to select two corners which define a bounding box for the ROI, and then a menu with ROI options will appear. Customization includes:
    - [ ] Color map for ROI (by default, is the 'same' as the image color map)
    - [ ] Scale of zoom (factor by which it is enlarged relative to base image)
    - [ ] Location (corner selection like 'top right', 'bottom left', etc)
    - [ ] Crops (slider to control L/R crop in terms of pixel coordinates, and similar control for U/D)
    - [ ] Line color selector for bounding box of ROI
    - [ ] Line width for bounding box of ROI
    - [ ] Zoom order (interpolation, zero, first, cubic, etc)
    - [ ] Also an option to disable the ROI overlay, in which case it is displayed below the display image rather than over it. In this case, control for 'Zoom Scale' and 'ROI Location' are disabled, as these will be automatically parameterized.

### "Analysis" functionalities
- [ ] Compute metrics relative to one dataset, specified as the "reference". This dataset is automatically displayed on the left-hand side
- [ ] First, allow selection of a 'difference metric map'. This includes L1 difference between 2 images, NRMSE, SSIM maps, etc. Modifications include
    - [ ] Error Map Type
    - [ ] Error scale
    - [ ] Error color map
    - [ ] An "autoformat" button which automatically sets the scale and color depending on the type
- [ ] Clicking the "Error Map" button will enable this display below all Display Images except the reference
- [ ] Also "Text" metrics can be displayed depending on which ones are selected in the GUI. These by default display on the Error Map plot, unless not shown, in which case they are instead placed over the main image. Modifications for these include:
    - [ ] text metrics font size
    - [ ] text metrics location

### "Export" functionalities
- [ ] User can export the config used to generate the viewer to a path, which can then be loaded in as a paramter to the viewer as `config_path="/path/to/cfg/`.
- [ ] User can export a static HTML of the viewer in the GUI, specifiing a reduced slice along each sliceable dimension to save for efficiency. Only saves along the current viewing dimension, as the viewer is static
- [ ] Not in GUI, but viewer can also export the viewer as runnable .py file

### Other functionalities
In the backend, user choices on contrast, slicing, and cropping are cached when appropriate views are changed. For example, if the user changes between magnitude and phase views, a cache of the clim and cmap are made so that returning to the other view restores the contrast settings at the time of the view switch.

## Overview of Tasks
The next steps for this repo are binned into 4 major categories:
1) Create a `pytest` test suite (translating existing tests as well) which exhaust testing the features in the section `ComparativeViewer Functionalities`. This will be a bit challenging, given that most of the features require interaction from the GUI, but we may need to create a virtual way of scripting these interactions. Task requests described more in `ComparativeViewer PyTests Setup`.
2) Once all tests pass, we will re-factor the repository as described in the section `Refactor`. This is meant to create a bit more flexibility for integrating new features and building new Viewers which can use redundant code for similar features to those in `ComparativeViewer`.
3) Then, we will fix the bugs in `ComparativeViewer Bug Fix List`, and build some of the features in the `ComparativeViewer Feature Request Log`.
4) Finally, we will build a prototype of a new viewer, the `MultisliceViewer`, as described in the `MultisliceViewer` section.

Each task should be performed one at a time. Tasks are designated by "TASK {N}".

## ComparativeViewer PyTests Setup
This task is focused on creating a more serious set of python tests which test the following cases.

### Basic Feature tests to add

TASK 1: Translate current test scripts to pytest.
- [x] Pytest should have different levels of exhaustion. For "basic", include the tests done in `test_blank.py`, `test_cplx.py`, `test_dim_input.py`, and `test_single_image.py`. All other tests should be in "full" pytest mode.
- [x] Translate current test scripts to pytests
- [x] Also add a basic test to ensure different data input modes are supported, similar to how `test_dim_input.py` ensures different dimensional input is OK. this includes
    - [x] Dict of numpy inputs for multiple datasets
    - [x] Single array for viewing only one dataset
- [x] Move "basic" tests into `tests/comparative/basic`.
- [x] Move the rest of the tests into `tests/comparative/load`
TASK 2: Add one basic pytest for GUI interactive features to test editing the value of the sdim widget.
- [x] Add scripted ways to test GUI features through pytest
- [x] Test should load a viewer (using dataset from test_cmplx.py), simulate `editing` the value of the sdim widget, and then ensure no errors pop up.
- [x] If possible, test should also ensure the viewer updates as desired.
- [x] These tests will go into `tests/comparative/gui`
TASK 3: Implement interactive testing for exhaustive features.
- [ ] Implement tests to exhaust the feature list and put in `tests/comparative/gui`. These can be done on one loaded dataset, rather than loading a new dataset for each test, for speed.

For Scripting
- [ ] Add a unique CSS class identifier css class to each widget, which can be found with page.locator

## Refactor

TASK 4: Perform refactor
One issue with the `ComparativeViewer` is that using widgets to update the display of the `NDSlicer` is easy, but having the `NDSlicer` trigger updates to the widgets gets a bit messy.

A way around this might be to treat all interactable / servable objects in the backend as some generalized `Widget`. This object would have the property to subscribe to callbacks, be assigned to a viewer, etc. Exact implementation of the functionality is still to be defined, but a prototype of this is available on branch `refactor` from several months ago, specifically in `src/pyeyes/gui/widget.py` and `src/pyeyes/viewer2.py`.

In general, even the `Slicer` object would be considered a `Widget`, as it is displayed in the GUI, can be interacted with, and then upon some update to it's properties, might require update to other widgets. For example, if the user scrolled their mouse over the frame produced by the `NDSlicer`, then if a new slice is viewed, the slider showing the slice index along that dimension should also be updated. This could be done by having the `sdim` EditableIntSlider "subscribe" to changes in some parameter of the `NDSlicer`.

This is not an essential task but might help with making things more flexible and clean for future features.
TASK 4 END

TASK 4.5
- might also be nice to have each widget class have a string returned for how to locate it

## ComparativeViewer Bug Fix List
TASK 5
- [ ] Issue with ROI indexing and flipping L/R and U/D buttons if these display ranges are set to crop within the full range. Documentation on this TODO

## ComparativeViewer Feature Request Log
TASK 6
- [ ] `mouse_scroll`: This feature should allow the user to scroll through a slice of images when moving the mouse scroll wheel when the mouse is over the display image.
    - this should automatically scroll along the last modified sliceable dimension, or by default the first dimension in the list of sliceable dims.
    - Every time the view_dims is updated, this scrollable dimension should be updated to the first dimension in sliceable dims
    - categorical dimensions should also be scrollable
    - this may require disabling some of the scroll-based interactive features in the default Bokeh window
TASK 7
- [ ] Add widgets on the `Export` tab to support exporting the dynamic runnable format of the viewer
TASK 8
- [ ] Add toggle for "Single View" mode on display images to be short-cutted by keyboard inputs (possibly up/down arrows or left-right arrows?). If the user has checked `Single View` button, then clicking the up/down buttons on the keyboard will trigger moving along the list of possible displayed images.
TASK 9
- [ ] Add checkbox widget to toggle if titles are displayed or not.
TASK 10
- [ ] Add option to normalize DisplayImages to reference on the Analysis tab, similarly to how the Error maps are optionally normalized.
- [ ] Allow renaming of Display Image titles / names. This can be edited by a set of text boxes. This should alos be combined with allowing the user to modify the order of display images by dragging and dropping names along the list.
TASK 11
- [ ] For the error bars, if the scale of the image is either: (a) max value less than 0.1, or (b) max value above 100, then display the colorbar labels in scientific notation with 1 decimal. Otherwise, display in float notation with exactly 3 digits max (including integer component)
TASK 12
- [ ] `popup-pixel`: If the user clicks their mouse anywhere on a display image, a popup should then display the value of the pixel where they clicked their mouse, for all display images at that pixel location.
TASK 13
- [ ] On the `Analysis` Tab, add a widget which specifies to what precision the metrics should be displayed. Also add a widget that allows the metrics to instead be displayed in scientific notation, which is just a toggle (always set to 3 sig figs for scientific notation)
TASK 14
- [ ] Add an "Auto-crop" button to set the L/R and U/D crops such that white-space in the image is cut. This can be overfit to the displayed slice and done with a simple energy measurement.
TASK 15
- [ ] Support masking, where if some pixels in the input image are "nan", then display these as transparent.
TASK 16
- [ ] Add option to add borders for gridspec layout. This would be controlled with a button on the "View" panel.


## MultisliceViewer

This viewer should take a single dataset with the same inputs as the `ComparativeViewer`, except that the `data` includes only one numpy array (not multiple datasets). Then by default the viewer will show just one 2D image, which is sliceable as the ComparativeViewer normally is (change viewing dims, change slice dims, change colormap limits, etc). But then, the user will be able to press a "+" button to the right of this display, which will add a second un-linked view of the data, which can also be modified with the same controls in a separate panel. this can continue up to 4 different simultaneous views.

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

As tasks:
- TASK 17: Implement a basic version of MultiSlice Viewer with just the 2D pane option and Export feautre
- TASK 18: Add the 1D viewer functionality
