# User Guide

This guide provides an overview of pyeyes functionality and features. For installation and usage examples, see the [README](https://github.com/zachary-shah/pyeyes#readme). For  Python API, see [API](api_main.md). 

---

## What is pyeyes?

`pyeyes` is an interactive visualization tool for exploring arbitrarily high-dimensional data, designed specifically for MRI analysis but applicable to any multi-dimensional numpy arrays (including complex-valued data).

### Core Concepts

- **Dataset**: An N-dimensional array-like input
- **Named dimensions**: Descriptive names for each dimension (e.g., `["x", "y", "z", "time", "contrast"]`)
- **View dimensions** (`view_dims`): The 2D slice currently displayed (typically spatial dimensions like `["x", "y"]`)
- **Slice dimensions** (`slice_dims`): All other dimensions that can be navigated using sliders or selectors

The viewer allows you to interactively navigate through your data by changing which dimensions to view and which slice to display along the remaining dimensions.

---

## ComparativeViewer

The `ComparativeViewer` is the main interface for comparing multiple datasets of the same shape and dimensionality. It enables side-by-side visualization with synchronized navigation, quantitative comparison metrics, and flexible customization options.

The interface is organized into a figure window and a control panel, comprised of six panes, each providing specific functionality:

1. **View** - Control which dimensions to display and navigate through data
2. **Contrast** - Adjust color mapping and intensity ranges
3. **ROI** - Define and zoom into regions of interest
4. **Analysis** - Compute and display difference maps and comparative metrics
5. **Misc** - Customize display options, fonts, and enable pixel inspection
6. **Export** - Save configurations and export static visualizations

---

### Figure Window

The figure window displays the datasets at the current slice. In the top right corner of the figure is the **Bokeh Toolbar**, containing Bokeh-native navigation tools only for the currently displayed data. The most useful feature is the "Save" Icon to save the current displayed data as a `.png` file. 

### Viewer Control Panel

#### View Pane

| Feature | Description |
|---------|-------------|
| **View Dimension Selection** | Choose which 2D slice to display by selecting left/right (L/R) and up/down (U/D) view dimensions |
| **Slice Navigation** | Navigate through non-displayed dimensions using GUI wigets and scrolling over image with mouse wheel |
| **Categorical Dimensions** | Define categorical dimensions (e.g., contrasts) that use dropdown selectors instead of sliders |
| **Image Flipping** | Flip images along L/R or U/D axes |
| **Size Scaling** | Modify base image size with a slider |
| **Display Range Control** | Adjust pixel coordinate ranges to crop the displayed region (for cropping white-space) |
| **Single View Mode** | Toggle between grid view (view all images) and single image view |
| **Display Selection** | Click image names to show/hide specific datasets |

#### Contrast Pane

| Feature | Description |
|---------|-------------|
| **Complex Data Views** | Toggle between viewing the real, imaginary, magnitude, and phase components for complex-valued data |
| **Color Limits (clim)** | Adjust intensity range with interactive slider |
| **Colormap Selection** | Choose from various colormaps; supports quantitative maps (T1/T2) for MRF data |
| **Auto Scale** | Automatically set color limits and maps based on data percentiles and view type |
| **Colorbar** | Toggle colorbar display and customize label |

#### ROI Pane

| Feature | Description |
|---------|-------------|
| **Interactive ROI Drawing** | Click "Draw ROI" and select two corners to define a bounding box |
| **Zoom Control** | Set magnification factor for the ROI |
| **ROI Position** | Choose ROI location (top-right, bottom-left, etc.) or display as separate plot |
| **Crop Adjustment** | Fine-tune ROI boundaries with pixel-level crop controls |
| **Colormap** | Use same colormap as main image or select a different one |
| **Bounding Box Style** | Customize line color, width, and interpolation order |
| **Overlay Toggle** | Display ROI as overlay or in separate view below main images |

#### Analysis

| Feature | Description |
|---------|-------------|
| **Reference Selection** | Designate one dataset as reference (to be displayed on left-most side of figure) |
| **Error Maps** | Generate difference maps w.r.t. reference (Difference, Rel. Difference, L1, SSIM) |
| **Error Map Customization** | Adjust error scale, colormap, and use autoformat for optimal display |
| **Text Metrics** | Overlay quantitative metrics (PSNR, SSIM, NRMSE, etc.) on images or error maps |
| **Metrics Customization** | Control font size and placement of text metrics |
| **Image Normalization** | Normalize displayed images or only error metrics to reference |

#### Misc Pane

| Feature | Description |
|---------|-------------|
| **Font Selection** | Choose from system fonts for all text elements |
| **Title Display** | Toggle display of image titles and error map titles |
| **Grid Outline** | Enable grid lines to delineate images |
| **Pixel Inspection** | Click images to show popup with pixel values across all datasets |
| **Pixel Popup Options** | When pixel inspection is enabled, toggle coordinates, error map values, and location; or clear inspection |
| **Image Name Editing** | Customize display names for each dataset directly in the GUI |

#### Export Pane

| Feature | Description |
|---------|-------------|
| **Config Export** | Save current viewer settings to JSON file to re-launch GUI with same or modified data |
| **Static HTML Export** | Export interactive (but non-editable) HTML with sliceable DynamicMaps over the full or sub-sliced dataset |

### Additional Features

| Feature | Description |
|---------|-------------|
| **Config Import** | Load previously saved configurations as an input to the viewer through `config_path` |
| **Reloadable Python Export** | Generate standalone `.py` file to recreate the viewer (currently headless, via `viewer.export_reloadable_pyeyes()`) |
| **Mouse Wheel Scrolling** | Scroll through slices using mouse wheel along the most recently modified dimension |
| **State Caching** | Contrast and display settings are cached when switching between view types (e.g., magnitude ↔ phase) |
| **Multi-Viewer Support** | Launch multiple viewer instances simultaneously with `launch_viewers()` |
| **Detached Mode** | Launch viewer as a detached process with `launch_comparative_viewer(..., detached=True)` |
| **Theme Modification** | Change between light and dark mode before GUI launch with `pyeyes.set_theme()` |

---
