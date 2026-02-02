# Changelog

This document provides a detailed changelog for pyeyes releases, documenting new features, improvements, and bug fixes in each version.

---

## v0.4.1

**Release Date:** February 02 2026

### Overview

Patch release addressing GUI improvements for pixel inspection and fixing an ROI boundary issue.

### Improvements

| Area | Description |
|------|-------------|
| Pixel inspection | Exposed size and color customization options for pixel popup markers |
| GUI layout | More compact Misc tab layout for pixel inspection controls |
| Dependencies | Updated package dependencies in `pyproject.toml` |

### Bug Fixes

| Issue | Description |
|-------|-------------|
| ROI bounding box | Fixed padding issue with ROI bounding box calculations |

---

## v0.4.0

**Release Date:** February 01 2026

### Overview

Introduces significant performance improvements, enhanced interactivity, and expanded customization options. This release focuses on making the viewer faster, more responsive, and easier to use with new mouse-based navigation and pixel inspection capabilities.

### New Features

| Category | Feature | Description |
|----------|---------|-------------|
| **API** | Input Arg Flexibility | `ComparativeViewer` Dimension inputs now accept multiple formats: (1) no input (default), (2) list of strings (e.g. `["Read", "PE", "Slice"]`), or (3) delimited strings (e.g., `"xyz"` or `"x,y,z"` for 3D) |
| **Interactivity** | MouseWheel-based navigation | Scroll through slice dimensions directly with mouse wheel. Automatically scrolls along last modified sliceable dimension (including categorical). Bokeh scroll-based zoom now disabled by default. |
| **Analysis** | Image Normalization | Options to normalize all images to selected reference dataset for display and error map computation |
| | Normalize for Error Metrics Only | Normalizes only for error map computation without affecting displayed images (previously "Normalize Error Maps") |
| ***New* Misc Tab** | Font Selection | Choose from system fonts for all text elements |
| | Display toggles | Optional display of image titles and error map titles |
| | Grid outline | Option to enable grid outline to delineate images |
| | Pixel popup inspection | Click-to-inspect pixel values across all displayed images. Disabled by default for performance. Features: optional pixel coordinates, toggle for error maps, configurable popup location, clear button |
| | Image title editing | Edit display names for each image directly in the GUI |

### Bug Fixes

| Issue | Description |
|-------|-------------|
| Overlay behavior | Fixed overlay behavior issues related to flipping or cropping image dimensions |
| In-figure overlays | Resolved inconsistent in-figure overlay locations for different image flip combinations |
| Widget bounds | Prevented widgets from incrementing out of bounds (e.g., slice dimensions below 0 or above maximum) |
| Error maps | Fixed NaN values in error maps for quantitative map cases and excessive NRMSE values |
| Figure updates | Reduced redundant figure update calls for certain widget events |
| Colorbar margins | Adjusted colorbar margin overflow |
| Numeric display | Improved numeric display consistency with `bokeh.models.BasicTickFormatter` integration and significant figure control |

---

## v0.3.0

**Release Date:** January 2026

### Overview

v0.3.0 focuses on improving error map visualization, expanding quantitative map support, enhancing configuration management, and fixing critical colorbar scaling issues. This release also introduces a prototype 1D viewer and adds support for exporting reloadable viewer instances.

### New Features

| Category | Feature | Description |
|----------|---------|-------------|
| **Error Map Enhancements** | Error metric overlays | Error metrics now overlay on difference maps by default when available. Metrics automatically relocate to error plot area if present, otherwise display on main figure |
| | Optional normalization | Normalizing error maps is now optional (previously always normalized) |
| **Quantitative Map Support** | T2s support | Added support for T2s in quantitative colormaps (uses Navia colormap) |
| | Improved colormap selection | Improved automatic colormap selection for MRF-like categorical dimensions |
| **Configuration & Export** | Config compatibility | Fixed config compatibility issues for error maps |
| | Parameter deprecation | Resolved parameter deprecation warnings when loading configs |
| | Reloadable viewer export | Added support for exporting reloadable instances of pyeyes viewer (`export_reloadable_pyeyes()`) |
| | Config serialization | Improved config serialization for object selection plot elements |
| **Viewer Improvements** | Title alignment | Default image titles now center-aligned rather than left-aligned |
| | Display image buttons | Display image buttons stack vertically to prevent overflow |
| | Hot colormap | Added `hot` colormap option |
| | Optional torch import | Made torch import optional (graceful fallback if not installed) |
| | Detached viewer | Added functionality to spawn detached viewer instances |
| **Prototype Features** | 1D viewer | Prototyped 1D viewer (`launch_1d_viewer`) for line plot visualization |

### Bug Fixes

| Issue | Description |
|-------|-------------|
| Colorbar scaling | Major fixes for vmin/vmax scaling with colorbar limits: fixed vmin out of bounds issues, handled flat/all-zero input cases, resolved GUI/backend inconsistencies |
| Error map formatting | Fixed scale rounding issue on autoformatting error maps |
| Tolerance handling | Cleaned up tolerances for low image scales |
| Error handling | Assigned global error handler only after launching a pyeyes viewer to prevent interference |

---

## v0.2.0

**Release Date:** March 2025

### Overview

v0.2.0 expands error map capabilities, improves input flexibility, adds export functionality, and introduces a multi-viewer launcher. This release also includes important fixes for phase error map calculations and singleton dimension handling.

### New Features

| Category | Feature | Description |
|----------|---------|-------------|
| **Error Map Types** | Relative L1 metric | Added Relative L1 difference as a text metric |
| | Difference maps | Added regular and relative difference maps |
| | SSIM maps | Added SSIM (Structural Similarity Index) maps |
| | Variable naming | Improved variable naming for difference metrics |
| **Input Flexibility** | Non-numpy inputs | Allow non-numpy inputs to viewer (e.g., PyTorch tensors) with automatic conversion to numpy arrays |
| | Single image handling | Improved handling of single-image datasets not provided as dictionaries |
| **Export Functionality** | Static HTML export | Added option to export interactive GUI to static but sliceable DynamicMap HTML. Export respects current viewing dimensions and supports reduced slice ranges for efficiency. Note: Limited support for categorical dimensions |
| **Multi-Viewer Support** | Multi-viewer launcher | Added multi-viewer launcher (`launch_viewers()`) for launching multiple viewer instances |
| | Port handling | Improved default port argument handling |

### Improvements

| Area | Description |
|------|-------------|
| Title customization | Freely editable title font size with improved scaling |
| Singleton dimensions | Removed sliders for singleton dimensions (dimensions with size 1) |
| Phase error maps | Updated formula for phase error maps to use correct phase difference calculation |
| Plot rendering | Fixed plot height bug for font scaling with data pipes |

### Bug Fixes

| Issue | Description |
|-------|-------------|
| Metrics calculations | Fixed divide-by-zero errors in metrics calculations |
| Config export | Fixed export from config bugs |
| Object serialization | Limited object serialization to user-changeable parameters only |
| View dimension changes | Improved handling of view dimension changes with singleton dimensions |

---

## v0.1.3

**Release Date:** January 2025

### Overview

v0.1.3 introduces core features that form the foundation of pyeyes' interactive capabilities: ROI support, configuration save/load, user notifications, improved error handling, and performance optimizations through data streaming.

### New Features

| Category | Feature | Description |
|----------|---------|-------------|
| **ROI Support** | Interactive region selection | Added comprehensive ROI feature for focusing on specific image regions. Draw ROI by selecting two corners to define bounding box. ROI can be displayed as overlay or in separate view. Customizable zoom scale, location, crops, line color, and line width |
| **Configuration Management** | Save-to-config | Added save-to-config functionality for repeatability |
| | Load-from-config | Added load-from-config functionality to restore viewer state. Config files store viewer, slicer, and ROI settings. Enables easy re-examination of similar datasets with consistent settings |
| **User Notifications** | Panel notifications | Integrated `panel.state.notifications` for user-facing messages. Error messages, warnings, and info notifications displayed in the GUI. Better user experience for error handling and status updates |
| **Error Handling** | Function decorators | Improved error handling with function decorators. Error handler decorator for graceful error display |
| | Global error handler | Global error handler for unhandled exceptions |
| **Performance Improvements** | Data streams | Improved refresh speed with data streams for slicer updates. Uses HoloViews Pipe streams for efficient data updates. Reduces unnecessary figure regeneration |

### Improvements

| Area | Description |
|------|-------------|
| Documentation | Included documentation in project build |
| Python requirements | Relaxed Python version requirements for broader compatibility |
| Release workflow | Added workflow for automatic release to PyPI |

---

## Version History Summary

| Version | Release Date | Key Features |
|---------|--------------|-------------|
| v0.4.1 | February 2026 | Additional popup customization, pip dependency fixes |
| v0.4.0 | February 2026 | Mouse scrolling, pixel popup, normalization options, Misc tab features |
| v0.3.0 | January 2026 | Error map overlays, reloadable viewer export, colorbar fixes, 1D viewer prototype |
| v0.2.0 | March 2025 | Additional error maps (SSIM, relative), HTML export, multi-viewer launcher |
| v0.1.3 | January 2025 | ROI support, config save/load, notifications, error handling, data streams |
