# Pyeyes Tests

This directory contains the test suite for the pyeyes package.

## Test Organization

```
tests/
├── comparative/              # ComparativeViewer tests
│   ├── basic/               # Basic functionality tests (fast)
│   ├── load/                # Full load tests with larger datasets
│   ├── gui/                 # GUI interaction tests with Playwright
│   └── conftest.py          # Shared fixtures
├── prototypes/              # Prototype viewer tests (deprecated)
├── cfgs/                    # Configuration files for tests
├── test-data/               # Test data files (numpy arrays)
│   ├── af/                  # Autofocus data
│   ├── dwi/                 # Diffusion-weighted imaging data
│   ├── mrf/                 # MRF data
│   └── se/                  # Spin echo data
└── exports/                 # Export test outputs
```

## Test Categories

### Basic Tests (`-m basic`)

Basic tests cover fundamental functionality and are designed to run quickly:

- **test_blank.py**: Tests handling of flat and zero-valued data inputs
- **test_cplx.py**: Tests complex-valued image data, config loading, and metric computation
- **test_dim_input.py**: Tests various formats for dimensional input (strings, lists, etc.)
- **test_single_image.py**: Tests single image viewing with configs and categorical dimensions
- **test_data_input_modes.py**: Tests different data input modes (dict, single array, high-dimensional, etc.)

```bash
pytest -m basic
```

### Full Load Tests (`-m load`)

Full load tests use larger datasets and test more comprehensive scenarios:

- **test_mrf.py**: MRF data with multiple images, categorical dimensions, and configs
- **test_diffusion.py**: Diffusion-weighted imaging data with three reconstruction methods
- **test_cfg_compat.py**: Config compatibility with different dataset names
- **test_multiviewers.py**: Multiple viewers with different datasets (autofocus + spin echo)
- **test_export_reloadable.py**: Export functionality to reloadable Python scripts

```bash
pytest -m load
```

### GUI Tests (`-m gui`)

GUI tests use Playwright to interact with the viewer in a headless browser. These tests verify that all widgets are functional and respond correctly to user interactions.

#### Tab Navigation Tests (`test_tab_navigation.py`)
- **test_tab_navigation_by_text**: Verifies all tabs can be navigated by clicking tab headers
- **test_tab_navigation_cycle**: Tests cycling through all tabs in sequence
- **test_view_tab_widgets_present**: Verifies View tab widgets exist
- **test_contrast_tab_widgets_present**: Verifies Contrast tab widgets exist
- **test_roi_tab_widgets_present**: Verifies ROI tab widgets exist
- **test_analysis_tab_widgets_present**: Verifies Analysis tab widgets exist
- **test_export_tab_widgets_present**: Verifies Export tab widgets exist

#### View Tab Tests (`test_view_features.py`)
Uses SE (spin echo) data via `se_viewer` fixture:
- **test_view_title_font_size**: Tests changing title font size via slider
- **test_view_size_scale**: Tests changing image size scale via slider
- **test_view_flip_ud_lr**: Tests flip U/D and flip L/R checkboxes
- **test_view_sdim_slider_navigation**: Tests navigating slice dimensions via slider

#### Contrast Tab Tests (`test_contrast_features.py`)
Uses SE data via `se_viewer` fixture:
- **test_contrast_cplx_view_mag_phase_switching**: Tests switching between magnitude/phase views with cmap restoration
- **test_contrast_colorbar_toggle**: Tests toggling colorbar on/off
- **test_contrast_colorbar_label**: Tests changing colorbar label
- **test_contrast_clim_modification**: Tests modifying contrast limits
- **test_contrast_cmap_change**: Tests changing colormap

#### ROI Tab Tests (`test_roi_features.py`)
Uses DWI (diffusion-weighted imaging) data via `dwi_viewer` fixture:
- **test_roi_overlay_enabled_checkbox**: Tests toggling ROI overlay mode
- **test_roi_clear_button**: Tests the Clear ROI button
- **test_roi_cmap_change_to_jet**: Tests changing ROI colormap
- **test_roi_overlay_and_location_change**: Tests ROI overlay and location selection
- **test_roi_zoom_order_change**: Tests changing ROI zoom order
- **test_roi_all_widgets_exist**: Verifies all ROI widgets exist

#### Analysis Tab Tests (`test_analysis_features.py`)
Uses SE data via `se_viewer` fixture:
- **test_analysis_change_reference_dataset**: Tests changing reference dataset for metrics
- **test_analysis_normalize_error_map**: Tests toggling error map normalization
- **test_analysis_error_cmap_change**: Tests changing error map colormap
- **test_analysis_metrics_location**: Tests moving metrics text location
- **test_analysis_autoformat_error_map**: Tests autoformat button
- **test_analysis_error_map_type_change**: Tests changing error map type

#### Export Tab Tests (`test_export_features.py`)
Uses SE data via `se_viewer` fixture:
- **test_export_config_to_temp_location**: Tests exporting config YAML to temp file
- **test_export_html_to_temp_location**: Tests exporting HTML to temp file
- **test_export_widgets_exist**: Verifies all Export widgets exist

```bash
pytest -m gui
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run only basic tests (fast)
```bash
pytest -m basic
```

### Run only full load tests
```bash
pytest -m load
```

### Run only GUI tests
```bash
pytest -m gui
```

### Run tests in a specific file
```bash
pytest tests/comparative/gui/test_view_features.py
```

### Run a specific test
```bash
pytest tests/comparative/gui/test_view_features.py::test_view_flip_ud_lr
```

### Run with verbose output and print statements
```bash
pytest -v -s
```

### Run with verbose output for GUI tests (recommended for debugging)
```bash
pytest -m gui -v -s
```

## Test Timeouts

All tests have a default timeout of 15 seconds configured in `pytest.ini`. GUI tests may override this with `@pytest.mark.timeout()` for operations that take longer (e.g., HTML export uses 30 seconds).

## Test Fixtures

Shared test fixtures are defined in `tests/comparative/conftest.py`:

### Data Fixtures
- `data_path`: Path to test data directory
- `cfg_path`: Path to test config directory
- `se_data`: Spin echo images dictionary (`4avg` and `1avg`)
- `af_data`: Autofocus datasets
- `dwi_data`: DWI datasets (`festive`, `skope`, `uncorr`)
- `mrf_data`: MRF datasets (`1min` and `2min`)

### Viewer Fixtures
- `se_viewer`: ComparativeViewer with SE data and `cfg_cplx.yaml` config
- `dwi_viewer`: ComparativeViewer with DWI data and `cfg_diff.yaml` config

### Server Fixtures
- `launched_viewer`: Launches a viewer silently for testing server startup
- `viewer_page`: Launches viewer + headless browser for GUI testing (Playwright)

### Utility Fixtures
- `navigate_to_tab`: Function to navigate to a specific tab by name
- `isclose`: Tolerance-based comparison function
- `cplx_slc_data`: Extracts slice data and metrics from a viewer

### Using the `viewer_page` Fixture (Playwright)

The `viewer_page` fixture launches a viewer server and provides a Playwright browser page:

```python
@pytest.mark.gui
def test_gui_interaction(viewer_page, se_viewer, navigate_to_tab):
    """Test GUI widget interaction with Playwright."""
    # Launch viewer and get Playwright page
    viewer, page, server = viewer_page(se_viewer)

    # Navigate to a specific tab
    navigate_to_tab(page, "Contrast")

    # Find and interact with widgets using CSS class selectors
    cmap_widget = page.locator(".pyeyes-cmap")
    cmap_select = cmap_widget.locator("select")
    cmap_select.select_option("jet")
    page.wait_for_timeout(1000)

    # Verify viewer state changed
    assert viewer.slicer.cmap == "jet"

    server.stop()
```

### Widget Interaction Patterns

GUI tests use Playwright to interact with Panel widgets. Common patterns:

#### Select Dropdowns
```python
widget = page.locator(".pyeyes-cmap")
select = widget.locator("select")
select.select_option("jet")
page.wait_for_timeout(1000)
```

#### Checkboxes
```python
widget = page.locator(".pyeyes-flip-ud")
checkbox = widget.locator("input[type='checkbox']")
checkbox.check()  # or checkbox.uncheck()
page.wait_for_timeout(1000)
```

#### Numeric Inputs (Sliders/IntInput)
```python
widget = page.locator(".pyeyes-size-scale")
input_el = widget.locator(":scope input:visible")
input_el.fill("600")
input_el.press("Enter")
page.wait_for_timeout(1000)
```

#### Text Inputs
```python
widget = page.locator(".pyeyes-colorbar-label")
input_el = widget.locator("input[type='text']")
input_el.fill("My Label")
input_el.press("Enter")
page.wait_for_timeout(1000)
```

#### Textareas
```python
widget = page.locator(".pyeyes-export-config-path")
textarea = widget.locator("textarea")
textarea.fill("/path/to/file.yaml")
textarea.press("Tab")
page.wait_for_timeout(1000)
```

#### Buttons
```python
widget = page.locator(".pyeyes-export-config-button")
button = widget.locator("button")
button.click()
page.wait_for_timeout(1000)
```

#### Toggle Buttons (e.g., cplx_view)
```python
widget = page.locator(".pyeyes-cplx-view")
phase_btn = widget.get_by_role("button", name="phase", exact=True)
phase_btn.click()
page.wait_for_timeout(1000)
```

## Requirements

Tests require pytest and pytest-timeout to be installed:
```bash
pip install pytest pytest-timeout
```

The full pyeyes package must also be installed:
```bash
pip install -e .
```

### GUI Testing with Playwright

For GUI interaction tests (marked with `@pytest.mark.gui`), you need pytest-playwright:

```bash
# Install test dependencies
pip install -e ".[test]"

# Install Playwright browsers (required once)
playwright install chromium
```

## CSS Class Naming Convention

All interactive widgets in pyeyes have CSS classes following the pattern `pyeyes-<widget-name>`. These are used by Playwright to locate and interact with widgets:

| Widget | CSS Class |
|--------|-----------|
| L/R Viewing Dimension | `pyeyes-vdim-lr` |
| U/D Viewing Dimension | `pyeyes-vdim-ud` |
| Flip U/D | `pyeyes-flip-ud` |
| Flip L/R | `pyeyes-flip-lr` |
| Size Scale | `pyeyes-size-scale` |
| Title Font Size | `pyeyes-title-font-size` |
| Slice Dimension (z) | `pyeyes-sdim-z` |
| Complex View | `pyeyes-cplx-view` |
| Contrast Limits | `pyeyes-clim` |
| Colormap | `pyeyes-cmap` |
| Colorbar Toggle | `pyeyes-colorbar` |
| Colorbar Label | `pyeyes-colorbar-label` |
| ROI Mode | `pyeyes-roi-mode` |
| Draw ROI | `pyeyes-draw-roi` |
| Clear ROI | `pyeyes-clear-roi` |
| ROI Colormap | `pyeyes-roi-cmap` |
| ROI Location | `pyeyes-roi-loc` |
| Reference Dataset | `pyeyes-reference-dataset` |
| Error Map Type | `pyeyes-error-map-type` |
| Error Map Colormap | `pyeyes-error-cmap` |
| Error Map Normalize | `pyeyes-error-normalize` |
| Export Config Path | `pyeyes-export-config-path` |
| Export Config Button | `pyeyes-export-config-button` |
| Export HTML Path | `pyeyes-export-html-path` |
| Export HTML Button | `pyeyes-export-html-button` |
