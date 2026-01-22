# Pyeyes Tests

This directory contains the test suite for the pyeyes package.

## Test Organization

The tests are organized into the following structure:

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

Basic tests cover fundamental functionality and are designed to run quickly. They include:

- **test_blank.py**: Tests handling of flat and zero-valued data inputs
- **test_cplx.py**: Tests complex-valued image data, config loading, and metric computation
- **test_dim_input.py**: Tests various formats for dimensional input (strings, lists, etc.)
- **test_single_image.py**: Tests single image viewing with configs and categorical dimensions
- **test_data_input_modes.py**: Tests different data input modes (dict, single array, high-dimensional, etc.)

Run basic tests only:
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

Run full load tests only:
```bash
pytest -m load
```

### GUI Tests (`-m gui`)

GUI tests use Playwright to interact with the viewer in a headless browser:

- **test_viewtab.py**: Tests View tab interactions including:
  - Swapping L/R and U/D viewing dimensions
  - Toggling "Single View" checkbox

Run GUI tests only:
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
pytest tests/comparative/basic/test_blank.py
```

### Run tests in a specific directory
```bash
pytest tests/comparative/basic/
```

### Run with verbose output
```bash
pytest -v
```

### Run with detailed output and print statements
```bash
pytest -v -s
```

## Test Fixtures

Shared test fixtures are defined in `tests/comparative/conftest.py`:

### Data Fixtures
- `data_path`: Path to test data directory
- `cfg_path`: Path to test config directory
- `se_image_dict`: Spin echo images (4avg and 1avg)
- `mrf_data`: MRF datasets (1min and 2min)
- `dwi_data`: DWI datasets (festive, skope, uncorr)

### Server Fixtures
- `launched_viewer`: Launches a viewer silently for testing server startup
- `viewer_page`: Launches viewer + headless browser for GUI testing (Playwright)

### Using the `launched_viewer` Fixture

The `launched_viewer` fixture allows tests to verify that the viewer can launch and start a server without opening a browser:

```python
@pytest.mark.basic
def test_viewer_launches(launched_viewer):
    """Test that viewer can launch without error."""
    viewer = ComparativeViewer(data=my_data, named_dims=["x", "y", "z"])

    # Launch silently (server starts in background, no browser)
    server = launched_viewer(viewer)

    # Verify server started and test viewer properties
    assert server is not None

    # Always stop the server when done
    server.stop()
```

## Requirements

Tests require pytest to be installed:
```bash
pip install pytest
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

### Using the `viewer_page` Fixture (Playwright)

The `viewer_page` fixture launches a viewer server and provides a Playwright browser page:

```python
@pytest.mark.gui
def test_gui_interaction(viewer_page):
    """Test GUI widget interaction with Playwright."""
    viewer = ComparativeViewer(data=my_data, named_dims=["x", "y", "z"])

    # Launch viewer and get Playwright page
    viewer, page, server = viewer_page(viewer)

    # Find and interact with widgets
    page.get_by_label("L/R Viewing Dimension").select_option("z")
    page.wait_for_timeout(150)

    # Verify viewer state changed
    assert "z" in viewer.slicer.vdims

    server.stop()
```
