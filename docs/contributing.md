
## Contributing

We welcome contributions to pyeyes! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### Development Setup

1. **Clone the repository**

```bash
git clone https://github.com/zachary-shah/pyeyes.git
cd pyeyes
```

2. **Create the development environment**

```bash
mamba env create -n pyeyes --file env.yaml
mamba activate pyeyes
pip install -e ".[dev,test,docs]"
pre-commit install
```

### Repository Structure

Understanding the codebase organization will help you navigate and contribute effectively:

```
src/pyeyes/
├── __init__.py           # Package exports and public API
├── app.py                # Application launchers (launch_viewers, etc.)
├── viewers.py            # ComparativeViewer and base Viewer classes
├── slicers.py            # NDSlicer for managing dimensional slicing
├── roi.py                # ROI functionality and drawing tools
├── metrics.py            # Error metrics (SSIM, NRMSE, etc.)
├── config.py             # Configuration save/load handling
├── themes.py             # Theme definitions and set_theme()
├── utils.py              # Utility functions (normalize, tonp, etc.)
├── error.py              # Error handling decorators
├── enums.py              # Enumerations for viewer settings
├── profilers.py          # Performance profiling decorators
├── gui/                  # GUI widget abstractions
│   ├── widget.py         # Base Widget class for panel widgets
│   ├── pane.py           # Pane management for tab organization
│   └── scroll.py         # Scroll-based slice navigation
├── cmap/                 # Colormap utilities
│   ├── cmap.py           # ColorMap and QuantitativeColorMap classes
│   └── *.csv             # Quantitative colormap data files
└── prototypes/           # Experimental/unsupported features
    ├── line.py           # 1D viewer prototype (launch_1d_viewer)
    └── mpl/              # Matplotlib-based diffusion viewer prototype
```

**Key Design Concepts:**

- **`gui/widget.py`**: Provides a generalized `Widget` interface for creating reusable panel components with consistent behavior
- **`viewers.py`**: Contains the main `ComparativeViewer` class which orchestrates all GUI tabs and data display
- **`slicers.py`**: Manages n-dimensional data slicing and caching of view-specific settings
- **`prototypes/`**: Contains experimental features not yet part of the stable API; do not rely on these for production use

### Running Tests

pyeyes uses `pytest` for automated testing and `playwright` for GUI interaction tests.

**Grab data for pytests:**

This can be done automatically with `tests/download_test_data.py`, or manually in bash with:

```bash
wget https://github.com/zachary-shah/pyeyes/releases/download/test-data-v0.4.0/test-data-v0.4.0.tar.gz
tar -xzf test-data-v0.4.0.tar.gz -C tests/
```

**Run the full test suite:**
```bash
pytest
```

**Run specific test categories:**
```bash
# Basic functionality tests
pytest -m basic

# GUI feature tests (requires playwright)
pytest -m gui

# Different configs and data scenarios
pytest -m load
```

**Interactive demos (not part of automated tests):**
```bash
# Files prefixed with 'ztest_' are runnable demos
python tests/ztest_mrf.py
python tests/ztest_diffusion.py
```

These demo files (`ztest_*.py`) launch full viewer instances for manual testing and are useful for understanding usage patterns, but they are excluded from the automated test suite.

**GUI tests with Playwright:**
GUI tests simulate user interactions (clicks, sliders, tab navigation) using Playwright. To run GUI tests, ensure Playwright browsers are installed:
```bash
playwright install
```
### Testing Guidelines

- **Basic tests** (`tests/comparative/basic/`): Test core functionality like data input modes, dimension handling, and initialization
- **GUI tests** (`tests/comparative/gui/`): Test interactive features using Playwright to simulate user actions
- **Load tests** (`tests/comparative/load/`): Test configuration loading, export functionality, and multi-viewer scenarios
- Use test data from `tests/test-data/` for consistency across tests
- Add new test configurations to `tests/cfgs/` if needed for your feature

### Questions?

For questions or discussion about contributing, please open an issue on [GitHub](https://github.com/zachary-shah/pyeyes/issues).
