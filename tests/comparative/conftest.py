"""
Shared fixtures and configuration for comparative viewer tests.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pyeyes.viewers import ComparativeViewer

TEST_PORT = 52891


@dataclass
class CplxSlcData:
    """Container for complex slice data and metrics."""

    imgs: np.ndarray
    l1diff: float
    rmse: float
    nrmse: float
    psnr: float
    ssim: float


@pytest.fixture
def isclose():
    """
    Fixture providing a tolerance-based comparison function.

    Returns a function that checks if x is close to y within tol% tolerance.

    Usage:
        def test_something(isclose):
            assert isclose(1.01, 1.0, tol=0.02)  # True, within 2%
    """

    def _isclose(x, y, tol=0.03):
        """Ensure x is close to y by tol% (default 1%)."""
        return (np.abs(x - y) / np.abs(y)) < tol

    return _isclose


@pytest.fixture
def cplx_slc_data():
    """
    Fixture providing a function to extract slice data and metrics from a viewer.

    Returns a function that takes a ComparativeViewer and optional image key,
    and returns a CplxSlcData object with images and computed metrics.

    Usage:
        def test_viewer_metrics(cplx_slc_data, launched_viewer):
            viewer = ComparativeViewer(...)
            server = launched_viewer(viewer)
            data = cplx_slc_data(viewer, ikey="reference_img")
            assert data.nrmse < 0.2
    """

    def _cplx_slc_data(viewer: ComparativeViewer, ikey: str = "1avg"):
        """
        Extract slice data and metrics from a viewer.

        Parameters
        ----------
        viewer : ComparativeViewer
            The viewer instance to extract data from
        ikey : str
            The reference image key (default: "1avg")

        Returns
        -------
        CplxSlcData
            Container with images and metrics (l1diff, rmse, nrmse, psnr, ssim)
        """
        data = viewer.slicer.slice()
        metrics = data["metrics"]
        assert ikey in metrics, f"Reference image {ikey} not found in metrics"

        imgs = np.stack([d.data["Value"] for d in data["img"].values()])
        imgs[np.isnan(imgs)] = 0

        def getkey(metrics, ikey, key):
            if key in metrics[ikey]:
                out = metrics[ikey][key]
                try:
                    out = out.item()
                except Exception:
                    pass
                return out
            return np.nan

        return CplxSlcData(
            imgs=imgs,
            l1diff=getkey(metrics, ikey, "L1Diff"),
            rmse=getkey(metrics, ikey, "RMSE"),
            nrmse=getkey(metrics, ikey, "NRMSE"),
            psnr=getkey(metrics, ikey, "PSNR"),
            ssim=getkey(metrics, ikey, "SSIM"),
        )

    return _cplx_slc_data


@pytest.fixture
def data_path():
    """Path to the tracked test data directory."""
    test_data_folder = Path(__file__).parent.parent / "test-data"
    if not test_data_folder.exists():
        raise FileNotFoundError(f"Test data folder not found at {test_data_folder}")
    return test_data_folder


@pytest.fixture
def cfg_path():
    """Path to test config directory."""
    return Path(__file__).parent.parent / "cfgs"


@pytest.fixture
def se_data(data_path):
    """Load spin echo image dictionary for testing."""
    return {
        "4avg": np.load(data_path / "se" / "se_1avg.npy"),
        "1avg": np.load(data_path / "se" / "se_4avg.npy"),
    }


@pytest.fixture
def af_data(data_path):
    """Load autofocus dataset for testing."""
    af_folder = data_path / "af"
    return {
        "af": np.load(af_folder / "x_af.npy"),
        "no_comp": np.load(af_folder / "x_no_comp.npy"),
        "smooth": np.load(af_folder / "x_smooth.npy"),
        "gt": np.load(af_folder / "x_gt.npy"),
    }


@pytest.fixture
def dwi_data(data_path):
    """Load DWI data for testing."""
    dwi_folder = data_path / "dwi"
    return {
        "festive": np.load(dwi_folder / "festive.npy"),
        "skope": np.load(dwi_folder / "skope.npy"),
        "uncorr": np.load(dwi_folder / "uncorr.npy"),
    }


@pytest.fixture
def mrf_data(data_path):
    """Load MRF data for testing."""
    mrf_folder = data_path / "mrf"
    llr_1min_pd = np.load(mrf_folder / "llr_1min_pd.npy")
    llr_1min_t1 = np.load(mrf_folder / "llr_1min_t1.npy")
    llr_1min_t2 = np.load(mrf_folder / "llr_1min_t2.npy")
    llr_2min_pd = np.load(mrf_folder / "llr_2min_pd.npy")
    llr_2min_t1 = np.load(mrf_folder / "llr_2min_t1.npy")
    llr_2min_t2 = np.load(mrf_folder / "llr_2min_t2.npy")

    mrf_1min = np.stack([llr_1min_pd, llr_1min_t1, llr_1min_t2], axis=0)
    mrf_2min = np.stack([llr_2min_pd, llr_2min_t1, llr_2min_t2], axis=0)

    return {
        "1min": mrf_1min,
        "2min": mrf_2min,
    }


@pytest.fixture
def se_viewer(se_data, cfg_path):
    """Create a viewer with SE data for Analysis tab tests."""
    print("[se_viewer] Creating ComparativeViewer with SE data...")
    viewer = ComparativeViewer(
        data=se_data, named_dims="xyz", config_path=cfg_path / "cfg_cplx.yaml"
    )
    print(f"[se_viewer] Viewer created. Image names: {viewer.img_names}")
    return viewer


@pytest.fixture
def dwi_viewer(dwi_data, cfg_path):
    """Create a viewer with DWI data for ROI tab tests."""
    print("[dwi_viewer] Creating ComparativeViewer with DWI data...")
    viewer = ComparativeViewer(
        data=dwi_data,
        named_dims=["Bdir", "x", "y", "z"],
        config_path=cfg_path / "cfg_diff.yaml",
    )
    print(f"[dwi_viewer] Viewer created. Image names: {viewer.img_names}")
    return viewer


@pytest.fixture
def launched_viewer():
    """
    Fixture that launches a viewer silently and cleans up after test.

    Usage:
        def test_viewer_launch(launched_viewer):
            viewer = ComparativeViewer(data=my_data, named_dims=['x', 'y', 'z'])
            server = launched_viewer(viewer)
            # Test code here
            # Server is automatically stopped after test

    Parameters
    ----------
    viewer : ComparativeViewer
        The viewer instance to launch
    timeout : float
        Time to wait for server to start (default: 1.0 seconds)

    Returns
    -------
    server : panel.server.Server
        The running server instance
    """
    import time

    servers = []

    def launch(viewer: ComparativeViewer, timeout=0.5):
        """Launch viewer and verify server starts within timeout."""
        # Start server in background thread
        server = viewer.launch(
            show=False, start=True, threaded=True, verbose=True, port=TEST_PORT
        )
        servers.append(server)

        # Give server time to start
        time.sleep(timeout)

        return server

    yield launch

    # Cleanup: stop all servers
    for server in servers:
        try:
            server.stop()
        except Exception:
            pass


@pytest.fixture
def viewer_page():
    """
    Fixture that launches a viewer and provides a Playwright page connected to it.

    This fixture enables GUI testing by:
    1. Starting the viewer server on a known port
    2. Launching a headless browser
    3. Navigating to the viewer page
    4. Providing the page object for interaction
    5. Cleaning up browser and server after test

    Usage:
        def test_gui_interaction(viewer_page):
            viewer, page, server = viewer_page(my_viewer)

            # Click a checkbox by its label
            page.get_by_label("Single View").click()

            # Verify the viewer state changed
            assert viewer.single_image_toggle == True
    """
    import time

    from playwright.sync_api import sync_playwright

    resources = []  # Track (server, browser, playwright) for cleanup

    def launch(viewer: ComparativeViewer, timeout=0.5):
        """
        Launch viewer and return (viewer, page, server) tuple.

        Parameters
        ----------
        viewer : ComparativeViewer
            The viewer instance to launch
        timeout : float
            Time to wait for server to start (default: 0.5 seconds)

        Returns
        -------
        tuple : (viewer, page, server)
            The viewer, Playwright page, and server objects
        """
        # Start server
        server = viewer.launch(
            show=False, start=True, threaded=True, verbose=False, port=TEST_PORT
        )
        time.sleep(timeout)

        # Launch headless browser
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to viewer
        page.goto(f"http://localhost:{TEST_PORT}")
        page.wait_for_load_state("networkidle")

        # Give Panel time to render widgets
        time.sleep(timeout)

        resources.append((server, browser, pw))

        return viewer, page, server

    yield launch

    # Cleanup
    for server, browser, pw in resources:
        try:
            browser.close()
        except Exception:
            pass
        try:
            pw.stop()
        except Exception:
            pass
        try:
            server.stop()
        except Exception:
            pass


@pytest.fixture
def navigate_to_tab():
    """
    Fixture providing a function to navigate to a specific tab in the ComparativeViewer.

    This fixture handles clicking on tab headers in the Panel Tabs widget.

    Usage:
        def test_contrast_tab(viewer_page, navigate_to_tab):
            viewer, page, server = viewer_page(my_viewer)
            navigate_to_tab(page, "Contrast")
            # Now the Contrast tab is active
    """

    def _navigate_to_tab(page, tab_name: str, timeout: int = 300):
        """
        Navigate to a specific tab by clicking its header.

        Parameters
        ----------
        page : playwright.sync_api.Page
            The Playwright page object
        tab_name : str
            Name of the tab to navigate to. One of: "View", "Contrast", "ROI", "Analysis", "Export"
        timeout : int
            Milliseconds to wait after clicking for tab content to load (default: 300)

        Returns
        -------
        bool
            True if navigation succeeded, False otherwise
        """
        valid_tabs = ["View", "Contrast", "ROI", "Analysis", "Export"]
        if tab_name not in valid_tabs:
            raise ValueError(
                f"Invalid tab name '{tab_name}'. Must be one of: {valid_tabs}"
            )

        print(f"[navigate_to_tab] Attempting to navigate to '{tab_name}' tab...")

        # Try to find the tab header by its text content
        tab_header = page.locator(f"text='{tab_name}'").first

        if tab_header.count() == 0:
            print(f"[navigate_to_tab] ERROR: Tab header '{tab_name}' not found in page")
            return False

        print("[navigate_to_tab] Found tab header, clicking...")
        tab_header.click()
        page.wait_for_timeout(timeout)
        print(f"[navigate_to_tab] Successfully navigated to '{tab_name}' tab")
        return True

    return _navigate_to_tab


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "basic: marks tests as basic tests (deselect with '-m \"not basic\"')",
    )
    config.addinivalue_line("markers", "load: marks tests as full load tests")
    config.addinivalue_line(
        "markers", "gui: marks tests as GUI interaction tests (requires playwright)"
    )
