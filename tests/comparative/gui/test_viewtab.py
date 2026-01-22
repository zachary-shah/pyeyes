"""
Test case: Flat data with zeros by default.
Tests that viewer can handle data where one slice is all zeros.
"""

import time

import numpy as np
import pytest

from pyeyes.viewers import ComparativeViewer

# Timeout for GUI tests (in seconds)
GUI_TEST_TIMEOUT = 10


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_blank_data_and_vdim_swaps(viewer_page):
    """Test viewer with data that has a blank (all zeros) slice by default. Ensure we can swap viewing dimensions."""
    data = np.zeros((100, 100, 3))
    data[:, :, 0] = 1
    data[:, :, 2] = 3

    viewer = ComparativeViewer(data=data, named_dims=["x", "y", "z"])
    viewer, page, server = viewer_page(viewer)

    # Verify it exists
    assert viewer is not None
    assert viewer.slicer is not None
    assert "x" in viewer.slicer.ndims
    assert "y" in viewer.slicer.ndims
    assert "z" in viewer.slicer.ndims

    # Verify we can interact with the viewer
    LR = page.get_by_label("L/R Viewing Dimension")
    UD = page.get_by_label("U/D Viewing Dimension")

    LR.select_option("z")
    time.sleep(0.15)
    assert viewer.slicer.sdims == ["x"]
    UD.select_option("x")
    time.sleep(0.15)
    assert viewer.slicer.sdims == ["y"]
    LR.select_option("y")
    time.sleep(0.15)
    assert viewer.slicer.sdims == ["z"]

    server.stop()


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_single_view_toggle(viewer_page):
    """
    Test that the 'Single View' checkbox can be toggled via Playwright.

    This test verifies:
    1. The viewer can be launched and accessed via browser
    2. The 'Single View' checkbox widget can be found and clicked
    3. The viewer state updates correctly when the checkbox is toggled
    """

    data = np.zeros((100, 100, 3))
    data[:, :, 0] = 1
    data[:, :, 2] = 3
    data2 = np.zeros((100, 100, 3))
    data_dict = {
        "flats": data,
        "zero": data2,
    }

    # Create viewer
    viewer = ComparativeViewer(data=data_dict, named_dims=["x", "y", "z"])

    # Launch viewer and get Playwright page
    viewer, page, server = viewer_page(viewer)

    # Initial state: single_image_toggle should be False
    assert (
        viewer.single_image_toggle is False
    ), "Initial single_image_toggle should be False"
    assert (
        len(viewer.display_images) == 2
    ), "Initial display_images should have 2 images"

    # Find and click the "Single View" checkbox
    # Panel checkboxes render as <input type="checkbox"> with a label
    single_view_checkbox = page.locator(".pyeyes-single-view input[type='checkbox']")
    assert single_view_checkbox.count() > 0, "Single View checkbox not found in page"

    # Click the checkbox to toggle it
    single_view_checkbox.check()
    page.wait_for_timeout(500)

    # Verify the viewer state changed
    assert (
        viewer.single_image_toggle is True
    ), "single_image_toggle should be True after clicking"
    assert (
        len(viewer.display_images) == 1
    ), "display_images should have 1 image after clicking"

    # Toggle it back
    single_view_checkbox.uncheck()
    page.wait_for_timeout(500)

    # Verify it toggled back
    assert (
        viewer.single_image_toggle is False
    ), "single_image_toggle should be False after second click"
    assert (
        len(viewer.display_images) == 2
    ), "display_images should have 2 images after second click"

    server.stop()
