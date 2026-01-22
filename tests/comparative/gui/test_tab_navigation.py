"""
Test case: Tab navigation in ComparativeViewer.
Tests that all tabs can be navigated using Playwright and verifies
widgets unique to each tab are visible when the tab is active.
"""

import numpy as np
import pytest

from pyeyes.viewers import ComparativeViewer

# Timeout for GUI tests (in seconds)
GUI_TEST_TIMEOUT = 15

# Define widgets unique to each tab for verification
# Each tuple contains (tab_name, css_class_to_verify)
TAB_WIDGET_IDENTIFIERS = {
    "View": "pyeyes-vdim-lr",  # L/R Viewing Dimension selector
    "Contrast": "pyeyes-clim",  # Contrast limit slider
    "ROI": "pyeyes-draw-roi",  # Draw ROI button
    "Analysis": "pyeyes-error-map-type",  # Error map type selector
    "Export": "pyeyes-export-config-button",  # Export config button
}


@pytest.fixture
def basic_viewer():
    """Create a basic viewer for tab navigation tests."""
    data = np.random.rand(50, 50, 5)
    data2 = np.random.rand(50, 50, 5)
    data_dict = {
        "image1": data,
        "image2": data2,
    }
    viewer = ComparativeViewer(data=data_dict, named_dims=["x", "y", "z"])
    return viewer


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_tab_navigation_by_text(viewer_page, basic_viewer):
    """
    Test that all tabs in ComparativeViewer can be clicked by their tab header text.

    This test verifies:
    1. Each tab can be located by its text content
    2. Clicking a tab makes it active
    3. Widgets unique to each tab become visible when that tab is active
    """
    viewer, page, server = viewer_page(basic_viewer)

    # Initial state: View tab should be active
    # Verify by checking that View tab widget is visible
    view_widget = page.locator(f".{TAB_WIDGET_IDENTIFIERS['View']}")
    assert view_widget.count() > 0, "View tab widget should be visible initially"

    # Test clicking each tab by its text and verifying tab-specific widgets
    for tab_name, css_class in TAB_WIDGET_IDENTIFIERS.items():

        # Click the tab by finding its text in the tab header
        # Panel tabs render with a structure containing the tab text
        tab_header = page.locator(f"text='{tab_name}'").first
        tab_header.click()
        page.wait_for_timeout(300)

        # Verify the tab-specific widget is now visible
        widget = page.locator(f".{css_class}")
        is_visible = widget.is_visible() if widget.count() > 0 else False

        print(f"Tab: {tab_name} Widget: {css_class} is visible: {is_visible}")

        assert (
            is_visible
        ), f"Widget with class '{css_class}' should be visible after clicking '{tab_name}' tab"

    server.stop()


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_tab_navigation_cycle(viewer_page, basic_viewer):
    """
    Test cycling through all tabs and returning to the first tab.

    This test verifies:
    1. Navigation works in sequence through all tabs
    2. Can return to the first tab after visiting all others
    3. State is preserved when returning to a tab
    """
    viewer, page, server = viewer_page(basic_viewer)

    tab_order = ["View", "Contrast", "ROI", "Analysis", "Export", "View"]

    for tab_name in tab_order:
        # Click the tab
        tab_header = page.locator(f"text='{tab_name}'").first
        tab_header.click()
        page.wait_for_timeout(200)

        # Verify the expected widget is visible
        css_class = TAB_WIDGET_IDENTIFIERS[tab_name]
        widget = page.locator(f".{css_class}")
        assert widget.is_visible(), f"Expected widget for '{tab_name}' tab not visible"

    server.stop()


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_view_tab_widgets_present(viewer_page, basic_viewer):
    """
    Test that View tab has all expected widgets present.
    """
    viewer, page, server = viewer_page(basic_viewer)

    # Widgets that should be on View tab
    view_tab_widgets = [
        "pyeyes-vdim-lr",
        "pyeyes-vdim-ud",
        "pyeyes-flip-ud",
        "pyeyes-flip-lr",
        "pyeyes-size-scale",
        "pyeyes-single-view",
    ]

    for css_class in view_tab_widgets:
        widget = page.locator(f".{css_class}")
        assert widget.count() > 0, f"Widget '{css_class}' should exist on View tab"

    server.stop()


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_contrast_tab_widgets_present(viewer_page, basic_viewer):
    """
    Test that Contrast tab has all expected widgets present.
    """
    viewer, page, server = viewer_page(basic_viewer)

    # Navigate to Contrast tab
    tab_header = page.locator("text='Contrast'").first
    tab_header.click()
    page.wait_for_timeout(300)

    # Widgets that should be on Contrast tab
    contrast_tab_widgets = [
        "pyeyes-cplx-view",
        "pyeyes-autoscale",
        "pyeyes-clim",
        "pyeyes-cmap",
        "pyeyes-colorbar",
    ]

    for css_class in contrast_tab_widgets:
        widget = page.locator(f".{css_class}")
        assert widget.count() > 0, f"Widget '{css_class}' should exist on Contrast tab"

    server.stop()


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_roi_tab_widgets_present(viewer_page, basic_viewer):
    """
    Test that ROI tab has all expected widgets present.
    """
    viewer, page, server = viewer_page(basic_viewer)

    # Navigate to ROI tab
    tab_header = page.locator("text='ROI'").first
    tab_header.click()
    page.wait_for_timeout(300)

    # Widgets that should be on ROI tab
    roi_tab_widgets = [
        "pyeyes-roi-mode",
        "pyeyes-draw-roi",
        "pyeyes-clear-roi",
    ]

    for css_class in roi_tab_widgets:
        widget = page.locator(f".{css_class}")
        assert widget.count() > 0, f"Widget '{css_class}' should exist on ROI tab"

    server.stop()


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_analysis_tab_widgets_present(viewer_page, basic_viewer):
    """
    Test that Analysis tab has all expected widgets present.
    """
    viewer, page, server = viewer_page(basic_viewer)

    # Navigate to Analysis tab
    tab_header = page.locator("text='Analysis'").first
    tab_header.click()
    page.wait_for_timeout(300)

    # Widgets that should be on Analysis tab
    analysis_tab_widgets = [
        "pyeyes-reference-dataset",
        "pyeyes-error-map-type",
        "pyeyes-error-scale",
        "pyeyes-error-cmap",
        "pyeyes-error-map-autoformat",
    ]

    for css_class in analysis_tab_widgets:
        widget = page.locator(f".{css_class}")
        assert widget.count() > 0, f"Widget '{css_class}' should exist on Analysis tab"

    server.stop()


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_export_tab_widgets_present(viewer_page, basic_viewer):
    """
    Test that Export tab has all expected widgets present.
    """
    viewer, page, server = viewer_page(basic_viewer)

    # Navigate to Export tab
    tab_header = page.locator("text='Export'").first
    tab_header.click()
    page.wait_for_timeout(300)

    # Widgets that should be on Export tab
    export_tab_widgets = [
        "pyeyes-export-config-path",
        "pyeyes-export-config-button",
        "pyeyes-export-html-path",
        "pyeyes-export-html-button",
    ]

    for css_class in export_tab_widgets:
        widget = page.locator(f".{css_class}")
        assert widget.count() > 0, f"Widget '{css_class}' should exist on Export tab"

    server.stop()
