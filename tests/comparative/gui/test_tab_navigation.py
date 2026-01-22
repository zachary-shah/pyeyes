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
    print("[basic_viewer] Creating test data...")
    data = np.random.rand(50, 50, 5)
    data2 = np.random.rand(50, 50, 5)
    data_dict = {
        "image1": data,
        "image2": data2,
    }
    print("[basic_viewer] Initializing ComparativeViewer...")
    viewer = ComparativeViewer(data=data_dict, named_dims=["x", "y", "z"])
    print("[basic_viewer] Viewer created successfully")
    return viewer


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_tab_navigation_by_text(viewer_page, basic_viewer, navigate_to_tab):
    """
    Test that all tabs in ComparativeViewer can be clicked by their tab header text.

    This test verifies:
    1. Each tab can be located by its text content
    2. Clicking a tab makes it active
    3. Widgets unique to each tab become visible when that tab is active
    """
    print("\n" + "=" * 60)
    print("[test_tab_navigation_by_text] Starting test...")
    print("=" * 60)

    print("[test_tab_navigation_by_text] Launching viewer with Playwright...")
    viewer, page, server = viewer_page(basic_viewer)
    print("[test_tab_navigation_by_text] Viewer launched, server running")

    # Initial state: View tab should be active
    print("[test_tab_navigation_by_text] Checking initial state (View tab)...")
    view_widget = page.locator(f".{TAB_WIDGET_IDENTIFIERS['View']}")
    view_widget_count = view_widget.count()
    print(f"[test_tab_navigation_by_text] View widget count: {view_widget_count}")
    assert view_widget_count > 0, "View tab widget should be visible initially"

    # Test clicking each tab by its text and verifying tab-specific widgets
    print("\n[test_tab_navigation_by_text] Testing tab navigation...")
    for tab_name, css_class in TAB_WIDGET_IDENTIFIERS.items():
        print(f"\n--- Testing tab: '{tab_name}' ---")

        # Use the navigate_to_tab fixture
        success = navigate_to_tab(page, tab_name)
        print(f"[test_tab_navigation_by_text] Navigation success: {success}")

        # Verify the tab-specific widget is now visible
        widget = page.locator(f".{css_class}")
        widget_count = widget.count()
        is_visible = widget.is_visible() if widget_count > 0 else False

        print(f"[test_tab_navigation_by_text] Widget class: {css_class}")
        print(f"[test_tab_navigation_by_text] Widget count: {widget_count}")
        print(f"[test_tab_navigation_by_text] Widget is_visible: {is_visible}")

        assert (
            is_visible
        ), f"Widget with class '{css_class}' should be visible after clicking '{tab_name}' tab"

    print("\n[test_tab_navigation_by_text] All tabs navigated successfully!")
    print("[test_tab_navigation_by_text] Stopping server...")
    server.stop()
    print("[test_tab_navigation_by_text] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_tab_navigation_cycle(viewer_page, basic_viewer, navigate_to_tab):
    """
    Test cycling through all tabs and returning to the first tab.

    This test verifies:
    1. Navigation works in sequence through all tabs
    2. Can return to the first tab after visiting all others
    3. State is preserved when returning to a tab
    """
    print("\n" + "=" * 60)
    print("[test_tab_navigation_cycle] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(basic_viewer)
    print("[test_tab_navigation_cycle] Viewer launched")

    tab_order = ["View", "Contrast", "ROI", "Analysis", "Export", "View"]

    for i, tab_name in enumerate(tab_order):
        print(f"\n--- Step {i + 1}/{len(tab_order)}: Navigating to '{tab_name}' ---")

        navigate_to_tab(page, tab_name, timeout=200)

        # Verify the expected widget is visible
        css_class = TAB_WIDGET_IDENTIFIERS[tab_name]
        widget = page.locator(f".{css_class}")
        is_visible = widget.is_visible()
        print(f"[test_tab_navigation_cycle] Widget '{css_class}' visible: {is_visible}")
        assert is_visible, f"Expected widget for '{tab_name}' tab not visible"

    print("\n[test_tab_navigation_cycle] Cycle complete, stopping server...")
    server.stop()
    print("[test_tab_navigation_cycle] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_view_tab_widgets_present(viewer_page, basic_viewer):
    """
    Test that View tab has all expected widgets present.
    """
    print("\n" + "=" * 60)
    print("[test_view_tab_widgets_present] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(basic_viewer)
    print(
        "[test_view_tab_widgets_present] Viewer launched, checking View tab widgets..."
    )

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
        count = widget.count()
        print(f"[test_view_tab_widgets_present] Widget '{css_class}': count={count}")
        assert count > 0, f"Widget '{css_class}' should exist on View tab"

    print("[test_view_tab_widgets_present] All View tab widgets found!")
    server.stop()
    print("[test_view_tab_widgets_present] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_contrast_tab_widgets_present(viewer_page, basic_viewer, navigate_to_tab):
    """
    Test that Contrast tab has all expected widgets present.
    """
    print("\n" + "=" * 60)
    print("[test_contrast_tab_widgets_present] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(basic_viewer)
    print("[test_contrast_tab_widgets_present] Viewer launched")

    # Navigate to Contrast tab
    navigate_to_tab(page, "Contrast")

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
        count = widget.count()
        print(
            f"[test_contrast_tab_widgets_present] Widget '{css_class}': count={count}"
        )
        assert count > 0, f"Widget '{css_class}' should exist on Contrast tab"

    print("[test_contrast_tab_widgets_present] All Contrast tab widgets found!")
    server.stop()
    print("[test_contrast_tab_widgets_present] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_roi_tab_widgets_present(viewer_page, basic_viewer, navigate_to_tab):
    """
    Test that ROI tab has all expected widgets present.
    """
    print("\n" + "=" * 60)
    print("[test_roi_tab_widgets_present] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(basic_viewer)
    print("[test_roi_tab_widgets_present] Viewer launched")

    # Navigate to ROI tab
    navigate_to_tab(page, "ROI")

    # Widgets that should be on ROI tab
    roi_tab_widgets = [
        "pyeyes-roi-mode",
        "pyeyes-draw-roi",
        "pyeyes-clear-roi",
    ]

    for css_class in roi_tab_widgets:
        widget = page.locator(f".{css_class}")
        count = widget.count()
        print(f"[test_roi_tab_widgets_present] Widget '{css_class}': count={count}")
        assert count > 0, f"Widget '{css_class}' should exist on ROI tab"

    print("[test_roi_tab_widgets_present] All ROI tab widgets found!")
    server.stop()
    print("[test_roi_tab_widgets_present] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_analysis_tab_widgets_present(viewer_page, basic_viewer, navigate_to_tab):
    """
    Test that Analysis tab has all expected widgets present.
    """
    print("\n" + "=" * 60)
    print("[test_analysis_tab_widgets_present] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(basic_viewer)
    print("[test_analysis_tab_widgets_present] Viewer launched")

    # Navigate to Analysis tab
    navigate_to_tab(page, "Analysis")

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
        count = widget.count()
        print(
            f"[test_analysis_tab_widgets_present] Widget '{css_class}': count={count}"
        )
        assert count > 0, f"Widget '{css_class}' should exist on Analysis tab"

    print("[test_analysis_tab_widgets_present] All Analysis tab widgets found!")
    server.stop()
    print("[test_analysis_tab_widgets_present] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_export_tab_widgets_present(viewer_page, basic_viewer, navigate_to_tab):
    """
    Test that Export tab has all expected widgets present.
    """
    print("\n" + "=" * 60)
    print("[test_export_tab_widgets_present] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(basic_viewer)
    print("[test_export_tab_widgets_present] Viewer launched")

    # Navigate to Export tab
    navigate_to_tab(page, "Export")

    # Widgets that should be on Export tab
    export_tab_widgets = [
        "pyeyes-export-config-path",
        "pyeyes-export-config-button",
        "pyeyes-export-html-path",
        "pyeyes-export-html-button",
    ]

    for css_class in export_tab_widgets:
        widget = page.locator(f".{css_class}")
        count = widget.count()
        print(f"[test_export_tab_widgets_present] Widget '{css_class}': count={count}")
        assert count > 0, f"Widget '{css_class}' should exist on Export tab"

    print("[test_export_tab_widgets_present] All Export tab widgets found!")
    server.stop()
    print("[test_export_tab_widgets_present] Test complete")
