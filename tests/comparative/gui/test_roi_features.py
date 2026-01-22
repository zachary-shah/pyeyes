"""
Test case: ROI Tab features in ComparativeViewer.
Tests all ROI tab widget interactions using DWI (diffusion-weighted imaging) data.
All interactions are done through GUI widgets using Playwright.
"""

import pytest

from pyeyes.enums import ROI_LOCATION, ROI_STATE

# Timeout for GUI tests (in seconds)
GUI_TEST_TIMEOUT = 15
TASK_TIMEOUT_DEFAULT = 1000  # [ms]
TASK_TIMEOUT_LONG = 3000  # [ms]


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_roi_overlay_enabled_checkbox(viewer_page, dwi_viewer, navigate_to_tab):
    """
    Test toggling the 'ROI Overlay Mode' checkbox via GUI.
    Verify no bug occurs.
    """
    print("\n" + "=" * 60)
    print("[test_roi_overlay_enabled] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(dwi_viewer)
    print("[test_roi_overlay_enabled] Viewer launched")

    navigate_to_tab(page, "ROI")

    # Check widget exists
    roi_mode_widget = page.locator(".pyeyes-roi-mode")
    print(
        f"[test_roi_overlay_enabled] ROI mode widget found: {roi_mode_widget.count() > 0}"
    )
    assert roi_mode_widget.count() > 0, "ROI mode widget should exist"

    # Get initial ROI mode
    initial_mode = viewer.slicer.roi_mode
    print(f"[test_roi_overlay_enabled] Initial ROI mode: {initial_mode}")

    # Find the checkbox and toggle via GUI
    roi_mode_checkbox = roi_mode_widget.locator("input[type='checkbox']")

    if roi_mode_checkbox.count() > 0:
        # Uncheck if checked, or check if unchecked
        if roi_mode_checkbox.is_checked():
            roi_mode_checkbox.uncheck()
            page.wait_for_timeout(TASK_TIMEOUT_LONG)
            print("[test_roi_overlay_enabled] Unchecked ROI overlay via GUI")
        else:
            roi_mode_checkbox.check()
            page.wait_for_timeout(TASK_TIMEOUT_LONG)
            print("[test_roi_overlay_enabled] Checked ROI overlay via GUI")
    else:
        raise ValueError("ROI mode checkbox not found")

    print(f"[test_roi_overlay_enabled] ROI mode after toggle: {viewer.slicer.roi_mode}")
    print("[test_roi_overlay_enabled] ROI overlay toggle completed without error")

    # Toggle back
    if roi_mode_checkbox.is_checked():
        roi_mode_checkbox.uncheck()
    else:
        roi_mode_checkbox.check()
    page.wait_for_timeout(TASK_TIMEOUT_LONG)
    print("[test_roi_overlay_enabled] ROI overlay toggled back")

    server.stop()
    print("[test_roi_overlay_enabled] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_roi_clear_button(viewer_page, dwi_viewer, navigate_to_tab):
    """
    Test clicking the 'Clear ROI' button via GUI.
    Verify no bug occurs.
    """
    print("\n" + "=" * 60)
    print("[test_roi_clear_button] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(dwi_viewer)
    print("[test_roi_clear_button] Viewer launched")

    navigate_to_tab(page, "ROI")

    # Check widget exists
    clear_roi_widget = page.locator(".pyeyes-clear-roi")
    print(
        f"[test_roi_clear_button] Clear ROI widget found: {clear_roi_widget.count() > 0}"
    )
    assert clear_roi_widget.count() > 0, "Clear ROI widget should exist"

    # Get initial ROI state
    initial_state = viewer.slicer.roi_state
    print(f"[test_roi_clear_button] Initial ROI state: {initial_state}")

    # Click clear ROI button via GUI
    clear_btn = clear_roi_widget.locator("button")
    if clear_btn.count() > 0:
        clear_btn.click()
        page.wait_for_timeout(TASK_TIMEOUT_LONG)
        print("[test_roi_clear_button] Clear ROI button clicked via GUI")
    else:
        clear_roi_widget.click()
        page.wait_for_timeout(TASK_TIMEOUT_LONG)
        print("[test_roi_clear_button] Clear ROI widget clicked via GUI")

    # Verify state after clear (should be inactive)
    print(f"[test_roi_clear_button] ROI state after clear: {viewer.slicer.roi_state}")
    assert (
        viewer.slicer.roi_state == ROI_STATE.INACTIVE
    ), "ROI state should be INACTIVE after clear"
    print("[test_roi_clear_button] Clear ROI completed without error")

    server.stop()
    print("[test_roi_clear_button] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_roi_cmap_change_to_jet(viewer_page, dwi_viewer, navigate_to_tab):
    """
    Test changing ROI colormap to 'jet' via GUI select.
    Verify no error.
    """
    print("\n" + "=" * 60)
    print("[test_roi_cmap_change] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(dwi_viewer)
    print("[test_roi_cmap_change] Viewer launched")

    navigate_to_tab(page, "ROI")

    # Check widget exists
    roi_cmap_widget = page.locator(".pyeyes-roi-cmap")
    print(
        f"[test_roi_cmap_change] ROI cmap widget found: {roi_cmap_widget.count() > 0}"
    )
    assert roi_cmap_widget.count() > 0, "ROI cmap widget should exist"

    # Get initial ROI cmap
    initial_cmap = viewer.slicer.roi_cmap
    print(f"[test_roi_cmap_change] Initial ROI cmap: {initial_cmap}")

    # Change to 'jet' via GUI select
    roi_cmap_select = roi_cmap_widget.locator("select")

    if roi_cmap_select.count() > 0:
        roi_cmap_select.select_option("jet")
        page.wait_for_timeout(TASK_TIMEOUT_LONG)
        print("[test_roi_cmap_change] Selected 'jet' via GUI dropdown")
    else:
        raise ValueError("ROI cmap select not found")

    # Verify change
    assert viewer.slicer.roi_cmap == "jet", "ROI cmap should be 'jet'"
    print(f"[test_roi_cmap_change] ROI cmap changed to: {viewer.slicer.roi_cmap}")

    server.stop()
    print("[test_roi_cmap_change] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_roi_overlay_and_location_change(viewer_page, dwi_viewer, navigate_to_tab):
    """
    Test enabling ROI overlay via GUI and changing location to 'Top Left' via GUI.
    """
    print("\n" + "=" * 60)
    print("[test_roi_overlay_location] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(dwi_viewer)
    print("[test_roi_overlay_location] Viewer launched")

    navigate_to_tab(page, "ROI")

    # Check widgets exist
    roi_mode_widget = page.locator(".pyeyes-roi-mode")
    roi_loc_widget = page.locator(".pyeyes-roi-loc")
    print(
        f"[test_roi_overlay_location] ROI mode widget found: {roi_mode_widget.count() > 0}"
    )
    print(
        f"[test_roi_overlay_location] ROI location widget found: {roi_loc_widget.count() > 0}"
    )
    assert roi_mode_widget.count() > 0, "ROI mode widget should exist"
    assert roi_loc_widget.count() > 0, "ROI location widget should exist"

    # Enable overlay mode via GUI checkbox
    roi_mode_checkbox = roi_mode_widget.locator("input[type='checkbox']")
    if roi_mode_checkbox.count() > 0:
        if not roi_mode_checkbox.is_checked():
            roi_mode_checkbox.check()
            page.wait_for_timeout(TASK_TIMEOUT_LONG)
            print("[test_roi_overlay_location] Enabled ROI overlay via GUI")

    print(f"[test_roi_overlay_location] ROI mode: {viewer.slicer.roi_mode}")

    # Change location to Top Left via GUI select
    roi_loc_select = roi_loc_widget.locator("select")
    target_loc = ROI_LOCATION.TOP_LEFT.value

    if roi_loc_select.count() > 0:
        roi_loc_select.select_option(target_loc)
        page.wait_for_timeout(TASK_TIMEOUT_LONG)
        print(f"[test_roi_overlay_location] Selected '{target_loc}' location via GUI")
    else:
        raise ValueError("ROI location select not found")

    # Verify location changed
    print(f"[test_roi_overlay_location] ROI location: {viewer.slicer.ROI.roi_loc}")
    assert (
        viewer.slicer.ROI.roi_loc == ROI_LOCATION.TOP_LEFT
    ), "ROI location should be TOP_LEFT"

    server.stop()
    print("[test_roi_overlay_location] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_roi_zoom_order_change(viewer_page, dwi_viewer, navigate_to_tab):
    """
    Test changing the ROI zoom order to 2 via GUI input.
    """
    print("\n" + "=" * 60)
    print("[test_roi_zoom_order] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(dwi_viewer)
    print("[test_roi_zoom_order] Viewer launched")

    navigate_to_tab(page, "ROI")

    # Check widget exists
    zoom_order_widget = page.locator(".pyeyes-roi-zoom-order")
    print(
        f"[test_roi_zoom_order] ROI zoom order widget found: {zoom_order_widget.count() > 0}"
    )
    assert zoom_order_widget.count() > 0, "ROI zoom order widget should exist"

    # Get initial zoom order
    initial_order = viewer.slicer.ROI.zoom_order
    print(f"[test_roi_zoom_order] Initial zoom order: {initial_order}")

    # Change zoom order via GUI visible input (IntInput widget)
    zoom_order_input = zoom_order_widget.locator(":scope input:visible")
    if zoom_order_input.count() > 0:
        zoom_order_input.fill("2")
        zoom_order_input.press("Enter")
        page.wait_for_timeout(TASK_TIMEOUT_LONG)
        print("[test_roi_zoom_order] Set zoom order to 2 via GUI")
    else:
        raise ValueError("Zoom order input not found")

    # Verify change
    print(
        f"[test_roi_zoom_order] Zoom order after change: {viewer.slicer.ROI.zoom_order}"
    )
    assert viewer.slicer.ROI.zoom_order == 2, "Zoom order should be 2"

    server.stop()
    print("[test_roi_zoom_order] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_roi_all_widgets_exist(viewer_page, dwi_viewer, navigate_to_tab):
    """
    Test that all ROI tab widgets exist and are accessible.
    """
    print("\n" + "=" * 60)
    print("[test_roi_all_widgets] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(dwi_viewer)
    print("[test_roi_all_widgets] Viewer launched")

    navigate_to_tab(page, "ROI")

    # List of expected widgets on ROI tab
    roi_widgets = [
        ("pyeyes-roi-mode", "ROI mode"),
        ("pyeyes-draw-roi", "Draw ROI button"),
        ("pyeyes-clear-roi", "Clear ROI button"),
        ("pyeyes-roi-cmap", "ROI colormap"),
        ("pyeyes-zoom-scale", "Zoom scale"),
        ("pyeyes-roi-loc", "ROI location"),
        ("pyeyes-roi-line-color", "Line color"),
        ("pyeyes-roi-line-width", "Line width"),
        ("pyeyes-roi-zoom-order", "Zoom order"),
    ]

    for css_class, widget_name in roi_widgets:
        widget = page.locator(f".{css_class}")
        count = widget.count()
        print(f"[test_roi_all_widgets] {widget_name} (.{css_class}): count={count}")
        assert count > 0, f"{widget_name} widget should exist"

    print("[test_roi_all_widgets] All ROI widgets verified")

    server.stop()
    print("[test_roi_all_widgets] Test complete")
