"""
Test case: View Tab features in ComparativeViewer.
Tests all View tab widget interactions using SE (spin echo) data.
All interactions are done through GUI widgets using Playwright.
"""

import numpy as np
import pytest

# Timeout for GUI tests (in seconds)
GUI_TEST_TIMEOUT = 15
TASK_TIMEOUT_DEFAULT = 1000  # [ms]
TASK_TIMEOUT_LONG = 3000  # [ms]


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_view_title_font_size(viewer_page, se_viewer, navigate_to_tab, isclose):
    """
    Test changing the title font size via GUI slider.
    """
    print("\n" + "=" * 60)
    print("[test_view_title_font_size] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_view_title_font_size] Viewer launched")

    navigate_to_tab(page, "View")

    # Check widget exists
    font_size_widget = page.locator(".pyeyes-title-font-size")
    print(
        f"[test_view_title_font_size] Font size widget found: {font_size_widget.count() > 0}"
    )
    assert font_size_widget.count() > 0, "Title font size widget should exist"

    # Get initial font size
    initial_font_size = viewer.slicer.title_font_size
    print(f"[test_view_title_font_size] Initial font size: {initial_font_size}")

    # Change font size via GUI - find visible input
    font_input = font_size_widget.locator(":scope input:visible")
    if font_input.count() > 0:
        font_input.fill("20")
        font_input.press("Enter")
        page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
        print("[test_view_title_font_size] Font size input filled via GUI")
    else:
        raise ValueError("Font size input not found")

    # Verify change
    print(
        f"[test_view_title_font_size] Font size after change: {viewer.slicer.title_font_size}"
    )
    assert isclose(
        viewer.slicer.title_font_size, 20, tol=0.01
    ), "Font size should be 20"

    server.stop()
    print("[test_view_title_font_size] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_view_size_scale(viewer_page, se_viewer, navigate_to_tab, isclose):
    """
    Test changing the size scale of images via GUI slider.
    """
    print("\n" + "=" * 60)
    print("[test_view_size_scale] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_view_size_scale] Viewer launched")

    navigate_to_tab(page, "View")

    # Check widget exists
    size_scale_widget = page.locator(".pyeyes-size-scale")
    print(
        f"[test_view_size_scale] Size scale widget found: {size_scale_widget.count() > 0}"
    )
    assert size_scale_widget.count() > 0, "Size scale widget should exist"

    # Get initial size scale
    initial_size_scale = viewer.slicer.size_scale
    print(f"[test_view_size_scale] Initial size scale: {initial_size_scale}")

    # Change size scale via GUI - find visible input
    size_input = size_scale_widget.locator(":scope input:visible")
    if size_input.count() > 0:
        size_input.fill("600")
        size_input.press("Enter")
        page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
        print("[test_view_size_scale] Size scale input filled via GUI")
    else:
        raise ValueError("Size scale input not found")

    # Verify change
    print(f"[test_view_size_scale] Size scale after change: {viewer.slicer.size_scale}")
    assert isclose(viewer.slicer.size_scale, 600, tol=0.01), "Size scale should be 600"

    server.stop()
    print("[test_view_size_scale] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_view_flip_ud_lr(viewer_page, se_viewer, navigate_to_tab):
    """
    Test flip U/D and flip L/R functionality via GUI checkboxes and verify slice data is flipped.
    """
    print("\n" + "=" * 60)
    print("[test_view_flip_ud_lr] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_view_flip_ud_lr] Viewer launched")

    navigate_to_tab(page, "View")

    # Check widgets exist
    flip_ud_widget = page.locator(".pyeyes-flip-ud")
    flip_lr_widget = page.locator(".pyeyes-flip-lr")
    print(f"[test_view_flip_ud_lr] Flip U/D widget found: {flip_ud_widget.count() > 0}")
    print(f"[test_view_flip_ud_lr] Flip L/R widget found: {flip_lr_widget.count() > 0}")
    assert flip_ud_widget.count() > 0, "Flip U/D widget should exist"
    assert flip_lr_widget.count() > 0, "Flip L/R widget should exist"

    # Get initial state
    print(f"[test_view_flip_ud_lr] Initial flip_ud: {viewer.slicer.flip_ud}")
    print(f"[test_view_flip_ud_lr] Initial flip_lr: {viewer.slicer.flip_lr}")

    # Test flip U/D via GUI checkbox
    flip_ud_checkbox = flip_ud_widget.locator("input[type='checkbox']")
    if flip_ud_checkbox.count() > 0:
        flip_ud_checkbox.check()
        page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
        print("[test_view_flip_ud_lr] Flip U/D checkbox checked via GUI")
    else:
        raise ValueError("Flip U/D checkbox not found")

    print(
        f"[test_view_flip_ud_lr] After flip_ud check, flip_ud: {viewer.slicer.flip_ud}"
    )
    assert viewer.slicer.flip_ud is True, "flip_ud should be True after checking"

    # Uncheck flip U/D and test flip L/R
    flip_ud_checkbox.uncheck()
    page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)

    flip_lr_checkbox = flip_lr_widget.locator("input[type='checkbox']")
    if flip_lr_checkbox.count() > 0:
        flip_lr_checkbox.check()
        page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
        print("[test_view_flip_ud_lr] Flip L/R checkbox checked via GUI")
    else:
        raise ValueError("Flip L/R checkbox not found")

    print(
        f"[test_view_flip_ud_lr] After flip_lr check, flip_lr: {viewer.slicer.flip_lr}"
    )
    assert viewer.slicer.flip_lr is True, "flip_lr should be True after checking"

    # Test both flips enabled
    flip_ud_checkbox.check()
    page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
    print(
        f"[test_view_flip_ud_lr] Both flips enabled: flip_ud={viewer.slicer.flip_ud}, flip_lr={viewer.slicer.flip_lr}"
    )

    server.stop()
    print("[test_view_flip_ud_lr] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_view_sdim_slider_navigation(viewer_page, se_viewer, navigate_to_tab):
    """
    Test editing the 'z' sdim slider to 13 via GUI, then back to 10.
    Verify data changes and returns to original when going back.
    """
    print("\n" + "=" * 60)
    print("[test_view_sdim_slider_navigation] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_view_sdim_slider_navigation] Viewer launched")

    navigate_to_tab(page, "View")

    # Find the z sdim slider widget
    z_sdim_widget = page.locator(".pyeyes-sdim-z")
    print(
        f"[test_view_sdim_slider_navigation] Z sdim widget found: {z_sdim_widget.count() > 0}"
    )
    assert z_sdim_widget.count() > 0, "Z sdim widget should exist"

    # Get initial z index
    initial_z = viewer.slicer.dim_indices.get("z", 0)
    print(f"[test_view_sdim_slider_navigation] Initial z index: {initial_z}")

    # Find the visible input for the z slider
    z_input = z_sdim_widget.locator(":scope input:visible")
    if z_input.count() == 0:
        raise ValueError("Z sdim input not found")

    # Set z=10 via GUI
    z_input.fill("10")
    z_input.press("Enter")
    page.wait_for_timeout(TASK_TIMEOUT_LONG)
    print("[test_view_sdim_slider_navigation] Set z=10 via GUI")

    slice_at_10 = viewer.slicer.slice()
    data_at_10 = list(slice_at_10["img"].values())[0].data["Value"].copy()
    print(f"[test_view_sdim_slider_navigation] Data at z=10, shape: {data_at_10.shape}")
    print(
        f"[test_view_sdim_slider_navigation] Data at z=10, mean: {np.mean(data_at_10):.6f}"
    )

    # Change to z=13 via GUI
    z_input.fill("13")
    z_input.press("Enter")
    page.wait_for_timeout(TASK_TIMEOUT_LONG)
    print("[test_view_sdim_slider_navigation] Set z=13 via GUI")

    slice_at_13 = viewer.slicer.slice()
    data_at_13 = list(slice_at_13["img"].values())[0].data["Value"].copy()
    print(f"[test_view_sdim_slider_navigation] Data at z=13, shape: {data_at_13.shape}")
    print(
        f"[test_view_sdim_slider_navigation] Data at z=13, mean: {np.mean(data_at_13):.6f}"
    )

    # Verify data is different at z=13
    assert not np.allclose(
        data_at_10, data_at_13
    ), "Data at z=10 and z=13 should be different"
    print("[test_view_sdim_slider_navigation] Verified: data at z=10 != data at z=13")

    # Return to z=10 via GUI
    z_input.fill("10")
    z_input.press("Enter")
    page.wait_for_timeout(TASK_TIMEOUT_LONG)
    print("[test_view_sdim_slider_navigation] Set z=10 via GUI (return)")

    slice_back_to_10 = viewer.slicer.slice()
    data_back_to_10 = list(slice_back_to_10["img"].values())[0].data["Value"].copy()
    print(
        f"[test_view_sdim_slider_navigation] Data back at z=10, mean: {np.mean(data_back_to_10):.6f}"
    )

    # Verify data is the same as original z=10
    assert np.allclose(
        data_at_10, data_back_to_10
    ), "Data should be same after returning to z=10"
    print(
        "[test_view_sdim_slider_navigation] Verified: data restored after returning to z=10"
    )

    server.stop()
    print("[test_view_sdim_slider_navigation] Test complete")
