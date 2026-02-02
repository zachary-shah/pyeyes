"""
Test case: Analysis Tab features in ComparativeViewer.
Tests all Analysis tab widget interactions using SE (spin echo) data.
All interactions are done through GUI widgets using Playwright.
"""

import numpy as np
import pytest

from pyeyes.enums import ROI_LOCATION
from pyeyes.viewers import ComparativeViewer

# Timeout for GUI tests (in seconds)
GUI_TEST_TIMEOUT = 15
TASK_TIMEOUT_DEFAULT = 1000  # [ms]


def get_ssim_metric(viewer, ref_key):
    """Helper to get SSIM metric from viewer."""
    slice_data = viewer.slicer.slice()
    metrics = slice_data.get("metrics", {})
    for img_key, img_metrics in metrics.items():
        if img_key != ref_key and "SSIM" in img_metrics:
            return img_metrics["SSIM"]
    return None


def get_psnr_metric(viewer, ref_key):
    """Helper to get PSNR metric from viewer."""
    slice_data = viewer.slicer.slice()
    metrics = slice_data.get("metrics", {})
    for img_key, img_metrics in metrics.items():
        if img_key != ref_key and "PSNR" in img_metrics:
            return img_metrics["PSNR"]
    return None


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_analysis_change_reference_dataset(viewer_page, se_viewer, navigate_to_tab):
    """
    Test changing the reference dataset from '4avg' to '1avg' via GUI.
    Verify that SSIM metric changes by more than 0.01%.
    """
    print("\n" + "=" * 60)
    print("[test_analysis_change_reference] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_analysis_change_reference] Viewer launched")

    navigate_to_tab(page, "Analysis")

    # Check widget exists
    ref_widget = page.locator(".pyeyes-reference-dataset")
    print(
        f"[test_analysis_change_reference] Reference widget found: {ref_widget.count() > 0}"
    )
    assert ref_widget.count() > 0, "Reference dataset widget should exist"

    # Get initial reference
    initial_ref = viewer.slicer.metrics_reference
    print(f"[test_analysis_change_reference] Initial reference: {initial_ref}")

    # Find the select element for reference dataset
    ref_select = ref_widget.locator("select")

    if ref_select.count() > 0:
        # Set reference to '4avg' via GUI if not already
        if initial_ref != "4avg":
            ref_select.select_option("4avg")
            page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
            print("[test_analysis_change_reference] Selected '4avg' via GUI")

    ssim_with_4avg = get_ssim_metric(viewer, "4avg")
    print(
        f"[test_analysis_change_reference] SSIM with 4avg as reference: {ssim_with_4avg}"
    )

    # Change reference to '1avg' via GUI
    if ref_select.count() > 0:
        ref_select.select_option("1avg")
        page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
        print("[test_analysis_change_reference] Selected '1avg' via GUI")

    ssim_with_1avg = get_ssim_metric(viewer, "1avg")
    print(
        f"[test_analysis_change_reference] SSIM with 1avg as reference: {ssim_with_1avg}"
    )

    # Verify reference changed
    assert viewer.slicer.metrics_reference == "1avg", "Reference should be '1avg'"
    print(
        f"[test_analysis_change_reference] Reference changed to: {viewer.slicer.metrics_reference}"
    )

    # Verify SSIM changed by more than 0.01%
    if ssim_with_4avg is not None and ssim_with_1avg is not None:
        relative_change = abs(ssim_with_4avg - ssim_with_1avg) / max(
            abs(ssim_with_4avg), 1e-10
        )
        print(
            f"[test_analysis_change_reference] Relative SSIM change: {relative_change * 100:.4f}%"
        )
        assert (
            relative_change > 0.0001
        ), "SSIM should change by more than 0.01% when reference changes"
    else:
        print("[test_analysis_change_reference] Warning: Could not compare SSIM values")
        raise ValueError(
            f"Could not compare SSIM values: {ssim_with_4avg} and {ssim_with_1avg}"
        )

    server.stop()
    print("[test_analysis_change_reference] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_analysis_normalize_error_map(viewer_page, se_viewer, navigate_to_tab):
    """
    Test toggling 'normalize error map' checkbox via GUI.
    Verify that the toggle works without error.
    """
    print("\n" + "=" * 60)
    print("[test_analysis_normalize_error] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_analysis_normalize_error] Viewer launched")

    navigate_to_tab(page, "Analysis")

    # Check widget exists
    normalize_widget = page.locator(".pyeyes-error-normalize")
    print(
        f"[test_analysis_normalize_error] Normalize widget found: {normalize_widget.count() > 0}"
    )
    assert normalize_widget.count() > 0, "Normalize error map widget should exist"

    # Get initial normalize state
    initial_normalize = viewer.slicer.normalize_error_map
    sref = viewer.slicer.metrics_reference
    initial_SSIM = get_ssim_metric(viewer, sref)
    print(
        f"[test_analysis_normalize_error] Initial normalize state: {initial_normalize}"
    )
    print(f"[test_analysis_normalize_error] Initial SSIM: {initial_SSIM}")

    # Find and toggle the checkbox via GUI
    normalize_checkbox = normalize_widget.locator("input[type='checkbox']")

    if normalize_checkbox.count() > 0:
        if initial_normalize:
            normalize_checkbox.uncheck()
        else:
            normalize_checkbox.check()
        page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
        print("[test_analysis_normalize_error] Toggled normalize checkbox via GUI")

    # Verify toggle worked
    print(
        f"[test_analysis_normalize_error] Normalize state after toggle: {viewer.slicer.normalize_error_map}"
    )
    new_SSIM = get_ssim_metric(viewer, sref)
    print(f"[test_analysis_normalize_error] New SSIM: {new_SSIM}")
    assert (
        viewer.slicer.normalize_error_map != initial_normalize
    ), "Normalize state should have toggled"

    if initial_SSIM is not None and new_SSIM is not None:
        relative_change = abs(initial_SSIM - new_SSIM) / max(abs(initial_SSIM), 1e-10)
        print(
            f"[test_analysis_normalize_error] Relative SSIM change: {relative_change * 100:.4f}%"
        )
        assert (
            relative_change > 0.0001
        ), "SSIM should change by more than 0.01% when normalize state toggled"
    else:
        print("[test_analysis_normalize_error] Warning: Could not compare SSIM values")
        raise ValueError(
            f"Could not compare SSIM values: {initial_SSIM} and {new_SSIM}"
        )

    server.stop()
    print("[test_analysis_normalize_error] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_analysis_error_cmap_change(viewer_page, se_viewer, navigate_to_tab):
    """
    Test changing the error map colormap to 'hot' via GUI select.
    """
    print("\n" + "=" * 60)
    print("[test_analysis_error_cmap] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_analysis_error_cmap] Viewer launched")

    navigate_to_tab(page, "Analysis")

    # Check widget exists
    error_cmap_widget = page.locator(".pyeyes-error-cmap")
    print(
        f"[test_analysis_error_cmap] Error cmap widget found: {error_cmap_widget.count() > 0}"
    )
    assert error_cmap_widget.count() > 0, "Error cmap widget should exist"

    # Get initial error cmap
    initial_error_cmap = viewer.slicer.error_map_cmap
    print(f"[test_analysis_error_cmap] Initial error cmap: {initial_error_cmap}")

    # Change to 'hot' via GUI select
    error_cmap_select = error_cmap_widget.locator("select")

    if error_cmap_select.count() > 0:
        error_cmap_select.select_option("hot")
        page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
        print("[test_analysis_error_cmap] Selected 'hot' via GUI dropdown")

    # Verify change
    assert viewer.slicer.error_map_cmap == "hot", "Error cmap should be 'hot'"
    print(
        f"[test_analysis_error_cmap] Error cmap changed to: {viewer.slicer.error_map_cmap}"
    )

    server.stop()
    print("[test_analysis_error_cmap] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_analysis_metrics_location(viewer_page, se_viewer, navigate_to_tab):
    """
    Test moving metrics text to bottom left via GUI select.
    """
    print("\n" + "=" * 60)
    print("[test_analysis_metrics_location] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_analysis_metrics_location] Viewer launched")

    navigate_to_tab(page, "Analysis")

    # Check widget exists
    metrics_loc_widget = page.locator(".pyeyes-metrics-text-font-loc")
    print(
        f"[test_analysis_metrics_location] Metrics location widget found: {metrics_loc_widget.count() > 0}"
    )
    assert metrics_loc_widget.count() > 0, "Metrics location widget should exist"

    # Get initial location
    initial_loc = viewer.slicer.metrics_text_location.value
    print(f"[test_analysis_metrics_location] Initial location: {initial_loc}")

    # Change to bottom left via GUI select
    loc_select = metrics_loc_widget.locator("select")

    blstr = ROI_LOCATION.BOTTOM_LEFT.value
    if loc_select.count() > 0:
        loc_select.select_option(blstr)
        page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
        print(f"[test_analysis_metrics_location] Selected '{blstr}' via GUI dropdown")

    # Verify change
    newstr = viewer.slicer.metrics_text_location.value
    assert newstr == blstr, f"Location should be '{blstr}'"
    print(f"[test_analysis_metrics_location] Location changed to: {newstr}")

    server.stop()
    print("[test_analysis_metrics_location] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_analysis_autoformat_error_map(viewer_page, se_viewer, navigate_to_tab):
    """
    Test pressing the autoformat error map button via GUI.
    """
    print("\n" + "=" * 60)
    print("[test_analysis_autoformat] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_analysis_autoformat] Viewer launched")

    navigate_to_tab(page, "Analysis")

    # Check widget exists
    autoformat_widget = page.locator(".pyeyes-error-map-autoformat")
    print(
        f"[test_analysis_autoformat] Autoformat widget found: {autoformat_widget.count() > 0}"
    )
    assert autoformat_widget.count() > 0, "Autoformat widget should exist"

    # Get initial error map settings
    initial_scale = viewer.slicer.error_map_scale
    initial_cmap = viewer.slicer.error_map_cmap
    print(f"[test_analysis_autoformat] Initial error scale: {initial_scale}")
    print(f"[test_analysis_autoformat] Initial error cmap: {initial_cmap}")

    # Click autoformat button via GUI
    autoformat_btn = autoformat_widget.locator("button")
    if autoformat_btn.count() > 0:
        autoformat_btn.click()
        page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
        print("[test_analysis_autoformat] Clicked autoformat button via GUI")
    else:
        # Try clicking the widget itself
        autoformat_widget.click()
        page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
        print("[test_analysis_autoformat] Clicked autoformat widget via GUI")

    # Get updated settings
    new_scale = viewer.slicer.error_map_scale
    new_cmap = viewer.slicer.error_map_cmap
    print(f"[test_analysis_autoformat] After autoformat error scale: {new_scale}")
    print(f"[test_analysis_autoformat] After autoformat error cmap: {new_cmap}")

    # Just verify no errors occurred
    print("[test_analysis_autoformat] Autoformat completed without error")

    server.stop()
    print("[test_analysis_autoformat] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_analysis_error_map_type_change(viewer_page, se_viewer, navigate_to_tab):
    """
    Test changing the error map type via GUI select.
    """
    print("\n" + "=" * 60)
    print("[test_analysis_error_map_type] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_analysis_error_map_type] Viewer launched")

    navigate_to_tab(page, "Analysis")

    # Check widget exists
    error_type_widget = page.locator(".pyeyes-error-map-type")
    print(
        f"[test_analysis_error_map_type] Error type widget found: {error_type_widget.count() > 0}"
    )
    assert error_type_widget.count() > 0, "Error map type widget should exist"

    # Get initial error map type
    initial_type = viewer.slicer.error_map_type
    print(f"[test_analysis_error_map_type] Initial error map type: {initial_type}")

    # Change type via GUI select
    error_type_select = error_type_widget.locator("select")
    new_type = "SSIM" if initial_type != "SSIM" else "L1Diff"

    if error_type_select.count() > 0:
        error_type_select.select_option(new_type)
        page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
        print(f"[test_analysis_error_map_type] Selected '{new_type}' via GUI dropdown")

    # Verify change
    assert (
        viewer.slicer.error_map_type == new_type
    ), f"Error map type should be '{new_type}'"
    print(
        f"[test_analysis_error_map_type] Error map type changed to: {viewer.slicer.error_map_type}"
    )

    server.stop()
    print("[test_analysis_error_map_type] Test complete")
