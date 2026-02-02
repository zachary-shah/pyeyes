"""
Test case: Contrast Tab features in ComparativeViewer.
Tests all Contrast tab widget interactions using SE (spin echo) data.
All interactions are done through GUI widgets using Playwright.
"""

import numpy as np
import pytest

# Timeout for GUI tests (in seconds)
GUI_TEST_TIMEOUT = 15
GUI_TEST_TIMEOUT_LONG = 30  # Extended timeout for complex view switching
TASK_TIMEOUT_DEFAULT = 1250  # [ms]


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT_LONG)
def test_contrast_cplx_view_mag_phase_switching(
    viewer_page, se_viewer, navigate_to_tab, isclose
):
    """
    Test switching complex view between 'mag' and 'phase' via GUI.

    Steps:
    1. Save initial vmin/vmax/cmap, verify cplx_view is 'mag'
    2. Switch to 'phase' via GUI, verify values in [-3.15, 3.15] and clim near +/-pi
    3. Change cmap to 'jet' via GUI, verify it changed
    4. Switch back to 'mag' via GUI, verify cmap and clim are restored
    """
    print("\n" + "=" * 60)
    print("[test_contrast_cplx_view] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_contrast_cplx_view] Viewer launched")

    navigate_to_tab(page, "Contrast")

    # Step 1: Save initial state
    print("\n--- Step 1: Save initial state ---")
    initial_cplx_view = viewer.slicer.cplx_view
    initial_vmin = viewer.slicer.vmin
    initial_vmax = viewer.slicer.vmax
    initial_cmap = viewer.slicer.cmap

    print(f"[test_contrast_cplx_view] Initial cplx_view: {initial_cplx_view}")
    print(f"[test_contrast_cplx_view] Initial vmin: {initial_vmin}")
    print(f"[test_contrast_cplx_view] Initial vmax: {initial_vmax}")
    print(f"[test_contrast_cplx_view] Initial cmap: {initial_cmap}")

    assert (
        initial_cplx_view == "mag"
    ), f"Initial cplx_view should be 'mag', got '{initial_cplx_view}'"

    # Step 2: Switch to phase view via GUI select widget
    print("\n--- Step 2: Switch to phase view via GUI ---")
    cplx_view_widget = page.locator(".pyeyes-cplx-view")
    phase_select = cplx_view_widget.get_by_role("button", name="phase", exact=True)
    mag_select = cplx_view_widget.get_by_role("button", name="mag", exact=True)

    if mag_select.count() == 0:
        raise ValueError("Mag select widget not found")
    elif mag_select.count() > 1:
        raise ValueError("Multiple mag select widgets found")

    if phase_select.count() == 0:
        raise ValueError("Phase select widget not found")
    elif phase_select.count() > 1:
        raise ValueError("Multiple phase select widgets found")

    phase_select.click()
    page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
    print("[test_contrast_cplx_view] Selected 'phase' via GUI dropdown")

    new_cplx_view = viewer.slicer.cplx_view
    print(f"[test_contrast_cplx_view] New cplx_view: {new_cplx_view}")
    assert (
        new_cplx_view == "phase"
    ), f"New cplx_view should be 'phase', got '{new_cplx_view}'"

    phase_vmin = viewer.slicer.vmin
    phase_vmax = viewer.slicer.vmax
    print(f"[test_contrast_cplx_view] Phase vmin: {phase_vmin}")
    print(f"[test_contrast_cplx_view] Phase vmax: {phase_vmax}")

    # Get slice data and verify values are in phase range
    slice_data = viewer.slicer.slice()
    img_data = list(slice_data["img"].values())[0].data["Value"]
    data_min = np.min(img_data)
    data_max = np.max(img_data)
    print(
        f"[test_contrast_cplx_view] Phase data range: [{data_min:.4f}, {data_max:.4f}]"
    )

    pi_val = np.pi
    assert isclose(
        data_min, -pi_val, tol=0.1
    ), f"Phase data min should be close to -pi, got {data_min}"
    assert isclose(
        data_max, pi_val, tol=0.1
    ), f"Phase data max should be close to pi, got {data_max}"
    print(
        f"[test_contrast_cplx_view] Phase data is in valid range [-pi, pi]: [{data_min:.4f}, {data_max:.4f}]"
    )

    # Step 3: Change cmap to 'jet' via GUI
    print("\n--- Step 3: Change cmap to 'jet' via GUI ---")
    cmap_widget = page.locator(".pyeyes-cmap")
    cmap_select = cmap_widget.locator("select")

    if cmap_select.count() > 0:
        cmap_select.select_option("jet")
        page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
        print("[test_contrast_cplx_view] Selected 'jet' cmap via GUI dropdown")

    assert (
        viewer.slicer.cmap == "jet"
    ), f"Cmap should be 'jet', got '{viewer.slicer.cmap}'"
    print(f"[test_contrast_cplx_view] Cmap changed to: {viewer.slicer.cmap}")

    # Step 4: Switch back to mag view via GUI
    print("\n--- Step 4: Switch back to mag view via GUI ---")
    mag_select.click()
    page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
    print("[test_contrast_cplx_view] Selected 'mag' via GUI dropdown")

    restored_vmin = viewer.slicer.vmin
    restored_vmax = viewer.slicer.vmax
    restored_cmap = viewer.slicer.cmap

    print(f"[test_contrast_cplx_view] Restored vmin: {restored_vmin}")
    print(f"[test_contrast_cplx_view] Restored vmax: {restored_vmax}")
    print(f"[test_contrast_cplx_view] Restored cmap: {restored_cmap}")

    # Verify cmap is restored to initial value
    assert (
        restored_cmap == initial_cmap
    ), f"Cmap should be restored to '{initial_cmap}', got '{restored_cmap}'"
    print("[test_contrast_cplx_view] Cmap restored correctly")

    # Verify vmin/vmax are restored within 10%
    assert isclose(
        restored_vmin, initial_vmin, tol=0.10
    ), "vmin should be restored within 10%"
    assert isclose(
        restored_vmax, initial_vmax, tol=0.10
    ), "vmax should be restored within 10%"
    print("[test_contrast_cplx_view] vmin/vmax restored within 10%")

    server.stop()
    print("[test_contrast_cplx_view] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_contrast_colorbar_toggle(viewer_page, se_viewer, navigate_to_tab):
    """
    Test toggling the colorbar checkbox on and off via GUI.
    """
    print("\n" + "=" * 60)
    print("[test_contrast_colorbar_toggle] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_contrast_colorbar_toggle] Viewer launched")

    navigate_to_tab(page, "Contrast")

    # Check widget exists
    colorbar_widget = page.locator(".pyeyes-colorbar")
    print(
        f"[test_contrast_colorbar_toggle] Colorbar widget found: {colorbar_widget.count() > 0}"
    )
    assert colorbar_widget.count() > 0, "Colorbar widget should exist"

    # Get initial state
    initial_colorbar_state = viewer.slicer.colorbar_on
    print(
        f"[test_contrast_colorbar_toggle] Initial colorbar_on: {initial_colorbar_state}"
    )

    # Find checkbox and toggle via GUI
    colorbar_checkbox = colorbar_widget.locator("input[type='checkbox']")

    if colorbar_checkbox.count() > 0:
        # Toggle off
        colorbar_checkbox.uncheck()
        page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
        print("[test_contrast_colorbar_toggle] Unchecked colorbar via GUI")

        assert viewer.slicer.colorbar_on is False, "Colorbar should be off"
        print(
            f"[test_contrast_colorbar_toggle] After uncheck: {viewer.slicer.colorbar_on}"
        )

        # Toggle on
        colorbar_checkbox.check()
        page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
        print("[test_contrast_colorbar_toggle] Checked colorbar via GUI")

        assert viewer.slicer.colorbar_on is True, "Colorbar should be on"
        print(
            f"[test_contrast_colorbar_toggle] After check: {viewer.slicer.colorbar_on}"
        )
    else:
        raise ValueError("Colorbar checkbox not found")

    server.stop()
    print("[test_contrast_colorbar_toggle] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_contrast_colorbar_label(viewer_page, se_viewer, navigate_to_tab):
    """
    Test changing the colorbar label via GUI text input.
    """
    print("\n" + "=" * 60)
    print("[test_contrast_colorbar_label] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_contrast_colorbar_label] Viewer launched")

    navigate_to_tab(page, "Contrast")

    # Check widget exists
    colorbar_label_widget = page.locator(".pyeyes-colorbar-label")
    print(
        f"[test_contrast_colorbar_label] Colorbar label widget found: {colorbar_label_widget.count() > 0}"
    )
    assert colorbar_label_widget.count() > 0, "Colorbar label widget should exist"

    # Get initial label
    initial_label = viewer.slicer.colorbar_label
    print(f"[test_contrast_colorbar_label] Initial label: '{initial_label}'")

    # Change label via GUI text input
    label_input = colorbar_label_widget.locator("input[type='text']")

    if label_input.count() > 0:
        label_input.fill("Test Label")
        # Press Enter or Tab to trigger update
        label_input.press("Enter")
        page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
        print("[test_contrast_colorbar_label] Filled label via GUI")
    else:
        raise ValueError("Colorbar label input not found")

    # Verify change
    print(
        f"[test_contrast_colorbar_label] Label after change: '{viewer.slicer.colorbar_label}'"
    )
    assert viewer.slicer.colorbar_label == "Test Label", "Label should be 'Test Label'"

    server.stop()
    print("[test_contrast_colorbar_label] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_contrast_clim_modification(viewer_page, se_viewer, navigate_to_tab, isclose):
    """
    Test modifying the contrast limits (clim) via GUI range slider.
    """
    print("\n" + "=" * 60)
    print("[test_contrast_clim] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_contrast_clim] Viewer launched")

    navigate_to_tab(page, "Contrast")

    # Check widget exists
    clim_widget = page.locator(".pyeyes-clim")
    print(f"[test_contrast_clim] Clim widget found: {clim_widget.count() > 0}")
    assert clim_widget.count() > 0, "Clim widget should exist"

    # Get initial clim
    initial_vmin = viewer.slicer.vmin
    initial_vmax = viewer.slicer.vmax
    print(f"[test_contrast_clim] Initial vmin: {initial_vmin}, vmax: {initial_vmax}")

    # Change vmax first
    clim_inputs = clim_widget.locator(":scope input:visible")
    assert clim_inputs.count() >= 2, "Clim inputs should be at least 2"
    print(f"[test_contrast_clim] Clim inputs: {clim_inputs.count()}")
    clim_inputs.nth(1).fill("100")
    clim_inputs.nth(1).press("Enter")
    page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
    print(f"[test_contrast_clim] New vmax: {viewer.slicer.vmax} ")
    assert isclose(viewer.slicer.vmax, 100, tol=0.01), "vmax should be ~100"

    # change vmin
    clim_inputs.nth(0).fill("0.1")
    clim_inputs.nth(0).press("Enter")
    page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
    print(f"[test_contrast_clim] New vmin: {viewer.slicer.vmin} ")
    assert isclose(viewer.slicer.vmin, 0.1, tol=0.01), "vmin should be ~0.1"

    server.stop()
    print("[test_contrast_clim] Test complete")


@pytest.mark.gui
@pytest.mark.timeout(GUI_TEST_TIMEOUT)
def test_contrast_cmap_change(viewer_page, se_viewer, navigate_to_tab):
    """
    Test changing the colormap via GUI select dropdown.
    """
    print("\n" + "=" * 60)
    print("[test_contrast_cmap] Starting test...")
    print("=" * 60)

    viewer, page, server = viewer_page(se_viewer)
    print("[test_contrast_cmap] Viewer launched")

    navigate_to_tab(page, "Contrast")

    # Check widget exists
    cmap_widget = page.locator(".pyeyes-cmap")
    print(f"[test_contrast_cmap] Cmap widget found: {cmap_widget.count() > 0}")
    assert cmap_widget.count() > 0, "Cmap widget should exist"

    # Get initial cmap
    initial_cmap = viewer.slicer.cmap
    print(f"[test_contrast_cmap] Initial cmap: {initial_cmap}")

    # Find the select element
    cmap_select = cmap_widget.locator("select")

    if cmap_select.count() > 0:
        # Change to different colormaps via GUI
        test_cmaps = ["viridis", "hot", "jet"]
        for cmap in test_cmaps:
            cmap_select.select_option(cmap)
            page.wait_for_timeout(TASK_TIMEOUT_DEFAULT)
            assert viewer.slicer.cmap == cmap, f"Cmap should be '{cmap}'"
            print(f"[test_contrast_cmap] Cmap changed to: {viewer.slicer.cmap} via GUI")

    server.stop()
    print("[test_contrast_cmap] Test complete")
