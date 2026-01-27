"""
Test case: Complex-valued image data.
Tests that viewer can handle two complex spin echo images of the same size.
"""

import time

import pytest

from pyeyes.viewers import ComparativeViewer

NRMSE_EXPECTED = 0.149
PSNR_EXPECTED = 26.7075
SSIM_EXPECTED = 0.84158


@pytest.mark.basic
def test_complex_images(se_data, cfg_path, launched_viewer, isclose, cplx_slc_data):
    """Test viewer with complex-valued spin echo images, with confirmed config loading."""

    # Parameters
    named_dims = ["x", "y", "z"]
    vdims = ["x", "y"]

    # Create viewer with config
    viewer = ComparativeViewer(
        data=se_data,
        named_dims=named_dims,
        view_dims=vdims,
        config_path=cfg_path / "cfg_cplx.yaml",
    )

    # Launch viewer silently
    server = launched_viewer(viewer)

    # Test 1: Ensure viewer features match config
    assert len(viewer.slicer.sdims) == 1, "Viewer should have 1 slice dim"
    assert "z" in viewer.slicer.sdims, "z dimension not found in viewer"
    assert viewer.slicer.title_font_size == 16, "Title font size should be 16"
    assert isclose(
        viewer.slicer.vmin, 0.003281
    ), f"vmin expected to be 5.7417e-8, but got {viewer.slicer.vmin}"
    assert isclose(
        viewer.slicer.vmax, 383.658
    ), f"vmax expected to be 383.658, but got {viewer.slicer.vmax}"
    assert viewer.slicer.size_scale == 310, "Size scale should be 310"
    assert viewer.slicer.flip_lr and (
        not viewer.slicer.flip_ud
    ), "Flip LR should be True and Flip UD should be False"
    assert viewer.slicer.cplx_view == "mag", "Complex view should be mag"
    assert viewer.slicer.display_images == [
        "4avg",
        "1avg",
    ], "Display images should be 4avg and 1avg"
    assert viewer.slicer.cmap == "gray", "Colormap should be gray"
    assert viewer.slicer.colorbar_on, "Colorbar should be on"
    assert viewer.slicer.colorbar_label == "Rad", "Colorbar label should be Rad"
    assert viewer.slicer.dim_indices["z"] == 10, "z dimension index should be 10"
    assert viewer.slicer.metrics_reference == "4avg", "Metrics reference should be 4avg"
    assert viewer.slicer.error_map_type == "L1Diff", "Error map type should be L1Diff"
    assert viewer.slicer.normalize_error_map, "Normalize error map should be True"
    assert (
        viewer.slicer.metrics_text_font_size == 7
    ), "Metrics text font size should be 7"

    # Test 2: Ensure metrics are correct
    data = cplx_slc_data(viewer)
    assert isclose(
        data.nrmse, NRMSE_EXPECTED
    ), f"NRMSE expected to be {NRMSE_EXPECTED}, but got {data.nrmse}"
    assert isclose(
        data.psnr, PSNR_EXPECTED
    ), f"PSNR expected to be {PSNR_EXPECTED}, but got {data.psnr}"
    assert isclose(
        data.ssim, SSIM_EXPECTED
    ), f"SSIM expected to be {SSIM_EXPECTED}, but got {data.ssim}"

    server.stop()


@pytest.mark.basic
def test_complex_images_misscaled_autoscale(
    se_data, cfg_path, launched_viewer, isclose, cplx_slc_data
):
    """Test viewer with complex-valued spin echo images, config loading, and metrics, with small scale images."""
    # Scale images to very small values to test scale invariance
    small_img_dict = {k: v * 1e-10 for k, v in se_data.items()}

    # Parameters
    named_dims = ["x", "y", "z"]
    vdims = ["x", "y"]

    # Create viewer with config
    viewer = ComparativeViewer(
        data=small_img_dict,
        named_dims=named_dims,
        view_dims=vdims,
        config_path=cfg_path / "cfg_cplx.yaml",
    )

    # Launch viewer silently
    server = launched_viewer(viewer)

    # Test 2: Ensure metrics are correct
    data = cplx_slc_data(viewer)
    assert isclose(
        data.nrmse, NRMSE_EXPECTED
    ), f"NRMSE expected to be {NRMSE_EXPECTED}, but got {data.nrmse}"
    assert isclose(
        data.psnr, PSNR_EXPECTED
    ), f"PSNR expected to be {PSNR_EXPECTED}, but got {data.psnr}"
    assert isclose(
        data.ssim, SSIM_EXPECTED
    ), f"SSIM expected to be {SSIM_EXPECTED}, but got {data.ssim}"

    # Test 3: Run autoscale clim. TODO: replace with button click.
    viewer._autoscale_clim(None)
    time.sleep(0.5)

    # Test 4: Metrics should not change after autoscaling (scale invariance)
    data = cplx_slc_data(viewer)
    assert isclose(
        data.nrmse, NRMSE_EXPECTED
    ), f"NRMSE expected to be {NRMSE_EXPECTED} after autoscale, but got {data.nrmse}"
    assert isclose(
        data.psnr, PSNR_EXPECTED
    ), f"PSNR expected to be {PSNR_EXPECTED} after autoscale, but got {data.psnr}"
    assert isclose(
        data.ssim, SSIM_EXPECTED
    ), f"SSIM expected to be {SSIM_EXPECTED} after autoscale, but got {data.ssim}"

    # Test 5: But vmin/vmax should change after autoscaling
    assert isclose(
        viewer.slicer.vmin, 3.28189e-13
    ), f"vmin expected to be 3.28189e-13, but got {viewer.slicer.vmin}"
    assert isclose(
        viewer.slicer.vmax, 3.83658e-8
    ), f"vmax expected to be 3.83658e-8, but got {viewer.slicer.vmax}"

    server.stop()
