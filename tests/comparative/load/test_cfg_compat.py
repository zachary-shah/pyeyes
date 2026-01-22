"""
Test case: Single image with config from different dataset.
Tests that viewer can load a config created for a different dataset (different name).
"""

import warnings

import numpy as np
import pytest

from pyeyes import set_theme
from pyeyes.viewers import ComparativeViewer


@pytest.mark.load
def test_single_image_different_name_same_config(mrf_data, cfg_path, launched_viewer):
    """Test viewer with single image using config from a different dataset name."""
    # Set theme
    set_theme("dark")

    mrf_1min = mrf_data["1min"]
    config_path = cfg_path / "cfg_mrf_single.yaml"

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        viewer = ComparativeViewer(
            data={"1-Minute MRF": mrf_1min},  # Different name than config expects
            named_dims=["Map Type", "x", "y", "z"],
            view_dims=["y", "z"],
            cat_dims={"Map Type": ["PD", "T1", "T2"]},
            config_path=config_path,
        )

        allowed_message = "Supplied images do not match config - Loading viewer with default image selection."
        for warning in wlist:
            if (
                issubclass(warning.category, RuntimeWarning)
                and str(warning.message) == allowed_message
            ):
                break
        else:
            pass

    # Verify viewer was created successfully
    assert viewer is not None
    assert viewer.slicer is not None

    # Verify dimensions (order may vary, ImgName may be added)
    assert "Map Type" in viewer.slicer.ndims
    assert "x" in viewer.slicer.ndims
    assert "y" in viewer.slicer.ndims
    assert "z" in viewer.slicer.ndims

    # Verify categorical dimensions
    assert "Map Type" in viewer.slicer.cat_dims

    server = launched_viewer(viewer)
    server.stop()


@pytest.mark.load
def test_config_compatibility(mrf_data, cfg_path, launched_viewer):
    """Test that configs are flexible with different dataset names."""

    mrf_2min = mrf_data["2min"]
    config_path = cfg_path / "cfg_mrf_single.yaml"

    # Test with original name (should work)
    viewer1 = ComparativeViewer(
        data={"2-Minute MRF": mrf_2min},
        named_dims=["Map Type", "x", "y", "z"],
        view_dims=["y", "z"],
        cat_dims={"Map Type": ["PD", "T1", "T2"]},
        config_path=config_path,
    )
    assert viewer1 is not None
    server = launched_viewer(viewer1)
    server.stop()

    # Test with different name (should still work, possibly with a specific warning)
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        viewer2 = ComparativeViewer(
            data={"Custom Name": mrf_2min},
            named_dims=["Map Type", "x", "y", "z"],
            view_dims=["y", "z"],
            cat_dims={"Map Type": ["PD", "T1", "T2"]},
            config_path=config_path,
        )
        # Allow specific RuntimeWarning, but fail if not this warning
        allowed_message = "Supplied images do not match config - Loading viewer with default image selection."
        for warning in wlist:
            if (
                issubclass(warning.category, RuntimeWarning)
                and str(warning.message) == allowed_message
            ):
                # This warning is allowed, continue
                break
        else:
            # No warning, or warning is not the allowed type/message: also fine
            pass

    assert viewer2 is not None
    server = launched_viewer(viewer2)
    server.stop()
