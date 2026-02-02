"""
Test case: Diffusion-weighted imaging (DWI) data.
Tests viewer with three large DWI datasets (skope, festive, uncorr).
"""

import warnings

import pytest

from pyeyes import set_theme
from pyeyes.viewers import ComparativeViewer


@pytest.mark.load
def test_dwi_three_datasets(dwi_data, cfg_path, launched_viewer):
    """Test viewer with three DWI reconstruction datasets."""
    # Set theme
    set_theme("dark")

    # Parameters
    named_dims = ["Bdir", "x", "y", "z"]
    vdims = ["x", "y"]

    config_path = cfg_path / "cfg_diff.yaml"

    viewer = ComparativeViewer(
        data=dwi_data, named_dims=named_dims, view_dims=vdims, config_path=config_path
    )
    server = launched_viewer(viewer)
    # Verify viewer was created successfully
    assert viewer is not None
    assert viewer.slicer is not None

    # Verify dimensions (order may vary, ImgName may be added)
    for dim in named_dims:
        assert dim in viewer.slicer.ndims
    for dim in vdims:
        assert dim in viewer.slicer.vdims

    # Verify all three datasets are loaded
    assert len(dwi_data) == 3
    assert "skope" in dwi_data
    assert "festive" in dwi_data
    assert "uncorr" in dwi_data

    server.stop()


@pytest.mark.load
def test_dwi_subset(dwi_data, cfg_path, launched_viewer):
    """Test viewer with a subset of DWI datasets (no config)."""
    # Test with just two datasets
    subset_data = {
        "skope": dwi_data["skope"],
        "festive": dwi_data["festive"],
    }

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        viewer = ComparativeViewer(
            data=subset_data,
            named_dims=["Bdir", "x", "y", "z"],
            view_dims=["x", "y"],
            config_path=cfg_path / "cfg_diff.yaml",
        )
        allowed_message = "Supplied images do not match config - Loading viewer with default image selection."
        for warning in wlist:
            if (
                issubclass(warning.category, RuntimeWarning)
                and str(warning.message) == allowed_message
            ):
                break

    server = launched_viewer(viewer)
    assert viewer is not None
    assert len(subset_data) == 2
    server.stop()
