"""
Test case: MRF data with multiple images and config.
Tests viewer with two MRF datasets with categorical dimensions and config loading.
"""

import numpy as np
import pytest

from pyeyes import set_theme
from pyeyes.viewers import ComparativeViewer


@pytest.mark.load
def test_mrf_single(mrf_data, launched_viewer):
    """Test viewer with MRF data using x,y as view dimensions."""
    # Make a Dictionary of volumes to compare
    img_dict = {"1 Minute MRF": mrf_data["1min"], "2 Minute MRF": mrf_data["2min"]}

    # Parameters
    named_dims = ["Map Type", "x", "y", "z"]
    vdims = ["x", "y"]

    # Allow categorical dimensions to be specified
    cat_dims = {"Map Type": ["PD", "T1", "T2"]}

    viewer = ComparativeViewer(
        data=img_dict,
        named_dims=named_dims,
        view_dims=vdims,
        cat_dims=cat_dims,
    )

    assert viewer is not None

    server = launched_viewer(viewer)
    server.stop()


@pytest.mark.load
def test_mrf_four_views(mrf_data, cfg_path, launched_viewer):
    """Test viewer with four MRF datasets with varying noise levels."""
    # Set theme
    set_theme("dark")

    # Use existing 1min and 2min data
    mrf_1min = mrf_data["1min"]
    mrf_2min = mrf_data["2min"]

    # Extract individual maps for manipulation
    llr_1min_pd = mrf_1min[0]
    llr_1min_t1 = mrf_1min[1]
    llr_1min_t2 = mrf_1min[2]

    # Simulate 45sec and 30sec with shot noise
    llr_45sec_pd = llr_1min_pd + np.random.normal(
        0, np.std(llr_1min_pd) * 0.1, llr_1min_pd.shape
    )
    llr_45sec_t1 = llr_1min_t1 + np.random.normal(
        0, np.std(llr_1min_t1) * 0.1, llr_1min_t1.shape
    )
    llr_45sec_t2 = llr_1min_t2 + np.random.normal(
        0, np.std(llr_1min_t2) * 0.1, llr_1min_t2.shape
    )

    llr_30sec_pd = llr_1min_pd + np.random.normal(
        0, np.std(llr_1min_pd) * 0.25, llr_1min_pd.shape
    )
    llr_30sec_t1 = llr_1min_t1 + np.random.normal(
        0, np.std(llr_1min_t1) * 0.25, llr_1min_t1.shape
    )
    llr_30sec_t2 = llr_1min_t2 + np.random.normal(
        0, np.std(llr_1min_t2) * 0.25, llr_1min_t2.shape
    )

    mrf_30sec = np.stack([llr_30sec_pd, llr_30sec_t1, llr_30sec_t2], axis=0)
    mrf_45sec = np.stack([llr_45sec_pd, llr_45sec_t1, llr_45sec_t2], axis=0)

    # mask
    mrf_30sec = mrf_30sec * (np.abs(mrf_2min) > 0)
    mrf_45sec = mrf_45sec * (np.abs(mrf_2min) > 0)

    img_dict = {
        "30sec": mrf_30sec,
        "45sec": mrf_45sec,
        "1min": mrf_1min,
        "2min": mrf_2min,
    }

    # Parameters
    named_dims = ["Map Type", "x", "y", "z"]
    vdims = ["y", "z"]

    # Allow categorical dimensions to be specified
    cat_dims = {"Map Type": ["PD", "T1", "T2"]}

    viewer = ComparativeViewer(
        data=img_dict,
        named_dims=named_dims,
        view_dims=vdims,
        cat_dims=cat_dims,
        config_path=cfg_path / "cfg_mrf_4view.yaml",
    )

    # Verify viewer was created successfully
    assert viewer is not None
    assert viewer.slicer is not None

    # Verify all four datasets are loaded
    assert len(img_dict) == 4

    # Verify dimensions (order may vary, ImgName may be added)
    for dim in named_dims:
        assert dim in viewer.slicer.ndims
    for dim in vdims:
        assert dim in viewer.slicer.vdims

    # Verify categorical dimensions
    assert "Map Type" in viewer.slicer.cat_dims

    server = launched_viewer(viewer)
    server.stop()
