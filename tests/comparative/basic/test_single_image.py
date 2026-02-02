"""
Test case: Single image with config.
Tests viewer with a single dataset and categorical dimensions (MRF data).
"""

import pytest

from pyeyes.viewers import ComparativeViewer


@pytest.mark.basic
def test_single_image_with_config(mrf_data, cfg_path, launched_viewer):
    """Test viewer with a single MRF image and config file."""
    mrf_2min = mrf_data["2min"]

    # Parameters
    config_path = cfg_path / "cfg_mrf_single.yaml"

    # Create viewer with single image in a dict
    viewer = ComparativeViewer(
        data={"2-Minute MRF": mrf_2min},
        named_dims=["Map Type", "x", "y", "z"],
        view_dims=["y", "z"],
        cat_dims={"Map Type": ["PD", "T1", "T2"]},
        config_path=config_path,
    )

    server = launched_viewer(viewer)

    # Verify viewer was created successfully
    assert viewer is not None
    assert viewer.slicer is not None

    # Verify dimensions (order may vary, ImgName added for dicts)
    assert "Map Type" in viewer.slicer.ndims
    assert "x" in viewer.slicer.ndims
    assert "y" in viewer.slicer.ndims
    assert "z" in viewer.slicer.ndims
    # Note: vdims may be overridden by config file
    assert len(viewer.slicer.vdims) == 2

    # Verify categorical dimensions
    assert "Map Type" in viewer.slicer.cat_dims

    server.stop()
