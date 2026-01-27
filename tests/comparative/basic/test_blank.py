"""
Test case: Flat data with zeros by default.
Tests that viewer can handle data where one slice is all zeros.
"""

import numpy as np
import pytest

from pyeyes.viewers import ComparativeViewer


@pytest.mark.basic
def test_flat_input(launched_viewer):
    """Test viewer with data that has a blank (all zeros) slice by default."""
    data = np.zeros((100, 100, 3))
    data[:, :, 0] = 1
    data[:, :, 2] = 3

    viewer = ComparativeViewer(data=data, named_dims=["x", "y", "z"])
    server = launched_viewer(viewer)

    # Verify it exists
    assert viewer is not None
    assert viewer.slicer is not None
    assert "x" in viewer.slicer.ndims
    assert "y" in viewer.slicer.ndims
    assert "z" in viewer.slicer.ndims

    server.stop()


@pytest.mark.basic
def test_zero_input(launched_viewer):
    """Test one input being fully all zeros."""

    data = np.zeros((100, 100, 3))
    data[:, :, 0] = 1
    data[:, :, 2] = 3
    data2 = np.zeros((100, 100, 3))
    data_dict = {
        "flats": data,
        "zero": data2,
    }

    # Create viewer
    viewer = ComparativeViewer(data=data_dict, named_dims=["x", "y", "z"])

    # Launch viewer and get Playwright page
    server = launched_viewer(viewer)

    server.stop()
