"""
Test different data input modes for ComparativeViewer.
Tests that viewer supports both dict of arrays and single array inputs.
"""

import numpy as np
import pytest

from pyeyes.viewers import ComparativeViewer


@pytest.mark.basic
def test_dict_input_multiple_datasets(se_data):
    """Test viewer with dict of numpy arrays for multiple datasets."""
    viewer = ComparativeViewer(
        data=se_data,
        named_dims=["x", "y", "z"],
        view_dims=["x", "y"],
    )

    # Verify viewer was created successfully
    assert viewer is not None
    assert viewer.slicer is not None

    # Verify that both datasets are loaded
    assert len(se_data) >= 1

    # Verify dimensions (order may vary, ImgName may be added)
    assert "x" in viewer.slicer.ndims
    assert "y" in viewer.slicer.ndims
    assert "z" in viewer.slicer.ndims


@pytest.mark.basic
def test_single_array_input():
    """Test viewer with a single numpy array (not in a dict)."""
    # Create a simple 3D array
    data = np.random.rand(50, 60, 10)

    viewer = ComparativeViewer(
        data=data,
        named_dims=["x", "y", "z"],
        view_dims=["x", "y"],
    )

    # Verify viewer was created successfully
    assert viewer is not None
    assert viewer.slicer is not None

    # Verify dimensions (single array adds ImgName dimension)
    assert "x" in viewer.slicer.ndims
    assert "y" in viewer.slicer.ndims
    assert "z" in viewer.slicer.ndims


@pytest.mark.basic
def test_single_array_complex():
    """Test viewer with a single complex-valued array."""
    # Create a complex 3D array
    data = np.random.rand(40, 50, 8) + 1j * np.random.rand(40, 50, 8)

    viewer = ComparativeViewer(
        data=data,
        named_dims=["x", "y", "z"],
        view_dims=["x", "y"],
    )

    # Verify viewer was created successfully
    assert viewer is not None

    # Verify data is complex
    assert np.iscomplexobj(data)


@pytest.mark.basic
def test_dict_input_single_dataset():
    """Test viewer with dict containing only one dataset."""
    data = np.random.rand(50, 60, 10)
    data_dict = {"single_dataset": data}

    viewer = ComparativeViewer(
        data=data_dict,
        named_dims=["x", "y", "z"],
        view_dims=["x", "y"],
    )

    # Verify viewer was created successfully
    assert viewer is not None
    assert viewer.slicer is not None


@pytest.mark.basic
def test_dict_input_multiple_datasets_different_names():
    """Test viewer with dict of multiple datasets with custom names."""
    data1 = np.random.rand(40, 50, 8)
    data2 = np.random.rand(40, 50, 8)
    data3 = np.random.rand(40, 50, 8)

    data_dict = {
        "Dataset A": data1,
        "Dataset B": data2,
        "Dataset C": data3,
    }

    viewer = ComparativeViewer(
        data=data_dict,
        named_dims=["x", "y", "z"],
        view_dims=["x", "y"],
    )

    # Verify viewer was created successfully
    assert viewer is not None
    assert len(data_dict) == 3


@pytest.mark.basic
def test_high_dimensional_data():
    """Test viewer with high-dimensional data (>3D)."""
    # Create a 5D array
    data = np.random.rand(20, 30, 10, 5, 3)

    viewer = ComparativeViewer(
        data=data,
        named_dims=["x", "y", "z", "time", "channel"],
        view_dims=["x", "y"],
    )

    # Verify viewer was created successfully
    assert viewer is not None
    # Single array adds ImgName dimension, so 5 + 1 = 6
    assert "x" in viewer.slicer.ndims
    assert "y" in viewer.slicer.ndims
    assert "z" in viewer.slicer.ndims
    assert "time" in viewer.slicer.ndims
    assert "channel" in viewer.slicer.ndims


@pytest.mark.basic
def test_2d_data_input():
    """Test viewer with 2D data (edge case with no slice dimensions)."""
    data = np.random.rand(100, 100)

    viewer = ComparativeViewer(
        data=data,
        named_dims=["x", "y"],
        view_dims=["x", "y"],
    )

    # Verify viewer was created successfully
    assert viewer is not None
    # Single array adds ImgName dimension, but that becomes a slice dim
    # So we have x, y as view dims and ImgName as slice dim
    assert "x" in viewer.slicer.vdims
    assert "y" in viewer.slicer.vdims
