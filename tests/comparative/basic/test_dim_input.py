"""
Test various ways of providing dimensional names to viewer.
Tests different formats for named_dims and view_dims parameters.
"""

import pytest

from pyeyes.viewers import ComparativeViewer


@pytest.mark.basic
def test_no_named_dims_input(se_data):
    """Test viewer creation with no named_dims input (should use defaults)."""
    viewer = ComparativeViewer(data=se_data)
    assert viewer is not None


@pytest.mark.basic
def test_list_of_strings(se_data):
    """Test viewer with named_dims as a list of strings."""
    viewer = ComparativeViewer(data=se_data, named_dims=["x", "y", "z"])
    assert viewer is not None
    # Check that all expected dims are present (order may vary, ImgName added for dicts)
    assert "x" in viewer.slicer.ndims
    assert "y" in viewer.slicer.ndims
    assert "z" in viewer.slicer.ndims


@pytest.mark.basic
def test_character_string(se_data):
    """Test viewer with named_dims as a character string."""
    viewer = ComparativeViewer(data=se_data, named_dims="xyz")
    assert viewer is not None
    assert "x" in viewer.slicer.ndims
    assert "y" in viewer.slicer.ndims
    assert "z" in viewer.slicer.ndims


@pytest.mark.basic
def test_character_string_with_spaces(se_data):
    """Test viewer with named_dims as a space-separated string."""
    viewer = ComparativeViewer(data=se_data, named_dims="x y z")
    assert viewer is not None
    assert "x" in viewer.slicer.ndims
    assert "y" in viewer.slicer.ndims
    assert "z" in viewer.slicer.ndims


@pytest.mark.basic
def test_character_string_with_commas(se_data):
    """Test viewer with named_dims as a comma-separated string."""
    viewer = ComparativeViewer(data=se_data, named_dims="x,y,z")
    assert viewer is not None
    assert "x" in viewer.slicer.ndims
    assert "y" in viewer.slicer.ndims
    assert "z" in viewer.slicer.ndims


@pytest.mark.basic
def test_with_view_dim_input(se_data):
    """Test viewer with view_dims as a list."""
    viewer = ComparativeViewer(
        data=se_data, named_dims=["x", "y", "z"], view_dims=["y", "z"]
    )
    assert viewer is not None
    assert viewer.slicer.vdims == ["y", "z"]


@pytest.mark.basic
def test_view_dim_input_with_spaces(se_data):
    """Test viewer with view_dims as a space-separated string."""
    viewer = ComparativeViewer(
        data=se_data, named_dims=["x", "y", "z"], view_dims="y z"
    )
    assert viewer is not None
    assert viewer.slicer.vdims == ["y", "z"]


@pytest.mark.basic
def test_view_dim_input_with_commas(se_data):
    """Test viewer with view_dims as a comma-separated string."""
    viewer = ComparativeViewer(
        data=se_data, named_dims=["x", "y", "z"], view_dims="y,z"
    )
    assert viewer is not None
    assert viewer.slicer.vdims == ["y", "z"]


@pytest.mark.basic
def test_longer_dimension_names(se_data):
    """Test viewer with longer dimension names."""
    viewer = ComparativeViewer(
        data=se_data, named_dims=["Ex", "Why", "Zee"], view_dims=["Why", "Zee"]
    )
    assert viewer is not None
    assert "Ex" in viewer.slicer.ndims
    assert "Why" in viewer.slicer.ndims
    assert "Zee" in viewer.slicer.ndims
    assert "Why" in viewer.slicer.vdims
    assert "Zee" in viewer.slicer.vdims


@pytest.mark.basic
def test_no_spaces_single_char(se_data):
    """Test viewer with single character string for both named_dims and view_dims."""
    viewer = ComparativeViewer(data=se_data, named_dims="xyz", view_dims="xz")
    assert viewer is not None
    assert "x" in viewer.slicer.ndims
    assert "y" in viewer.slicer.ndims
    assert "z" in viewer.slicer.ndims
    assert "x" in viewer.slicer.vdims
    assert "z" in viewer.slicer.vdims


@pytest.mark.basic
def test_longer_names_with_spaces(se_data):
    """Test viewer with longer names using mixed separators."""
    viewer = ComparativeViewer(
        data=se_data, named_dims="Ex, Why, Zee", view_dims="Why Zee"
    )
    assert viewer is not None
    assert "Ex" in viewer.slicer.ndims
    assert "Why" in viewer.slicer.ndims
    assert "Zee" in viewer.slicer.ndims
    assert "Why" in viewer.slicer.vdims
    assert "Zee" in viewer.slicer.vdims
