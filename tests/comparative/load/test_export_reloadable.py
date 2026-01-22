"""
Test case: Export reloadable viewer functionality.
Tests that viewer can export to a reloadable format.
"""

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from pyeyes.viewers import ComparativeViewer


@pytest.mark.load
def test_export_reloadable_sliced(mrf_data, launched_viewer):
    """Test exporting viewer to a reloadable Python script."""
    # Make a Dictionary of volumes to compare
    img_dict = {"1 Minute MRF": mrf_data["1min"], "2 Minute MRF": mrf_data["2min"]}

    # Parameters
    named_dims = ["Map Type", "x", "y", "z"]

    # Allow categorical dimensions to be specified
    cat_dims = {"Map Type": ["PD", "T1", "T2"]}

    viewer = ComparativeViewer(
        data=img_dict,
        named_dims=named_dims,
        view_dims=list("xy"),
        cat_dims=cat_dims,
    )
    server = launched_viewer(viewer)

    with TemporaryDirectory() as tmp_dir:
        export_path = Path(tmp_dir)

        viewer.export_reloadable_pyeyes(
            path=export_path / "test_export_reloadable.py",
            num_slices_to_keep={"z": 20},
            silent=True,
        )

        # Verify export files were created
        assert (export_path / "test_export_reloadable.py").exists()

        # The export should also create accompanying data files
        # Check that the .py file is not empty
        py_file = export_path / "test_export_reloadable.py"
        assert py_file.stat().st_size > 0, f"File {py_file} is empty"

        # Run exported script
        output = subprocess.run(f"python {py_file}", shell=True, check=True)
        assert output.returncode == 0, f"Script failed with status {output.returncode}"

    server.stop()


@pytest.mark.load
def test_export_reloadable_full(mrf_data, launched_viewer):
    """Test exporting viewer with full data (no slice reduction)."""
    # Make a Dictionary of volumes to compare
    img_dict = {"1 Minute MRF": mrf_data["1min"], "2 Minute MRF": mrf_data["2min"]}

    # Parameters
    named_dims = ["Map Type", "x", "y", "z"]

    # Allow categorical dimensions to be specified
    cat_dims = {"Map Type": ["PD", "T1", "T2"]}

    viewer = ComparativeViewer(
        data=img_dict,
        named_dims=named_dims,
        view_dims=list("xy"),
        cat_dims=cat_dims,
    )
    server = launched_viewer(viewer)

    with TemporaryDirectory() as tmp_dir:
        export_path = Path(tmp_dir)

        viewer.export_reloadable_pyeyes(
            path=export_path / "test_export_reloadable.py",
            silent=True,
        )

        # Verify export files were created
        assert (export_path / "test_export_reloadable.py").exists()

        # The export should also create accompanying data files
        # Check that the .py file is not empty
        py_file = export_path / "test_export_reloadable.py"
        assert py_file.stat().st_size > 0, f"File {py_file} is empty"

        # Run exported script
        output = subprocess.run(f"python {py_file}", shell=True, check=True)
        assert output.returncode == 0, f"Script failed with status {output.returncode}"

    server.stop()
