"""
Test case: Multiple viewers launched simultaneously.
Tests launching two different types of datasets in separate viewers at the same time.
"""

import pytest

from pyeyes import ComparativeViewer, launch_viewers


@pytest.mark.load
def test_multiple_viewers_creation(af_data, se_data, cfg_path):
    """Test creating multiple viewers with different datasets."""

    # Create two separate viewers
    viewer_se = ComparativeViewer(
        data=se_data,
        named_dims=["x", "y", "z"],
        view_dims=["x", "y"],
        config_path=cfg_path / "cfg_cplx.yaml",
    )

    viewer_af = ComparativeViewer(
        data=af_data,
        named_dims=["x", "y", "z"],
        view_dims=["x", "y"],
        config_path=cfg_path / "cfg_af.yaml",
    )

    # Verify both viewers were created successfully
    assert viewer_se is not None
    assert viewer_af is not None
    assert len(se_data) == 2
    assert len(af_data) == 4

    server = launch_viewers(
        {"SpinEcho": viewer_se, "Autofocus": viewer_af},
        title="Spin Echo vs Autofocus",
        show=False,
        start=True,
        threaded=True,
        verbose=True,
        port=9999,
    )
    server.stop()
