"""
Test various ways of providing dimensional names to viewer.
"""

import numpy as np
from paths import data_path

from pyeyes.viewers import ComparativeViewer

img_dict = {
    "4avg": np.load(data_path / "se" / "avg_se.npy"),
    "1avg": np.load(data_path / "se" / "single_se.npy"),
}
errors = 0

try:
    print("Testing no input...")
    Viewer = ComparativeViewer(data=img_dict)
except Exception as e:
    print(f"\tError: {e}")
    errors += 1

try:
    print("Testing list of strings...")
    Viewer = ComparativeViewer(data=img_dict, named_dims=["x", "y", "z"])
except Exception as e:
    print(f"\tError: {e}")
    errors += 1

try:
    print("Testing character string...")
    Viewer = ComparativeViewer(data=img_dict, named_dims="xyz")
except Exception as e:
    print(f"\tError: {e}")
    errors += 1

try:
    print("Testing character string with spaces...")
    Viewer = ComparativeViewer(data=img_dict, named_dims="x y z")
except Exception as e:
    print(f"\tError: {e}")
    errors += 1

try:
    print("Testing character string with commas...")
    Viewer = ComparativeViewer(data=img_dict, named_dims="x,y,z")
except Exception as e:
    print(f"\tError: {e}")
    errors += 1

try:
    print("Testing with view dim input...")
    Viewer = ComparativeViewer(
        data=img_dict, named_dims=["x", "y", "z"], view_dims=["y", "z"]
    )
except Exception as e:
    print(f"\tError: {e}")
    errors += 1

try:
    print("Testing with view dim input with spaces...")
    Viewer = ComparativeViewer(
        data=img_dict, named_dims=["x", "y", "z"], view_dims="y z"
    )
except Exception as e:
    print(f"\tError: {e}")
    errors += 1

try:
    print("Testing with view dim input with commas...")
    Viewer = ComparativeViewer(
        data=img_dict, named_dims=["x", "y", "z"], view_dims="y,z"
    )
except Exception as e:
    print(f"\tError: {e}")
    errors += 1

try:
    print("Testing with longer names")
    Viewer = ComparativeViewer(
        data=img_dict, named_dims=["Ex", "Why", "Zee"], view_dims=["Why", "Zee"]
    )
except Exception as e:
    print(f"\tError: {e}")
    errors += 1

try:
    print("Testing with no spaces")
    Viewer = ComparativeViewer(data=img_dict, named_dims="xyz", view_dims="xz")
except Exception as e:
    print(f"\tError: {e}")
    errors += 1
try:
    print("Testing with longer names with spaces...")
    Viewer = ComparativeViewer(
        data=img_dict, named_dims="Ex, Why, Zee", view_dims="Why Zee"
    )
except Exception as e:
    print(f"\tError: {e}")
    errors += 1

if errors > 0:
    print(f"Failed {errors} tests.")
else:
    print("All dim input tests passed!")
