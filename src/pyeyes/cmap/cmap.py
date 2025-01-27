import os
from typing import Union

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import cm

VALID_COLORMAPS = [
    "gray",
    "jet",
    "viridis",
    "inferno",
    "RdBu",
    "Magma",
    "Quantitative",
]

# No need for the 'Quantitative' colormap for error maps
VALID_ERROR_COLORMAPS = [
    "gray",
    "jet",
    "viridis",
    "inferno",
    "RdBu",
    "Magma",
    "Grays",  # this is good for SSIM
]

QUANTITATIVE_MAPTYPES = [
    "T1",
    "T2",
    "R1",
    "R2",
    "T1rho",
    "T1ρ",
    "R1rho",
    "R1ρ",
    "T2*",
    "R2*",
]


class ColorMap:
    """
    Abstract class for colormap generation.
    """

    def __init__(self, cmap: Union[mcolors.ListedColormap, str]):
        self.cmap = cmap

    def preprocess_data(self, x: np.ndarray) -> np.ndarray:
        """
        Pre-process data for display, if needed (e.g. clipping)

        Parameters
        ----------
        x : np.ndarray
            Data to be processed

        Returns
        -------
        np.ndarray
            Processed data of same shape
        """
        return x

    def get_cmap(self) -> Union[mcolors.ListedColormap, str]:
        """
        Get the colormap for display.

        Returns
        -------
        Union[matplotlib.colors.ListedColormap, str]
            Colormap for display. Can be a string or ListedColormap.
        """
        return self.cmap


class QuantitativeColorMap(ColorMap):

    def __init__(
        self,
        maptype: str,
        loLev: float,
        upLev: float,
    ):
        """
        General colormap for quantitative data.
        """

        self.maptype = maptype
        self.loLev = loLev
        self.upLev = upLev

        # Generate colormap and epsilon for data processing
        self.lut_cmap, self.eps = relaxation_color_map(maptype, loLev, upLev)

        # Process colormap into a matplotlib colormap list
        self.cmap = mcolors.ListedColormap(self.lut_cmap)

        super().__init__(self.cmap)

    def preprocess_data(self, x):
        return np.where(
            x < self.eps,
            self.loLev - self.eps,
            np.where(x < self.loLev + self.eps, self.loLev + 1.5 * self.eps, x),
        )


def relaxation_color_map(maptype, loLev, upLev):
    """
    Acts in two ways:
    1. Generates a colormap to be used on display, given the image type.
    2. Generates standard deviation for 'clipping' image, where values are clipped to the lower level.
    """

    # Load the colormap depending on the map type
    maptype = maptype.capitalize()

    assert (
        maptype in QUANTITATIVE_MAPTYPES
    ), f"Got {maptype} maptype, expected one of {QUANTITATIVE_MAPTYPES}"

    current_dir = os.path.dirname(os.path.abspath(__file__))

    if maptype in ["T1", "R1"]:
        file_path = os.path.join(current_dir, "lipari.csv")
        colortable = np.genfromtxt(file_path, delimiter=" ")
    elif maptype in ["T2", "T2*", "R2", "R2*", "T1rho", "T1ρ", "R1rho", "R1ρ"]:
        file_path = os.path.join(current_dir, "navia.csv")
        colortable = np.genfromtxt(file_path, delimiter=" ")
    else:
        raise ValueError("Expect 'T1', 'T2', 'R1', or 'R2' as maptype")

    # Flip colortable for R1 or R2 types
    if maptype[0] == "R":
        colortable = np.flipud(colortable)

    # Set the 'invalid value' color
    colortable[0, :] = 0.0

    # Epsilon for modification of the image to be displayed
    eps = (upLev - loLev) / colortable.shape[0]

    # Apply color remapping
    lut_cmap = color_log_remap(colortable, loLev, upLev)

    return lut_cmap, eps


def color_log_remap(ori_cmap, loLev, upLev):
    """
    Lookup of the original color map table according to a 'log-like' curve.
    """
    assert upLev > 0, "Upper level must be positive"
    assert upLev > loLev, "Upper level must be larger than lower level"

    map_length = ori_cmap.shape[0]
    e_inv = np.exp(-1.0)
    a_val = e_inv * upLev
    m_val = max(a_val, loLev)
    b_val = (1.0 / map_length) + (a_val >= loLev) * (
        (a_val - loLev) / (2 * a_val - loLev)
    )
    b_val += 1e-7  # Ensure rounding precision

    log_cmap = np.zeros_like(ori_cmap)
    log_cmap[0, :] = ori_cmap[0, :]

    log_portion = 1.0 / (np.log(m_val) - np.log(upLev))

    for g in range(1, map_length):
        x = g * (upLev - loLev) / map_length + loLev
        if x > m_val:
            f = map_length * (
                (np.log(m_val) - np.log(x)) * log_portion * (1 - b_val) + b_val
            )
        elif loLev < a_val and x > loLev:
            f = (
                map_length
                * ((x - loLev) / (a_val - loLev) * (b_val - (1.0 / map_length)))
                + 1.0
            )
        else:
            f = 1.0 if x <= loLev else f

        log_cmap[g, :] = ori_cmap[min(map_length - 1, int(np.floor(f))), :]

    return log_cmap


# Unused: will use in the future
def get_jet_black_cmp(ap=1.2):
    jet = cm.get_cmap("jet", 256)
    jet_colors = jet(np.linspace(0, 1, 256))
    jet_colors[0] = np.array([0.1, 0.1, 1, 1])
    jet_colors[255] = np.array([1, 0.1, 0.1, 1])
    blue_to_black_idx = 0
    black_idx = 128
    black_to_red_idx = 255
    blue = jet_colors[blue_to_black_idx]
    red = jet_colors[black_to_red_idx]
    ap = 1.2
    black = np.array([0, 0, 0, 1])
    for i in range(blue_to_black_idx, black_idx):
        alpha = ((i - blue_to_black_idx) / (black_idx - blue_to_black_idx)) ** ap
        jet_colors[i, :] = alpha * black + (1 - alpha) * blue
    for i in range(black_idx, black_to_red_idx):
        alpha = (1 - (i - black_idx) / (black_to_red_idx - black_idx)) ** ap
        jet_colors[i, :] = alpha * black + (1 - alpha) * red
    custom_jet_black = LinearSegmentedColormap.from_list("custom_jet_black", jet_colors)
    return custom_jet_black


# Get list of colors for plotting
JET_ERROR_CMAP = get_jet_black_cmp()
