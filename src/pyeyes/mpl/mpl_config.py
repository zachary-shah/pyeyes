import os
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np


# Plotting dataclass
@dataclass
class PlotConfig:
    # TODO: reselect best defaults
    save: bool = False
    figdir: str = None  # must set this up

    # Sizings
    figsize: tuple = field(default_factory=lambda: (4, 4))
    dpi: int = 120
    font_size: int = 18

    # Aesthetics
    remove_axes: bool = False
    plot_trim: list = field(
        default_factory=lambda: [5, 0]
    )  # trim x, y by this many pixels on each side
    fov_shift: list = field(
        default_factory=lambda: [0, 0]
    )  # cyclic shift x, y by this many pixels

    # Color Config
    background_color: str = "black"
    secondary_color: str = "white"

    # Phase Estimation and Error Display
    N_ests_plot: int = 8
    N_time_default_plot: int = 10
    cmap: str = "jet"
    error_scale: float = 0.1
    phase_scale: float = 1.0
    phase_error_scale: float = 0.5
    mask_thresh: float = 0.1

    # Gif Config
    N_gif: int = 500  # max number of timepoints to save in gif
    gif_duration: int = 5  # gif duration in seconds

    # Spherical Harmonic Plots
    N_sh: int = 500  # number of timepoints to estimate sh fit. None = do all

    def setup_figdir(self, root):
        self.figdir = os.path.join(root, "figures")
        os.makedirs(self.figdir, exist_ok=True)


# For Dark Mode
def get_line_colors(background_color="black"):
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    high_contrast_colors = [
        "#FF6347",  # Tomato (Bright Red)
        "#00FF00",  # Lime Green
        "#1E90FF",  # Dodger Blue
        "#FFFF00",  # Yellow
        "#FF69B4",  # Hot Pink
        "#00CED1",  # Dark Turquoise
        "#FF4500",  # Orange Red
        "#32CD32",  # Lime Green (darker)
        "#FFD700",  # Gold
        "#7FFF00",  # Chartreuse
    ]

    if background_color == "black":
        return high_contrast_colors
    else:
        return default_colors


def dark_mode(
    fig,
    ax,
    cbars: Optional[Sequence[plt.colorbar]] = None,
    background_color="black",
    secondary_color="white",
):

    fig.patch.set_facecolor(background_color)
    if fig._suptitle is not None:
        fig.suptitle(fig._suptitle.get_text(), color=secondary_color)
    if isinstance(ax, np.ndarray):
        for a in ax.ravel():
            a.set_facecolor(background_color)
            plt.setp(a.spines.values(), color=secondary_color)
            a.tick_params(axis="both", colors=secondary_color)
            a.xaxis.label.set_color(secondary_color)
            a.yaxis.label.set_color(secondary_color)
            a.title.set_color(secondary_color)
    else:
        ax.set_facecolor(background_color)
        plt.setp(ax.spines.values(), color=secondary_color)
        ax.tick_params(axis="both", colors=secondary_color)
        ax.xaxis.label.set_color(secondary_color)
        ax.yaxis.label.set_color(secondary_color)
        ax.title.set_color(secondary_color)

    if cbars is not None:
        for cbar in cbars:
            cbar.ax.yaxis.set_tick_params(color=secondary_color)
            plt.setp(cbar.ax.get_yticklabels(), color=secondary_color)

    return fig, ax
