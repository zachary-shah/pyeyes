"""
Plotting tools for DTI data.

TODO: Port to Bokeh (currently mpl backend)
"""

import os
import warnings

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import use as backend_use
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..utils import tonp
from .mpl_config import PlotConfig, dark_mode


def load_dti(dti_path):
    dti_keys = ["FA", "MD", "MO", "S0", "L1", "L2", "L3", "V1", "V2", "V3"]

    dti_imgs = {}
    for key in dti_keys:
        try:
            dti_imgs[key] = nib.load(f"{dti_path}_{key}.nii.gz").get_fdata()
        except:
            print(f"Could not load {key}. Loading as zeros")

    return dti_imgs


def plot_dti(
    dti_imgs_or_path,
    suptitle: str = None,
    transpose: bool = True,
    cfg: PlotConfig = PlotConfig(),
):
    """
    Standard plotting of DTI results.

    Parameters
    ----------
    dti_imgs_or_path : Union[dict, str]
        Either a str path, where path contains f"{path}_{X}.nii.gz", or a set of images in a dict with keys X, where:
         - X included for all ['FA', 'MD', 'S0', 'MO', 'L1', 'L2', 'L3', 'V1', 'V2', 'V3']
         - all images have shape (X, Y, Z), with (X, Y, Z, 3) for V1-V3
    suptitle: str
        Title of the plot
    transpose: bool
        Whether to transpose the images
    cfg: PlotConfig

    Returns
    -------
    PlotObj: Dynamic plotting figure
    """

    dti_keys = ["FA", "MD", "MO", "S0", "L1", "L2", "L3", "V1", "V2", "V3"]
    dti_keys_color = ["V1", "V2", "V3"]

    if isinstance(dti_imgs_or_path, str):
        dti_imgs = load_dti(dti_imgs_or_path)
    else:
        dti_imgs = dti_imgs_or_path

    # Get image shape from FA map
    assert "FA" in dti_imgs, "FA image must be provided"
    im_shape = dti_imgs["FA"].shape
    if len(im_shape) == 2:
        im_shape = (*im_shape, 1)

    # allow missing inputs
    for key in dti_keys:
        if key not in dti_imgs:
            if key in dti_keys_color:
                dti_imgs[key] = np.zeros((*im_shape, 3))
            else:
                dti_imgs[key] = np.zeros(im_shape)

    # preprocess all to include slice dim, if not already
    for key in dti_keys:
        if key in dti_keys_color:
            if len(dti_imgs[key].shape) == 3:
                dti_imgs[key] = dti_imgs[key][:, :, None, :]
        elif len(dti_imgs[key].shape) == 2:
            dti_imgs[key] = dti_imgs[key][:, :, None]

    Nz = dti_imgs["FA"].shape[2]

    # Now everything is either (X, Y, Z) or (X, Y, Z, 3). Extract and pre-process
    FA = tonp(dti_imgs["FA"])
    MD = tonp(dti_imgs["MD"])
    MO = tonp(dti_imgs["MO"])
    S0 = tonp(dti_imgs["S0"])
    L1 = tonp(dti_imgs["L1"])
    L2 = tonp(dti_imgs["L2"])
    L3 = tonp(dti_imgs["L3"])
    V1 = tonp(dti_imgs["V1"])
    V2 = tonp(dti_imgs["V2"])
    V3 = tonp(dti_imgs["V3"])

    # FA Should be [0, 1]
    FA = np.clip(np.abs(FA), 0, 1)

    # S0 abs
    S0 = np.abs(S0)

    # Expect diffusivity to be between 0 and 0.002 or so
    MD = np.abs(MD)
    L1 = np.abs(L1)
    L2 = np.abs(L2)
    L3 = np.abs(L3)

    # MO typically in range [-1, 1]
    MO = np.clip(MO, -1, 1)
    MO[np.abs(MO) < 1e-8] = np.nan

    # Modulate the eigenvectors by FA
    V1 = np.clip(np.abs(V1) * FA[..., None], 0, 1)
    V2 = np.clip(np.abs(V2) * FA[..., None], 0, 1)
    V3 = np.clip(np.abs(V3) * FA[..., None], 0, 1)

    if suptitle is None:
        suptitle = "DTI Fit"

    PlotObj = PlotDTIDynamic(
        FA,
        MD,
        MO,
        S0,
        L1,
        L2,
        L3,
        V1,
        V2,
        V3,
        suptitle=suptitle,
        transpose=transpose,
        cfg=cfg,
    )

    if cfg.save:
        orig_z = PlotObj.z
        for z in range(Nz):
            PlotObj.z = z
            PlotObj.update_axes()
            PlotObj.update_image()
            PlotObj.fig.canvas.draw()
            PlotObj.fig.savefig(os.path.join(cfg.figdir, f"dti_slc{z}.png"))
        PlotObj.z = orig_z
        PlotObj.update_axes()
        PlotObj.update_image()
        PlotObj.fig.canvas.draw()

    return PlotObj


class PlotDTIDynamic:

    def __init__(
        self,
        FA: np.ndarray,
        MD: np.ndarray,
        MO: np.ndarray,
        S0: np.ndarray,
        L1: np.ndarray,
        L2: np.ndarray,
        L3: np.ndarray,
        V1: np.ndarray,
        V2: np.ndarray,
        V3: np.ndarray,
        suptitle: str,
        transpose: bool = True,
        cfg: PlotConfig = PlotConfig(),
    ):
        # Unpack
        self.phase_scale = cfg.phase_scale
        self.background_color = cfg.background_color
        self.secondary_color = cfg.secondary_color
        self.plot_trim = cfg.plot_trim
        self.fov_shift = cfg.fov_shift

        # TODO: move basic preprocessing steps to generic infastructure
        if transpose:

            def tp(x):
                return np.flip(np.moveaxis(x, 1, 0), axis=(0, 1))

            FA = tp(FA)
            MD = tp(MD)
            MO = tp(MO)
            S0 = tp(S0)
            L1 = tp(L1)
            L2 = tp(L2)
            L3 = tp(L3)
            V1 = tp(V1)
            V2 = tp(V2)
            V3 = tp(V3)

            if self.fov_shift is not None:
                self.fov_shift = [self.fov_shift[1], self.fov_shift[0]]
            if self.plot_trim is not None:
                self.plot_trim = [self.plot_trim[1], self.plot_trim[0]]

        # range for MD and L1-L3 and init mag guess S0
        L_MAX = np.percentile((np.stack([MD, L1, L2, L3], axis=-1)), 99.5)
        S_MAX = np.percentile(S0, 99.5)

        # fov shift
        if self.fov_shift is not None:

            def fshift(x):
                x = np.roll(x, self.fov_shift[0], axis=0)
                x = np.roll(x, self.fov_shift[1], axis=1)
                return x

            FA = fshift(FA)
            MD = fshift(MD)
            MO = fshift(MO)
            S0 = fshift(S0)
            L1 = fshift(L1)
            L2 = fshift(L2)
            L3 = fshift(L3)
            V1 = fshift(V1)
            V2 = fshift(V2)
            V3 = fshift(V3)

        # plotting trim
        self.full_im_shape = FA.shape[:2]
        if self.plot_trim is not None:
            X, Y = FA.shape[:2]
            self.px, self.py = slice(self.plot_trim[0], X - self.plot_trim[0]), slice(
                self.plot_trim[1], Y - self.plot_trim[1]
            )
            self.im_shape = (X - 2 * self.plot_trim[0], Y - 2 * self.plot_trim[1])
        else:
            self.px, self.py = slice(None), slice(None)
            self.im_shape = FA.shape[:2]

        # Save
        self.Nz = FA.shape[2]
        self.z = self.Nz // 2
        self.suptitle = suptitle
        self.FA = FA
        self.MD = MD
        self.MO = MO
        self.S0 = S0
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.V1 = V1
        self.V2 = V2
        self.V3 = V3

        # larger font
        plt.rcParams.update({"font.size": cfg.font_size})

        # Initialize Figure. TODO: make sizings more intuitive / parameterizeable
        W, H = (self.im_shape[1] / self.im_shape[0]) * 1, 1
        S = cfg.figsize[0]
        W_CBAR = 0.15  # TODO: play with
        FF = 0.1
        figsize = (W * S * (5 + 2 * W_CBAR), H * S * (2 + FF))
        fig = plt.figure(figsize=figsize, dpi=cfg.dpi)

        # gridspec
        # from matplotlib.gridspec import GridSpec
        gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, W_CBAR, 1, W_CBAR])
        ax = gs.subplots()

        # Save Attribues
        self.fig = fig
        self.ax = ax
        self.axim = np.zeros((2, 5), dtype=object)

        # put colorbar in center of axis instead of to the right
        cbar_pm = dict(location="right", fraction=0.15)  # fraction=0.05, pad=0.05)
        cbar_ax_params = dict(width="25%", height="72%", loc="center left")
        cbars = []

        self.axim[0, 0] = ax[0, 0].imshow(V1[self.px, self.py, self.z, :])
        self.axim[0, 1] = ax[0, 1].imshow(V2[self.px, self.py, self.z, :])
        self.axim[0, 2] = ax[0, 2].imshow(V3[self.px, self.py, self.z, :])

        self.axim[0, 3] = ax[0, 3].imshow(
            FA[self.px, self.py, self.z], cmap="gray", vmin=0, vmax=1
        )
        cbars.append(
            fig.colorbar(
                self.axim[0, 3],
                cax=inset_axes(ax[0, 4], **cbar_ax_params),
                format="%d",
                ticks=[0, 1],
                **cbar_pm,
            )
        )

        self.axim[0, 4] = ax[0, 5].imshow(
            MO[self.px, self.py, self.z], cmap="coolwarm", vmin=-1, vmax=1
        )
        cbars.append(
            fig.colorbar(
                self.axim[0, 4],
                cax=inset_axes(ax[0, 6], **cbar_ax_params),
                format="%d",
                ticks=[-1, 0, 1],
                **cbar_pm,
            )
        )

        self.axim[1, 0] = ax[1, 0].imshow(
            L1[self.px, self.py, self.z], cmap="gray", vmin=0, vmax=L_MAX
        )
        self.axim[1, 1] = ax[1, 1].imshow(
            L2[self.px, self.py, self.z], cmap="gray", vmin=0, vmax=L_MAX
        )
        self.axim[1, 2] = ax[1, 2].imshow(
            L3[self.px, self.py, self.z], cmap="gray", vmin=0, vmax=L_MAX
        )
        self.axim[1, 3] = ax[1, 3].imshow(
            MD[self.px, self.py, self.z], cmap="gray", vmin=0, vmax=L_MAX
        )
        cbars.append(
            fig.colorbar(
                self.axim[1, 3],
                cax=inset_axes(ax[1, 4], **cbar_ax_params),
                format="%0.4f",
                **cbar_pm,
            )
        )

        self.axim[1, 4] = ax[1, 5].imshow(
            S0[self.px, self.py, self.z], cmap="gray", vmin=0, vmax=S_MAX
        )
        cbars.append(
            fig.colorbar(
                self.axim[1, 4],
                cax=inset_axes(ax[1, 6], **cbar_ax_params),
                format="%d",
                **cbar_pm,
            )
        )

        ax[0, 0].set_title("V1")
        ax[0, 1].set_title("V2")
        ax[0, 2].set_title("V3")
        ax[0, 3].set_title("FA")
        ax[0, 4].set_title("  ")
        ax[0, 5].set_title("MO")
        ax[1, 0].set_title("L1")
        ax[1, 1].set_title("L2")
        ax[1, 2].set_title("L3")
        ax[1, 3].set_title("MD")
        ax[1, 4].set_title("  ")
        ax[1, 5].set_title("S0")

        for cbar in cbars:
            cbar.ax.yaxis.set_tick_params(labelsize=round(cfg.font_size * 0.8))

        for a in ax.ravel():
            a.grid(False)
            a.set_xticks([])
            a.set_yticks([])
            for spine in a.spines.values():
                spine.set_visible(False)

        fig, ax = dark_mode(
            fig,
            ax,
            cbars=cbars,
            background_color=self.background_color,
            secondary_color=self.secondary_color,
        )

        self.fig.suptitle(
            f"Slice {self.z}: {self.suptitle}", color=self.secondary_color
        )

        # ignore warning about tight layout ncluding bad axes
        warnings.filterwarnings("ignore")
        self.fig.tight_layout()
        warnings.resetwarnings()

        # Setup key press
        self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        self.update_axes()
        self.update_image()
        self.fig.canvas.draw()

    def key_press(self, event):
        if event.key == "left":
            self.z = (self.z - 1) % self.Nz

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == "right":
            self.z = (self.z + 1) % self.Nz

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

    def update_axes(self):
        self.fig.suptitle(
            f"Slice {self.z}: {self.suptitle}", color=self.secondary_color
        )

    def update_image(self):
        self.axim[0, 0].set_data(self.V1[self.px, self.py, self.z, :])
        self.axim[0, 1].set_data(self.V2[self.px, self.py, self.z, :])
        self.axim[0, 2].set_data(self.V3[self.px, self.py, self.z, :])
        self.axim[0, 3].set_data(self.FA[self.px, self.py, self.z])
        self.axim[0, 4].set_data(self.MO[self.px, self.py, self.z])
        self.axim[1, 0].set_data(self.L1[self.px, self.py, self.z])
        self.axim[1, 1].set_data(self.L2[self.px, self.py, self.z])
        self.axim[1, 2].set_data(self.L3[self.px, self.py, self.z])
        self.axim[1, 3].set_data(self.MD[self.px, self.py, self.z])
        self.axim[1, 4].set_data(self.S0[self.px, self.py, self.z])

    def launch(self):
        plt.show()
