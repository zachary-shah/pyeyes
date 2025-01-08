import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from typing import Union, Sequence, Optional
from tqdm import tqdm

from ..utils import tonp, normalize, RMSE
from .mpl_config import PlotConfig, get_line_colors

# Custom Color Maps. TODO: move all custom color stuff to its own file
from matplotlib.colors import LinearSegmentedColormap
def get_jet_black_cmp(ap = 1.2):
    jet = plt.cm.get_cmap('jet', 256)
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

def dark_mode(fig, ax, 
              cbars: Optional[Sequence[plt.colorbar]] = None,
              background_color = 'black',
              secondary_color = 'white'):

    fig.patch.set_facecolor(background_color)
    if fig._suptitle is not None:
        fig.suptitle(fig._suptitle.get_text(), color=secondary_color)
    if isinstance(ax, np.ndarray):
        for a in ax.ravel():
            a.set_facecolor(background_color)                    
            plt.setp(a.spines.values(), color=secondary_color)   
            a.tick_params(axis='both', colors=secondary_color)  
            a.xaxis.label.set_color(secondary_color)             
            a.yaxis.label.set_color(secondary_color)             
            a.title.set_color(secondary_color)           
    else:
        ax.set_facecolor(background_color)                    
        plt.setp(ax.spines.values(), color=secondary_color)   
        ax.tick_params(axis='both', colors=secondary_color)  
        ax.xaxis.label.set_color(secondary_color)             
        ax.yaxis.label.set_color(secondary_color)             
        ax.title.set_color(secondary_color)         

    if cbars is not None:
        for cbar in cbars:
            cbar.ax.yaxis.set_tick_params(color=secondary_color)
            plt.setp(cbar.ax.get_yticklabels(), color=secondary_color)

    return fig, ax

# Get list of colors for plotting
JET_ERROR_CMAP = get_jet_black_cmp()

"""
Gifs
"""
def plot_cplx_recons_gif(recons: Sequence[np.ndarray],
                    recons_titles: Sequence[str],
                    times: np.ndarray,
                    mask: Optional[np.ndarray] = None,                       
                    cfg: PlotConfig = PlotConfig(),
                    fps: int = 5,
                    mode='time',
                    description: str = ""):
                       
    """
    Plot a time-evolving gif of set of complex-valued recons
    Will generate a gif showing a (2xN) of images evolving over time, where N is number of 
    image arrays in image_list.

    Each image array in the list should have the same shape of (Nt, Nx, Ny), or (Nz, Nx, Ny). 
    
    TODO: implement 3D case
    TODO: add reference option / just magnitude and difference with reference option
    
    Parameters
    ----------
    recons : np.ndarray
        Estimated maps. Expect sequence of [(gif_dim, Nx, Ny)]
    recons_titles : list[str]
        Titles for each recon in sequence
    plot_times : np.ndarray
        Times corresponding to maps. Expect (Nt, ). 
        OR, if mode is 'z', then expect (Nz, )
    mask : np.ndarray
        Mask to use for plotting. Expect (Nx, Ny)
    cfg : PlotConfig
        Plotting configuration object

    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """

    # No reason to do gif plotting in this case
    if not cfg.save: 
        return
    
    print(f"Saving gif of recons evolution...")
    
    # Validate inputs
    times = tonp(times)
    Nt = len(times) 
    N_method = len(recons)

    if mask is not None:
        assert mask.shape == recons[0].shape[1:], "Mask must have same shape as images"
        raise NotImplementedError("Mask not supported yet")

    # extract plotted frames
    stride = max(Nt // cfg.N_gif, 1)
    for i in range(N_method):
        assert recons[i].shape[0] == Nt, "Estimates and times of different length"
        recons[i] = tonp(recons[i][::stride])
    times = times[::stride]
    N_gif_eff = len(times)

    if N_gif_eff < 3:
        print("Not enough frames to generate a gif. Skipping.")
        return
    
    # scale images magnitude to [0,1]
    for i in range(N_method):
        recons[i] = recons[i] / np.max(np.abs(recons[i]))
    
    # figure
    # plt.rcParams.update({'font.size': cfg.font_size})
    H, W = 5, 5
    fig, axs = plt.subplots(2, N_method, figsize=((N_method)*W, 2*H))
    if N_method == 1:
        axs = np.expand_dims(axs, axis=1)

    # formatting            
    titles = recons_titles
    for ax in axs.ravel():
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    # initialize map images
    im_mags, im_phases = [], []
    for i in range(N_method):
        im_mag = axs[0,i].imshow(np.abs(recons[i][0]), cmap='gray', vmin=0, vmax=1)
        im_phase = axs[1,i].imshow(np.angle(recons[i][0]), cmap='jet', vmin=-np.pi, vmax=np.pi)
        axs[0,i].set_title(titles[i])
        im_mags.append(im_mag)
        im_phases.append(im_phase)
    tstr = f"t={times[0]:.2f}ms" if mode == 'time' else f"z={times[0]:.2f}mm"
    suptitle = fig.suptitle(f"Recons for {tstr}")
    fig.tight_layout()

    # animate
    pbar = tqdm(range(N_gif_eff), desc="Generating animation frames", leave=False)
    def update(k):
        for i in range(N_method):
            im_mags[i].set_data(np.abs(recons[i][k]))
            im_phases[i].set_data(np.angle(recons[i][k]))
        tstr = f"t={times[k]:.2f}ms" if mode == 'time' else f"z={times[k]:.2f}mm"
        suptitle.set_text(f"Recons for {tstr}")
        pbar.update(1)
        return im_mags + im_phases + [suptitle]
    
    ani = FuncAnimation(fig, update, frames=N_gif_eff, blit=True)
    output_path = os.path.join(cfg.figdir, f"recons-{description}.gif")
    ani.save(output_path, writer='imagemagick', fps = fps)
    plt.close(fig)
    return 

"""
From Festive
"""
def imshow_formatted(
        im: np.ndarray,
        title: str = None,
        cmap: str = 'gray',
        vmin: float = None,
        vmax: float = None,
        colorbar: bool = True,
        save_name: str = None,
        cfg: PlotConfig = PlotConfig()):
    
    """
    Plot the loss of the model over time

    Parameters
    ----------
    losses : np.ndarray
        Losses of the model. Expect (Nt, )
    cfg : PlotConfig
        Plotting configuration object
    cfgMdl : FestiveModelConfig
        Model configuration object
    """

    # Set color of everytihng based on plot config
    background_color = cfg.background_color
    secondary_color = cfg.secondary_color

    fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=cfg.dpi)
    axim = ax.imshow(im.T, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)

    if colorbar:
        cbar = plt.colorbar(axim, ax=ax, shrink=0.7)
        cbar.ax.yaxis.set_tick_params(color=secondary_color)
        plt.setp(cbar.ax.get_yticklabels(), color=secondary_color)
    
    # Set Colors
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    plt.setp(ax.spines.values(), color=secondary_color)   # Make axis spines white
    ax.tick_params(axis='both', colors=secondary_color)   # Set ticks (numbers) to white
    ax.xaxis.label.set_color(secondary_color)             # Set X-axis label to white
    ax.yaxis.label.set_color(secondary_color)             # Set Y-axis label to white
    ax.title.set_color(secondary_color)                   # Set title color to white

    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()

    if cfg.save and save_name is not None:
        plt.savefig(os.path.join(cfg.figdir, f"{save_name}.png"))

    return fig, ax

def plot_recon_errors(imgs, 
                      ref_img,
                      titles, 
                      ref_title = "Reference",
                      vmin=None, 
                      vmax=None, 
                      cmap='gray',
                      suptitle=None,
                      cplx_norm=True,
                      show_nrmse=True,
                      mode = 'mag_phase', # mag_phase or real_imag
                      cfg: PlotConfig = PlotConfig()):
    """
    Generate a 3 x N plot of magnitude, error, and phase of images

    Parameters
    ----------
    imgs : list[np.ndarray]
        List of images to plot. Each image should be (Nx, Ny) or (Nx, Ny, Nz)
    ref_img : np.ndarray
        Reference image to compare against. Should be (Nx, Ny) or (Nx, Ny, Nz)
    titles : list[str]
        Titles for each image
    ref_title : str
        Title for reference image
    vmin : float
        Minimum value for colormap
    vmax : float
        Maximum value for colormap
    suptitle : str
        Title for the entire figure
    cfg : PlotConfig

    Returns
    -------
    fig, ax, nrmses : matplotlib figure and axis objects, and list of NRMSE values for each recon
    """

    imgs = [tonp(img) for img in imgs]
    ref_img = tonp(ref_img)

    if len(imgs[0].shape) == 2:
        imgs = [img[...,None] for img in imgs]
        ref_img = ref_img[...,None]
    
    Nz = imgs[0].shape[-1]

    if vmin is None:
        vmin = 0
    if vmax is None: 
        vmax = np.max(np.abs(ref_img))

    # plt.rcParams.update({'font.size': cfg.font_size})

    PlotObj = PlotReconErrorDynamic(
        imgs,
        ref_img,
        titles,
        ref_title,
        vmin,
        vmax,
        cmap,
        suptitle,
        cplx_norm,
        show_nrmse,
        mode = mode,
        transpose=True,
        cfg = cfg,
    )


    # Save slices
    if cfg.save:
        orig_z = PlotObj.z
        for z in range(Nz):
            PlotObj.z = z
            PlotObj.update_axes()
            PlotObj.update_image()
            PlotObj.fig.canvas.draw()
            PlotObj.fig.savefig(os.path.join(cfg.figdir, f"recons_slc{z}.png"))
        PlotObj.z = orig_z
        PlotObj.update_axes()
        PlotObj.update_image()
        PlotObj.fig.canvas.draw()

    return PlotObj, PlotObj, PlotObj.nrmses

class PlotReconErrorDynamic:

    def __init__(
        self,
        imgs: list, # Must be (X, Y, Z)
        ref_img: np.ndarray,
        titles: list[str],
        ref_title: str,
        vmin: float,
        vmax: float, 
        cmap: str = 'gray',
        suptitle: str = None,
        cplx_norm: bool = True,
        show_nrmse: bool = True,
        mode: str = 'mag_phase', # mag_phase or real_imag
        transpose: bool = True,
        cfg: PlotConfig = PlotConfig(),
    ):
        
        # Unpack
        dpi = cfg.dpi
        phase_scale = cfg.phase_scale
        error_scale = cfg.error_scale
        background_color = cfg.background_color
        secondary_color = cfg.secondary_color
        plot_trim = cfg.plot_trim
        fov_shift = cfg.fov_shift
        font_size = 22

        assert mode in ['mag_phase', 'real_imag'], "Mode must be 'mag_phase' or 'real_imag'"

        # Precomputes
        imgs = np.stack([tonp(img) for img in imgs], axis=0) # Nm *im_shape Nz
        ref_img = tonp(ref_img)

        if transpose:
            imgs = np.transpose(imgs, (0, 2, 1, 3)) # for brain
            imgs = np.flip(imgs, axis=(1,2)) # for brain
            ref_img = np.transpose(ref_img, (1, 0, 2)) # for brain
            ref_img = np.flip(ref_img, axis=(0,1))
            if plot_trim is not None:
                plot_trim = [plot_trim[1], plot_trim[0]]
            if fov_shift is not None:
                fov_shift = [fov_shift[1], fov_shift[0]]
            
        # fov shift
        if fov_shift is not None:
            imgs = np.roll(imgs, fov_shift[0], axis=1)
            imgs = np.roll(imgs, fov_shift[1], axis=2)
            ref_img = np.roll(ref_img, fov_shift[0], axis=0)
            ref_img = np.roll(ref_img, fov_shift[1], axis=1)

        # plotting trim 
        self.full_im_shape = imgs.shape[1:-1]
        if plot_trim is not None:
            X, Y = imgs.shape[1:3]
            self.pslc = slice(plot_trim[0], X - plot_trim[0]), slice(plot_trim[1], Y - plot_trim[1])
            self.im_shape = (X - 2*plot_trim[0], Y - 2*plot_trim[1])
        else:
            self.pslc = (slice(None), slice(None))
            self.im_shape = imgs.shape[1:-1]

        self.Nz = imgs[0].shape[-1]
        self.Nm = len(imgs)

        nrmses = np.zeros((self.Nm, self.Nz))
        diffs = np.zeros((self.Nm, *self.full_im_shape, self.Nz), dtype=np.float32)
        targs = np.zeros((self.Nm, *self.full_im_shape, self.Nz), dtype=np.complex64)

        if mode == 'real_imag':
            imgs = np.real(imgs) * np.exp(1j * np.imag(imgs)) # hacky way to convert to complex
            ref_img = np.real(ref_img) * np.exp(1j * np.imag(ref_img))
            

        for i in range(self.Nm):
            for z in range(self.Nz):
                if cplx_norm:
                    x = normalize(imgs[i,...,z], ref_img[...,z], ofs=True, mag=False)
                    diffs[i,...,z] = np.abs(ref_img[...,z] - x)
                    targs[i,...,z] = x
                    nrmses[i,z] = RMSE(x, ref_img[...,z])
                else:
                    x = normalize(imgs[i,...,z], ref_img[...,z], ofs=True, mag=True)
                    diffs[i,...,z] = np.abs(np.abs(ref_img[...,z]) - np.abs(x))
                    targs[i,...,z] = x
                    nrmses[i,z] = RMSE(np.abs(x), np.abs(ref_img[...,z]))

        # Initialize Figure
        W, H = (self.im_shape[1] / self.im_shape[0]), 1
        S = cfg.figsize[0]
        figsize = (W * S * (self.Nm + 1), H * S * 3.08)
        fig, ax = plt.subplots(3, self.Nm + 1, figsize=figsize, dpi=dpi)
    
        # Set fig background to black as well
        fig.patch.set_facecolor(background_color)

        # Formatting
        for a in ax.ravel():
            a.set_xticks([])
            a.set_yticks([])
            a.set_facecolor(background_color)
            a.axis('equal')

        # Save Attribues
        self.targs = targs
        self.diffs = diffs
        self.ref_img = ref_img
        self.nrmses = nrmses
        self.phase_scale = phase_scale
        self.error_scale = error_scale
        self.fig = fig
        self.ax = ax
        self.titles = titles
        self.ref_title = ref_title
        self.show_nrmse = show_nrmse
        self.cplx_norm = cplx_norm
        self.cmap = cmap
        self.z = self.Nz // 2 # Initial slice index
        self.vmin = vmin
        self.vmax = vmax
        self.suptitle = suptitle
        self.cfg = cfg
        self.background_color = background_color
        self.secondary_color = secondary_color
        self.font_size = font_size
        
        # ax im
        self.axim = np.zeros((3, self.Nm + 1), dtype=object)
        self.t = np.zeros((self.Nm,), dtype=object)
        for i in range(self.Nm + 1):
            self.axim[0,i] = ax[0,i].imshow(np.zeros(self.im_shape), vmin=vmin, vmax=vmax, cmap=cmap, extent=[0, self.im_shape[1], 0, self.im_shape[0]])
            self.axim[1,i] = ax[1,i].imshow(np.zeros(self.im_shape), vmin=vmin, vmax=vmax * error_scale, cmap=cmap, extent=[0, self.im_shape[1], 0, self.im_shape[0]])
            self.axim[2,i] = ax[2,i].imshow(np.zeros(self.im_shape), vmin= - np.pi * phase_scale, vmax = np.pi * phase_scale, cmap='jet', extent=[0, self.im_shape[1], 0, self.im_shape[0]])

            # nrmse
            if self.show_nrmse and i >= 1:
                self.t[i-1] = self.ax[0, i].text(
                    self.im_shape[1]//2, self.im_shape[0] - 5, f'NRMSE={self.nrmses[i-1,self.z]:0.4f}', 
                    color=self.secondary_color, fontsize=22, ha='center', va='center',
                )

        # Initialize stuff
        self.ax[0,0].set_ylabel("Magnitude", color=self.secondary_color, fontsize=font_size)
        self.ax[1,0].set_ylabel(f"Difference ({1/self.error_scale:.1f}x)", color=self.secondary_color, fontsize=font_size)
        self.ax[2,0].set_ylabel("Phase", color=self.secondary_color, fontsize=font_size)
        self.ax[0,0].set_title(self.ref_title, color=self.secondary_color, fontsize=font_size)
        for i in range(self.Nm):
            self.ax[0, i + 1].set_title(self.titles[i], color=self.secondary_color, fontsize=font_size)
        
        self.suptit = self.fig.suptitle(f"Slice {self.z}: {self.suptitle}", color=self.secondary_color, fontsize=int(font_size*1.3))
        # keep tight layout but dont let suptit overlap ax
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.94)


        # Setup
        # self.fig.canvas.mpl_disconnect(
        #     self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.update_axes()
        self.update_image()
        self.fig.canvas.draw()

    def key_press(self, event):
        if event.key == 'left':
            self.z = (self.z - 1) % self.Nz

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == 'right':
            self.z = (self.z + 1) % self.Nz

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

    def update_axes(self):
        self.suptit.set_text(f"Slice {self.z}: {self.suptitle}")

    def update_image(self):

        # Reference
        mag = np.abs(self.ref_img[...,self.z])
        phs = np.angle(self.ref_img[...,self.z])
        im_msk = mag < 1e-10 * np.max(mag)
        phs_msk = mag < 5e-2 * np.max(mag)
        phs[phs_msk] = np.nan
        mag[im_msk] = np.nan
        self.axim[0, 0].set_data(mag[self.pslc])
        self.axim[2, 0].set_data(phs[self.pslc])
        self.axim[0, 0].set_extent([0, self.im_shape[1], 0, self.im_shape[0]])
        self.axim[2, 0].set_extent([0, self.im_shape[1], 0, self.im_shape[0]])

        # targets
        for i in range(self.Nm):
            # Error
            diff = self.diffs[i, ..., self.z]
            diff[im_msk] = np.nan
            self.axim[1,i+1].set_data(diff[self.pslc])
            self.axim[1,i+1].set_extent([0, self.im_shape[1], 0, self.im_shape[0]])

            if self.show_nrmse:
                self.t[i].set_text(f'NRMSE={self.nrmses[i,self.z]:0.4f}')
            
            # Recons 
            if np.iscomplexobj(self.targs[i,...,self.z]):
                # mag
                mag = np.abs(self.targs[i,...,self.z])
                mag[im_msk] = np.nan
                self.axim[0, i+1].set_data(mag[self.pslc])
                
                # phase 
                phs = np.angle(self.targs[i,...,self.z])
                # mask phase
                phs[phs_msk] = np.nan
                self.axim[2, i+1].set_data(phs[self.pslc])
            else:
                self.axim[0, i+1].set_data(self.targs[i,self.pslc,self.z])        
                self.axim[2, i+1].set_data(np.zeros(self.im_shape) * np.nan)

            for j in range(3):
                self.axim[j, i+1].set_extent([0, self.im_shape[1], 0, self.im_shape[0]])

        return
                                   
def plot_recon_errors_color(imgs, 
                      ref_img,
                      titles, 
                      ref_title = "Reference",
                      suptitle=None,
                      show_nrmse=True,
                      mask=None,
                      norm_scale=False,
                      cfg: PlotConfig = PlotConfig()):
    """
    Generate a 2 x N plot of magnitude and error plots of images for images with color channels.

    Parameters
    ----------
    imgs : list[np.ndarray]
        List of images to plot. Each image should be (Nx, Ny, C) or (Nx, Ny, Nz, C)
    ref_img : np.ndarray
        Reference image to compare against. Should be (Nx, Ny, C) or (Nx, Ny, Nz, C)
    titles : list[str]
        Titles for each image
    ref_title : str
        Title for reference image
    vmin : float
        Minimum value for colormap
    vmax : float
        Maximum value for colormap
    suptitle : str
        Title for the entire figure
    cfg : PlotConfig

    Returns
    -------
    fig, ax, nrmses : matplotlib figure and axis objects, and list of NRMSE values for each recon
    """
    imgs = [tonp(img) for img in imgs]
    ref_img = tonp(ref_img)

    # Images must be real-valued
    if np.iscomplexobj(imgs[0]):
        print(f"Warning: Input color images are complex-valued. Taking magnitude of images.")
    
    # Magnitude
    imgs = [np.abs(img) for img in imgs]
    ref_img = np.abs(ref_img)

    if len(imgs[0].shape) == 3:
        imgs = [img[...,None, :] for img in imgs]
        ref_img = ref_img[...,None, :]

    assert len(imgs[0].shape) == 4, "Input images must have color channels"

    # Handle grey-scale
    if imgs[0].shape[-1] == 1:
        imgs = [np.stack([img] * 3, axis=-1) for img in imgs]   
    if ref_img.shape[-1] == 1:
        ref_img = np.stack([ref_img] * 3, axis=-1)
    
    Nz = imgs[0].shape[-2]

    # plt.rcParams.update({'font.size': cfg.font_size})

    PlotObj = PlotReconErrorColorDynamic(
        imgs,
        ref_img,
        titles,
        ref_title,
        suptitle,
        show_nrmse,
        transpose=True,
        mask=mask,
        norm_scale=norm_scale,
        cfg = cfg,
    )


    # Save slices
    if cfg.save:
        orig_z = PlotObj.z
        for z in range(Nz):
            PlotObj.z = z
            PlotObj.update_axes()
            PlotObj.update_image()
            PlotObj.fig.canvas.draw()
            PlotObj.fig.savefig(os.path.join(cfg.figdir, f"recons_slc{z}.png"))
        PlotObj.z = orig_z
        PlotObj.update_axes()
        PlotObj.update_image()
        PlotObj.fig.canvas.draw()

    return PlotObj, PlotObj, PlotObj.nrmses

class PlotReconErrorColorDynamic:

    def __init__(
        self,
        imgs: list, # Must be (X, Y, Z, C)
        ref_img: np.ndarray,
        titles: list[str],
        ref_title: str,
        suptitle: str = None,
        show_nrmse: bool = True,
        transpose: bool = True,
        mask: np.ndarray = None,
        norm_scale: bool = False,
        cfg: PlotConfig = PlotConfig(),
    ):
        
        # Unpack
        dpi = cfg.dpi
        phase_scale = cfg.phase_scale
        error_scale = cfg.error_scale
        background_color = cfg.background_color
        secondary_color = cfg.secondary_color
        plot_trim = cfg.plot_trim
        fov_shift = cfg.fov_shift
        font_size = 22

        # Precomputes
        imgs = np.stack([tonp(img) for img in imgs], axis=0) # Nm X Y Z C
        ref_img = tonp(ref_img)

        if mask is not None:
            assert mask.shape == imgs.shape[1:-1], "Mask must have same shape as images"
        else:
            mask = np.ones(imgs.shape[1:-1])

        if transpose:
            imgs = np.transpose(imgs, (0, 2, 1, 3, 4)) # for brain
            imgs = np.flip(imgs, axis=(1,2)) # for brain
            ref_img = np.transpose(ref_img, (1, 0, 2, 3)) # for brain
            ref_img = np.flip(ref_img, axis=(0,1))
            mask = np.transpose(mask, (1, 0, 2)) # for brain
            mask = np.flip(mask, axis=(0,1))
            if plot_trim is not None:
                plot_trim = [plot_trim[1], plot_trim[0]]
            if fov_shift is not None:
                fov_shift = [fov_shift[1], fov_shift[0]]
            
        # fov shift
        if fov_shift is not None:
            imgs = np.roll(imgs, fov_shift[0], axis=1)
            imgs = np.roll(imgs, fov_shift[1], axis=2)
            ref_img = np.roll(ref_img, fov_shift[0], axis=0)
            ref_img = np.roll(ref_img, fov_shift[1], axis=1)
            mask = np.roll(mask, fov_shift[0], axis=0)
            mask = np.roll(mask, fov_shift[1], axis=1)

        # plotting trim 
        self.full_im_shape = imgs.shape[1:3]
        if plot_trim is not None:
            X, Y = imgs.shape[1:3]
            self.pslc = (slice(plot_trim[0], X - plot_trim[0]), slice(plot_trim[1], Y - plot_trim[1]), slice(None))
            self.im_shape = (X - 2*plot_trim[0], Y - 2*plot_trim[1])
        else:
            self.pslc = (slice(None), slice(None), slice(None))
            self.im_shape = imgs.shape[1:3]

        self.Nm = imgs.shape[0]
        self.Nz = imgs.shape[-2]
        self.C  = imgs.shape[-1]

        nrmses = np.zeros((self.Nm, self.Nz))
        diffs = np.zeros((self.Nm, *self.full_im_shape, self.Nz, self.C), dtype=np.float32)
        targs = np.zeros((self.Nm, *self.full_im_shape, self.Nz, self.C), dtype=np.float32)

        # normalization
        for i in range(self.Nm):
            for z in range(self.Nz):
                if norm_scale:
                    x = np.abs(normalize(imgs[i,...,z, :], ref_img[...,z, :], ofs=True, mag=True))
                else:
                    x = imgs[i,..., z, :]
                
                diffs[i,...,z, :] = np.abs(np.abs(ref_img[..., z, :]) - np.abs(x))
                
                targs[i,...,z, :] = x
                
                nrmses[i,z] = RMSE(np.abs(x), np.abs(ref_img[...,z, :]))

        # Initialize Figure
        H, W = (self.im_shape[1] / self.im_shape[0]) * 8, 8
        fig, ax = plt.subplots(2, self.Nm + 1, figsize=(H*(self.Nm+1),W*2.06), dpi=dpi)
    
        # Set fig background to black as well
        fig.patch.set_facecolor(background_color)

        # Formatting
        for a in ax.ravel():
            a.set_xticks([])
            a.set_yticks([])
            a.set_facecolor(background_color)
            a.axis('equal')

        # Save Attribues
        self.targs = targs
        self.diffs = diffs
        self.ref_img = ref_img
        self.nrmses = nrmses
        self.phase_scale = phase_scale
        self.error_scale = error_scale
        self.fig = fig
        self.ax = ax
        self.titles = titles
        self.ref_title = ref_title
        self.show_nrmse = show_nrmse
        self.z = self.Nz // 2 # Initial slice index
        self.suptitle = suptitle
        self.cfg = cfg
        self.background_color = background_color
        self.secondary_color = secondary_color
        self.font_size = font_size
        self.mask = mask
        
        # ax im
        self.axim = np.zeros((2, self.Nm + 1), dtype=object)
        self.t = np.zeros((self.Nm,), dtype=object)
        for i in range(self.Nm + 1):
            self.axim[0,i] = ax[0,i].imshow(np.zeros((*self.im_shape, 3), dtype=np.float32), extent=[0, self.im_shape[1], 0, self.im_shape[0]])
            self.axim[1,i] = ax[1,i].imshow(np.zeros((*self.im_shape, 3), dtype=np.float32), extent=[0, self.im_shape[1], 0, self.im_shape[0]])

            # nrmse
            if self.show_nrmse and i >= 1:
                self.t[i-1] = self.ax[0, i].text(
                    self.im_shape[1]//2, self.im_shape[0] - 5, f'NRMSE={self.nrmses[i-1,self.z]:0.4f}', 
                    color=self.secondary_color, fontsize=22, ha='center', va='center',
                )

        # Initialize stuff
        self.ax[0,0].set_ylabel("Magnitude", color=self.secondary_color, fontsize=font_size)
        self.ax[1,0].set_ylabel(f"Difference ({1/self.error_scale:.1f}x)", color=self.secondary_color, fontsize=font_size)
        self.ax[0,0].set_title(self.ref_title, color=self.secondary_color, fontsize=font_size)
        for i in range(self.Nm):
            self.ax[0, i + 1].set_title(self.titles[i], color=self.secondary_color, fontsize=font_size)
        
        self.suptit = self.fig.suptitle(f"Slice {self.z}: {self.suptitle}", color=self.secondary_color, fontsize=int(font_size*1.3))
        # keep tight layout but dont let suptit overlap ax
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.94)


        # Setup
        # self.fig.canvas.mpl_disconnect(
        #     self.fig.canvas.manager.key_press_handler_id)
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.update_axes()
        self.update_image()
        self.fig.canvas.draw()

    def key_press(self, event):
        if event.key == 'left':
            self.z = (self.z - 1) % self.Nz

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

        elif event.key == 'right':
            self.z = (self.z + 1) % self.Nz

            self.update_axes()
            self.update_image()
            self.fig.canvas.draw()

    def update_axes(self):
        self.suptit.set_text(f"Slice {self.z}: {self.suptitle}")

    def update_image(self):

        # Reference
        mag = np.abs(self.ref_img[...,self.z, :])
        im_msk = (mag < 1e-8 * np.max(mag)) 
        # mag2 = np.sum(np.abs(self.targs[...,self.z, :]), axis=0)
        # mag3 = np.abs(self.targs[0, ..., self.z, :])
        # im_msk = (mag < 1e-3 * np.max(mag)) & (mag2 < 1e-3 * np.max(mag2)) & (mag3 < 1e-3 * np.max(mag3))
        
        mag[im_msk] = np.nan
        mag *= self.mask[...,self.z][...,None]
        mag = np.clip(mag, 0, 1)
        self.axim[0, 0].set_data(mag[self.pslc])
        self.axim[0, 0].set_extent([0, self.im_shape[1], 0, self.im_shape[0]])

        # targets
        for i in range(self.Nm):
            # Error
            diff = self.diffs[i, ..., self.z, :] * 1 / self.error_scale
            diff[im_msk] = np.nan
            diff = np.clip(diff, 0, 1)
            diff *= self.mask[...,self.z][...,None]

            self.axim[1,i+1].set_data(diff[self.pslc])
            self.axim[1,i+1].set_extent([0, self.im_shape[1], 0, self.im_shape[0]])

            if self.show_nrmse:
                self.t[i].set_text(f'NRMSE={self.nrmses[i,self.z]:0.4f}')
            
            # Recons 
            tg = self.targs[i,...,self.z, :]
            tg *= self.mask[...,self.z][...,None]
            tg = np.clip(tg, 0, 1)
            self.axim[0, i+1].set_data(tg[self.pslc])        

            for j in range(2):
                self.axim[j, i+1].set_extent([0, self.im_shape[1], 0, self.im_shape[0]])

        return
