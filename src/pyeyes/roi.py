"""
Functions to help with plotting
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Sequence, Union

import cv2
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .utils import tonp
from .plotting import dark_mode

@dataclass
class ROIFeature:
    """
    Dataclass to store the features of a region of interest (ROI) in an image
    """

    upper_right_corner: tuple = field(default_factory=(350, 300))
    lower_left_corner: tuple = field(default_factory=(400, 350))
    interp: str = "bilinear"
    amp_cmap: str = "inferno"
    rectangle_color: str = "r"
    rectangle_linewidth: int = 1
    roi_scale: float = 1.0

    # worry about these if placing ROI within original image
    amp_factor: int = 2
    roi_placement_corner: str = "lower_right"
    edge_placement_offset: int = 5


# Default config for similar regions in the image
default_roi_diff = ROIFeature(
    upper_right_corner=(350, 290),
    lower_left_corner=(400, 340),
    interp="bilinear",
    amp_cmap="inferno",
    rectangle_color="r",
    rectangle_linewidth=2,
    amp_factor=3,
    roi_placement_corner="lower_right",
    edge_placement_offset=5,
)
# Default config for different regions in the image
default_roi_sim = ROIFeature(
    upper_right_corner=(140, 300),
    lower_left_corner=(190, 350),
    interp="bilinear",
    amp_cmap="inferno",
    rectangle_color="g",
    rectangle_linewidth=2,
    amp_factor=3,
    roi_placement_corner="upper_left",
    edge_placement_offset=5,
)


def get_roi(img: np.ndarray, roi_cfg: ROIFeature, return_img_size: bool = False):
    """
    Return the region of interest from the input image specified by the lower and upper corner

    If amp_factor is None, return roi in same shape as image
    """

    x0, y0 = roi_cfg.upper_right_corner
    x1, y1 = roi_cfg.lower_left_corner
    region = img[y0:y1, x0:x1]

    if roi_cfg.interp == "nearest":
        interpolator = cv2.INTER_NEAREST
    elif roi_cfg.interp == "bilinear":
        interpolator = cv2.INTER_LINEAR
    elif roi_cfg.interp == "bicubic":
        interpolator = cv2.INTER_CUBIC
    else:
        raise ValueError("Invalid interpolation method")

    if return_img_size:
        new_shape = (img.shape[1], img.shape[0])
    else:
        new_shape = (
            region.shape[1] * roi_cfg.amp_factor,
            region.shape[0] * roi_cfg.amp_factor,
        )

    roi = cv2.resize(
        region,
        new_shape,
        interpolation=interpolator,
    )

    # add scaling
    roi = np.clip(roi * roi_cfg.roi_scale, 0, 1)

    return roi


def place_rectangle(ax, roi_cfg: ROIFeature):

    x_0, y_0 = roi_cfg.upper_right_corner
    x_1, y_1 = roi_cfg.lower_left_corner
    box_height = y_1 - y_0
    box_width = x_1 - x_0

    rect = patches.Rectangle(
        (x_0, y_0),
        box_width,
        box_height,
        linewidth=roi_cfg.rectangle_linewidth,
        edgecolor=roi_cfg.rectangle_color,
        facecolor="none",
        linestyle="solid",
    )
    ax.add_patch(rect)
    return ax


def place_roi_and_mask(ax, img, roi, roi_cfg: ROIFeature):

    x_0, y_0 = roi_cfg.upper_right_corner
    x_1, y_1 = roi_cfg.lower_left_corner
    box_height = y_1 - y_0
    box_width = x_1 - x_0
    amp_height, amp_width = (
        box_height * roi_cfg.amp_factor,
        box_width * roi_cfg.amp_factor,
    )

    height, width = img.shape[:2]

    if roi_cfg.roi_placement_corner == "lower_right":
        amp_x0 = width - amp_width - roi_cfg.edge_placement_offset
        amp_y0 = height - amp_height - roi_cfg.edge_placement_offset
    elif roi_cfg.roi_placement_corner == "lower_left":
        amp_x0 = roi_cfg.edge_placement_offset
        amp_y0 = height - amp_height - roi_cfg.edge_placement_offset
    elif roi_cfg.roi_placement_corner == "upper_right":
        amp_x0 = width - amp_width - roi_cfg.edge_placement_offset
        amp_y0 = roi_cfg.edge_placement_offset
    elif roi_cfg.roi_placement_corner == "upper_left":
        amp_x0 = roi_cfg.edge_placement_offset
        amp_y0 = roi_cfg.edge_placement_offset

    # create image for amplified region
    amp_image = np.zeros_like(img)
    amp_mask = np.ones_like(img)
    amp_mask[amp_y0 : amp_y0 + amp_height, amp_x0 : amp_x0 + amp_width] = 0
    amp_image[amp_y0 : amp_y0 + amp_height, amp_x0 : amp_x0 + amp_width] = roi
    amp_image = np.ma.masked_array(amp_image, mask=amp_mask)

    ax.imshow(amp_image, cmap=roi_cfg.amp_cmap, vmin=0, vmax=1)

    rect = patches.Rectangle(
        (amp_x0, amp_y0),
        amp_width,
        amp_height,
        linewidth=roi_cfg.rectangle_linewidth,
        edgecolor=roi_cfg.rectangle_color,
        facecolor="none",
        linestyle="solid",
    )
    ax.add_patch(rect)

    return ax


def crop_img(img, crops):
    # Crop image to region of interest for plotting
    orig_img_size = img.shape[:2]

    if crops is None:
        return img

    crop_left, crop_right, crop_top, crop_bottom = crops

    if sum(crops) == 0:
        return img

    # ensure image stays square
    if (crop_left + crop_right) != (crop_bottom + crop_top):
        print("Cropping is not symmetric. Check the crop values.")
        crop_left = crop_bottom = min(crop_left, crop_bottom)
        crop_right = crop_top = min(crop_right, crop_top)

    img = img[
        crop_top : orig_img_size[0] - crop_bottom,
        crop_left : orig_img_size[1] - crop_right,
    ]

    # rescale to original img size
    img = cv2.resize(
        img,
        orig_img_size,
        interpolation=cv2.INTER_CUBIC,
    )

    return img


def remap_roi_locs_for_crop(img_size, roi_cfg: ROIFeature, crops):
    """
    Given an image and the cropping used to plot an image, return
    the pixel locations of the upper right and lower left corners of the
    ROI on the cropped image.
    """

    px, py = np.arange(img_size[0]), np.arange(img_size[1])

    px_cropped = px[crops[0] : img_size[1] - crops[1]]
    py_cropped = py[crops[2] : img_size[0] - crops[3]]

    px_interp = np.linspace(px_cropped[0], px_cropped[-1], img_size[1])
    py_interp = np.linspace(py_cropped[0], py_cropped[-1], img_size[0])

    # find closest location in pixel locations array to the upper right corner
    urc = roi_cfg.upper_right_corner
    lrc = roi_cfg.lower_left_corner

    urc_new = (
        np.argmin(np.abs(px_interp - urc[0])),
        np.argmin(np.abs(py_interp - urc[1])),
    )
    lrc_new = (
        np.argmin(np.abs(px_interp - lrc[0])),
        np.argmin(np.abs(py_interp - lrc[1])),
    )

    return urc_new, lrc_new


def amplify_plot_regions(img, ax, roi_cfgs: Union[ROIFeature, Sequence[ROIFeature]]):
    """
    Plots a region of the image specified by lower and upper corner
    pixel locations in the bottom quartile of the image, with an amplified
    version outlined by a red dotted line.

    :param input_image_path: Path to the input image
    :param lower_left_corner: Tuple (x, y) of the lower left corner pixel coordinates of the square region
    :param upper_right_corner: Tuple (x, y) of the upper right corner pixel coordinates of the square region

    Parameters
    ----------
    img : np.ndarray
        Input image in shape (H,W)
    ax : matplotlib.axes.Axes
        Axes to plot on
    roi_cfgs: list[ROIFeature]
        List of ROIFeature dataclasses for each ROI to plot

    Returns
    -------
    matplotlib.axes.Axes
        Axes with the image plotted
    """
    # Ensure image scaled [0,1]

    img = tonp(img)

    # show main image in grayscale, amplified region in cmap specified
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)

    if not isinstance(roi_cfgs, list):
        roi_cfgs = [roi_cfgs]

    for roi_cfg in roi_cfgs:

        # Calculate the region of interest
        roi = get_roi(img, roi_cfg)

        # add rectangle for original ROI
        ax = place_rectangle(ax, roi_cfg)

        # put amplified region on top of original image
        ax = place_roi_and_mask(ax, img, roi, roi_cfg)

    # turn of ticks
    ax.set_xticks([])
    ax.set_yticks([])

    return ax


def roi_grid_plot(img_list, roi_cfgs, img_titles=None, crops=(0, 0, 0, 0), transpose=True):
    """
    Given a list of N images and K configs for ROIs, make an N x (K+1) plot, showing
    each image with the respective ROIs outlined and amplified.
    """

    if not isinstance(img_list, list):
        img_list = [img_list]
    if not isinstance(roi_cfgs, list):
        roi_cfgs = [roi_cfgs]

    N = len(img_list)
    K = len(roi_cfgs)
    is_color = len(img_list[0].shape) == 3

    # increase font size for titles
    matplotlib.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(K + 1, N, figsize=(6 * N, 6 * (K + 1)))

    if transpose:
        tax = (1,0,2) if is_color else (1,0)
        img_list = [np.flip(np.transpose(img, tax), axis=0) for img in img_list]

    for n, img in enumerate(img_list):

        img = tonp(img)

        if is_color:
            ax[0, n].imshow(crop_img(img, crops))
        else:
            ax[0, n].imshow(crop_img(img, crops), cmap="gray", vmin=0, vmax=1)

        ax[0, n].axis("off")

        if img_titles:
            ax[0, n].set_title(img_titles[n])

        for k, roi_cfg in enumerate(roi_cfgs):

            # calculate the region of interest
            roi = get_roi(img, roi_cfg, return_img_size=True)

            # re-map locations of rectangle on cropped image
            if sum(crops) > 0:
                urc_crop, lrc_crop = remap_roi_locs_for_crop(img.shape, roi_cfg, crops)
                roi_cfg_rect = deepcopy(roi_cfg)
                roi_cfg_rect.upper_right_corner = urc_crop
                roi_cfg_rect.lower_left_corner = lrc_crop
            else:
                roi_cfg_rect = roi_cfg

            # add rectangle of roi on original image
            ax[0, n] = place_rectangle(ax[0, n], roi_cfg_rect)

            # plot this roi
            if is_color:
                ax[k + 1, n].imshow(roi)
            else:
                ax[k + 1, n].imshow(roi, cmap=roi_cfg.amp_cmap, vmin=0, vmax=1)

            # place a rectangle around this roi
            ox = roi_cfg.edge_placement_offset
            rect = patches.Rectangle(
                (ox, ox),
                roi.shape[1] - 2 * ox,
                roi.shape[0] - 2 * ox,
                linewidth=roi.shape[0] // 50,
                edgecolor=roi_cfg.rectangle_color,
                facecolor="none",
                linestyle="solid",
            )
            ax[k + 1, n].add_patch(rect)

            # cleanup by removing ticks
            ax[k + 1, n].set_xticks([])
            ax[k + 1, n].set_yticks([])

    fig.tight_layout()

    fig, ax = dark_mode(fig, ax)

    return fig, ax
