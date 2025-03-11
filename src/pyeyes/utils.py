from typing import Dict, Union

import holoviews as hv
import numpy as np
import panel as pn
import torch

from .enums import ROI_LOCATION


def rescale01(image):
    return (image - image.min()) / (image.max() - image.min())


def tonp(x: Union[np.ndarray, torch.tensor]):
    """
    Convert to numpy array
    """
    if torch.is_tensor(x):
        # resolve conj if needed
        if x.is_complex():
            x = x.detach().cpu().resolve_conj().numpy()
        else:
            x = x.detach().cpu().numpy()
        return x
    elif isinstance(x, (list, tuple)):
        return np.array(x)
    else:
        return x


def normalize(shifted, target, ofs=True, mag=False, eps=1e-12):
    """
    Assumes the following scaling/shifting offset:

    shifted = a * target + b

    solves for a, b and returns the corrected data

    Parameters:
    -----------
    shifted : array
        data to be corrected
    target : array
        reference data
    ofs : bool
        include b offset in the correction
    mag : bool
        use magnitude of data for correction

    Returns:
    --------
    array
        corrected data

    TODO: batch function so it is faster
    """

    shifted = tonp(shifted)
    target = tonp(target)

    try:
        # scale
        scale_func = np.abs if mag else lambda x: x
        x = scale_func(shifted).flatten()
        y = scale_func(target).flatten()

        # fit
        if ofs:
            A = np.vstack([x, np.ones_like(np.abs(x))]).T
            a, b = np.linalg.lstsq(A, y, rcond=eps)[0]
        else:
            A = np.array([x]).T
            a = np.linalg.lstsq(A, y, rcond=eps)[0]
            b = 0

        # numerical tolerance
        n_decimal = int(np.round(-np.log10(eps)))
        a, b = np.round(a, n_decimal), np.round(b, n_decimal)

        # apply
        if mag:
            out = (a * np.abs(shifted) + b) * np.exp(1j * np.angle(shifted))
        else:
            out = a * shifted + b

    except:
        print("Error in normalization. Returning original data.")
        out = shifted

    return out


def get_effective_location(
    loc: ROI_LOCATION, flip_lr: bool, flip_ud: bool
) -> ROI_LOCATION:
    """
    Given plot window flips, determine the real location of the plot.
    """

    effective_loc = loc
    if flip_lr:
        if loc == ROI_LOCATION.TOP_LEFT:
            effective_loc = ROI_LOCATION.TOP_RIGHT
        elif loc == ROI_LOCATION.TOP_RIGHT:
            effective_loc = ROI_LOCATION.TOP_LEFT
        elif loc == ROI_LOCATION.BOTTOM_LEFT:
            effective_loc = ROI_LOCATION.BOTTOM_RIGHT
        elif loc == ROI_LOCATION.BOTTOM_RIGHT:
            effective_loc = ROI_LOCATION.BOTTOM_LEFT
    if flip_ud:
        if loc == ROI_LOCATION.TOP_LEFT:
            effective_loc = ROI_LOCATION.BOTTOM_LEFT
        elif loc == ROI_LOCATION.TOP_RIGHT:
            effective_loc = ROI_LOCATION.BOTTOM_RIGHT
        elif loc == ROI_LOCATION.BOTTOM_LEFT:
            effective_loc = ROI_LOCATION.TOP_LEFT
        elif loc == ROI_LOCATION.BOTTOM_RIGHT:
            effective_loc = ROI_LOCATION.TOP_RIGHT
    return effective_loc


def clone_dataset(
    original_dataset: hv.Dataset,
    new_value: np.ndarray,
    link=False,
):
    """
    Clones an existing hv.Dataset, replacing its data with new_data.

    Parameters:
    - original_dataset (hv.Dataset): The original dataset to clone.
    - new_value (np.ndarray or similar): The new "Value" data to replace the original data.
    - link: Link streams / pipes to original dataset (probably don't want this).

    Returns:
    - hv.Dataset: A new dataset with the same properties as original_dataset but with new_data.
    """

    assert new_value.shape == original_dataset.data["Value"].shape

    ddims = [k for k in list(original_dataset.data.keys()) if k != "Value"]
    new_data_dict = {k: original_dataset.data[k] for k in ddims}
    new_data_dict["Value"] = new_value

    return original_dataset.clone(data=new_data_dict, link=link)


def debug_imshow(data_dict: Dict[str, hv.Dataset]):
    """
    Given a dictionary of hv.Datasets with Value as vdim,, display them in a grid for debugging purposes.
    """

    img_keys = list(data_dict.keys())
    vdims = list(data_dict[img_keys[0]].vdims)
    kdims_all = list(data_dict[img_keys[0]].kdims)[:2]

    # Create a grid of images
    layout = []

    # make a holoview
    for key in img_keys:
        layout.append(
            hv.Image(
                data_dict[key],
                label=key,
                vdims=vdims,
                kdims=kdims_all,
            )
        )

    layout = hv.Layout(layout).opts(shared_axes=True)

    # show
    pn.serve(pn.Column(layout), show=True)


def masked_angle(x, thresh=1e-2):
    """
    Display angle of any array where magnitude is above threshold
    """
    mask = np.abs(x) / np.max(np.abs(x)) < thresh
    x = np.angle(x)
    x[mask] = 0
    return x


# Complex view mapping
CPLX_VIEW_MAP = {
    "mag": np.abs,
    "phase": masked_angle,
    "real": np.real,
    "imag": np.imag,
}
