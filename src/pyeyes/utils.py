from typing import Dict, List, Optional, Sequence, Union

import holoviews as hv
import numpy as np
import panel as pn

TORCH_IMPORTED = False
try:
    import torch

    TORCH_IMPORTED = True
except ImportError:
    print(
        "Pyeyes Warning: torch install could not be found. Pyeyes may not be able to infer tensor types."
    )

from .enums import ROI_LOCATION


def rescale01(image):
    return (image - image.min()) / (image.max() - image.min())


def tonp(x: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """
    Convert to numpy array
    """
    if TORCH_IMPORTED:
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
    else:
        if isinstance(x, (list, tuple)):
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
    if flip_lr and flip_ud:
        if loc == ROI_LOCATION.TOP_LEFT:
            effective_loc = ROI_LOCATION.BOTTOM_RIGHT
        elif loc == ROI_LOCATION.TOP_RIGHT:
            effective_loc = ROI_LOCATION.BOTTOM_LEFT
        elif loc == ROI_LOCATION.BOTTOM_LEFT:
            effective_loc = ROI_LOCATION.TOP_RIGHT
        elif loc == ROI_LOCATION.BOTTOM_RIGHT:
            effective_loc = ROI_LOCATION.TOP_LEFT
    elif flip_lr:
        if loc == ROI_LOCATION.TOP_LEFT:
            effective_loc = ROI_LOCATION.TOP_RIGHT
        elif loc == ROI_LOCATION.TOP_RIGHT:
            effective_loc = ROI_LOCATION.TOP_LEFT
        elif loc == ROI_LOCATION.BOTTOM_LEFT:
            effective_loc = ROI_LOCATION.BOTTOM_RIGHT
        elif loc == ROI_LOCATION.BOTTOM_RIGHT:
            effective_loc = ROI_LOCATION.BOTTOM_LEFT
    elif flip_ud:
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


def round_str(x: float, ndec: int = 1) -> str:
    """
    Return string of x rounded to ndec decimal places.
    """
    MAXR = 30
    assert 0 < ndec < MAXR, f"ndec must be between 0 and {MAXR}."

    if ndec == 0:
        return str(int(round(x)))

    x = float(round(x, ndec))

    p = 10**ndec
    xi = int(x * p)

    nz = 0
    while ((xi % 10) == 0) and (nz < ndec):
        nz += 1
        xi = xi // 10

    if nz == ndec:
        return str(int(x))

    nrem_div = 10 ** (ndec - nz)

    return f"{float(float(xi) / nrem_div):.{(ndec - nz)}f}"


def pprint_str(x: float, D: int = 5, E: int = 1) -> str:
    """
    Pretty print a string representation of a number.

    Rules:
    If the number is an integer, return the integer as a string.
    If the number is <1e-(D-1) or >1e(D-1), return the number in scientific notation with D-1 decimal places.
    Otherwise, return the number in float notation with exactly D digits max (including integer component).
    """

    if isinstance(x, int):
        return str(x)

    if D == 0:
        return str(int(round(x)))

    if not (np.isfinite(x)):
        return "NaN"

    # approx integer
    is_int = False
    if (x > 1e-8) and np.isclose(round(x) - x, 0):
        x = int(round(x))
        is_int = True

    Dact = max(D - 2, 1)

    small_tol = 10 ** (-(Dact))
    big_tol = 10 ** ((Dact))
    if abs(x) < small_tol or abs(x) > big_tol:
        if is_int:
            return f"{x:0.0e}"
        else:
            return f"{x:0.{E}e}"

    # determine number of digits after decimal point
    if is_int:
        return str(x)
    else:
        ndec = D - len(str(int(x)))
        return f"{x:.{ndec}f}"


def parse_dimensional_input(
    input: Optional[Union[Sequence[str], str]], N: int
) -> List[str]:
    """
    Parse dimensional input into a list of strings.
    """
    VALID_DELIMTERS = [",", ";", "-", "_"]

    if isinstance(input, str):
        # Case where input is a string of N characters
        if len(input) == N:
            return list(input)

        # input is character delimited
        for delim in VALID_DELIMTERS:
            if delim in input:
                inp_list = input.split(delim)
                inp_list = [i.replace(" ", "") for i in inp_list]
                assert (
                    len(inp_list) == N
                ), f"Number of specified dimensions must match number of dimensions in data (N={N})."
                return inp_list

        # input is space delimited
        if " " in input:
            inp_list = input.split(" ")
            inp_list = [i.replace(" ", "") for i in inp_list]
            assert (
                len(inp_list) == N
            ), f"Number of specified dimensions must match number of dimensions in data (N={N})."
            return inp_list

        raise ValueError(f"Invalid dimensional input: {input}")
    elif isinstance(input, Sequence):
        input = [str(s) for s in input]
        assert (
            len(input) == N
        ), f"Number of specified dimensions must match number of dimensions in data (N={N})."
        return input

    elif input is not None:
        raise ValueError(f"Invalid dimensional input: {input}")

    # Default case
    return [f"Dim {i}" for i in range(N)]


def sanitize_css_class(name: str) -> str:
    """
    Make a string safe for use as a CSS class.
    """
    BAD_CHARS = [
        " ",
        ".",
        ":",
        ";",
        "<",
        ">",
        "[",
        "]",
        "{",
        "}",
        "|",
        "\\",
        "/",
        "?",
        "!",
        "@",
        "#",
        "$",
        "%",
        "^",
        "&",
        "*",
        "(",
        ")",
        "=",
        "+",
        "~",
        "`",
        "'",
        '"',
    ]

    for badchar in BAD_CHARS:
        name = name.replace(badchar, "-")

    return name


# Complex view mapping
CPLX_VIEW_MAP = {
    "mag": np.abs,
    "phase": masked_angle,
    "real": np.real,
    "imag": np.imag,
}
