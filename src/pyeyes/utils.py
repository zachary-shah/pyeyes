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
    """Rescale array to [0, 1] using min/max."""
    return (image - image.min()) / (image.max() - image.min())


def tonp(x: Union[np.ndarray, list, tuple]) -> np.ndarray:
    """
    Convert tensor, list, or tuple to numpy array; pass-through for ndarray.

    Parameters
    ----------
    x : np.ndarray, list, or tuple
        Input to convert.

    Returns
    -------
    np.ndarray
        Numpy array (complex tensors get resolve_conj on CPU).
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
    Fit shifted = a * target + b (optionally magnitude-only) and return corrected data.

    Parameters
    ----------
    shifted : array-like
        Data to correct.
    target : array-like
        Reference data.
    ofs : bool
        Include offset b in fit.
    mag : bool
        Fit on magnitudes only; phase preserved.
    eps : float
        Tolerance for lstsq.

    Returns
    -------
    np.ndarray
        a * shifted + b (or magnitude-scaled complex).
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


def normalize_scale(shifted, target, eps=1e-10, scale_tol=1e-5):
    """
    Normalize shifted to target by a single scale factor (no offset); fallback to lstsq if small.

    Parameters
    ----------
    shifted, target : array-like
        Images to align (scale only).
    eps : float
        Regularization for scale denominator.
    scale_tol : float
        Below this max|shifted|, use normalize() instead.
    """
    # If input scale is too small, then fall back to lstsq normalization computation for robustness
    inp_scale = np.max(np.abs(shifted))
    if inp_scale < scale_tol:
        return normalize(shifted, target, ofs=False, mag=np.iscomplexobj(shifted))

    a = np.sum(np.abs(shifted) * np.abs(target)) / (
        (np.linalg.norm(shifted) ** 2) + eps
    )

    return shifted * a


def get_effective_location(
    loc: ROI_LOCATION, flip_lr: bool, flip_ud: bool
) -> ROI_LOCATION:
    """
    Map ROI location to effective corner after L/R and U/D flips.

    Parameters
    ----------
    loc : ROI_LOCATION
        Nominal corner (e.g. TOP_LEFT).
    flip_lr, flip_ud : bool
        Whether plot is flipped.

    Returns
    -------
    ROI_LOCATION
        Effective corner in displayed coordinates.
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
    Clone an hv.Dataset, replacing its "Value" data with new_value.

    Parameters
    ----------
    original_dataset : hv.Dataset
        Dataset to clone.
    new_value : np.ndarray
        New "Value" array (same shape as original).
    link : bool
        If True, link to original (default False).

    Returns
    -------
    hv.Dataset
        New dataset with same kdims/vdims and new Value.
    """
    assert new_value.shape == original_dataset.data["Value"].shape

    ddims = [k for k in list(original_dataset.data.keys()) if k != "Value"]
    new_data_dict = {k: original_dataset.data[k] for k in ddims}
    new_data_dict["Value"] = new_value

    return original_dataset.clone(data=new_data_dict, link=link)


def debug_imshow(data_dict: Dict[str, hv.Dataset]):
    """
    Display dict of hv.Datasets in a shared-axis grid (debugging).

    Parameters
    ----------
    data_dict : dict of hv.Dataset
        Keys = labels; each dataset has 2D kdims and Value vdim.
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
    """Return angle of x where magnitude >= thresh * max; elsewhere 0."""
    mask = np.abs(x) / np.max(np.abs(x)) < thresh
    x = np.angle(x)
    x[mask] = 0
    return x


def round_str(x: float, ndec: int = 1) -> str:
    """
    Format x rounded to ndec decimal places, trimming trailing zeros.

    Parameters
    ----------
    x : float
        Value to format.
    ndec : int
        Max decimal places (0 < ndec < 30).

    Returns
    -------
    str
        Rounded string (e.g. "1.2", "3").
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
    Format number: int as-is; very small/large in scientific (E decimals); else float (D digits).

    Parameters
    ----------
    x : float
        Value to format.
    D : int
        Max total digits for float; exponent threshold ~ 10^-(D-1) to 10^(D-1).
    E : int
        Decimal places in scientific notation.

    Returns
    -------
    str
        Formatted string.
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
    Parse dimension names from string or sequence into list of N strings.

    Parameters
    ----------
    input : str, sequence of str, or None
        N chars, or N delimited tokens (e.g. "x,y,z"), or sequence of length N.
    N : int
        Expected number of dimensions.

    Returns
    -------
    list of str
        Dimension names; default ["Dim 0", ..., "Dim N-1"] if input is None.
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
    """Replace characters invalid in CSS class names with '-'."""
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
