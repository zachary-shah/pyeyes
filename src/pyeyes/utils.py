import numpy as np
import torch
from typing import Union

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
    
def RMSE(recon: np.ndarray, 
         true: np.ndarray,
         normalized: float = True) -> float:
    """
    Given 2 complex arrays of the same shape, recon (g) and true (f), compute the RMSE between arrays 
    If normalized --> compute NRMSE, else compute RMSE
        RMSE = SQRT(|f-g|^2)
        NRMSE = SQRT(|f-g|^2 / |f|^2)
    """

    assert recon.shape == true.shape, "Input array dimensions mismatch."

    mse = np.mean(np.abs(recon - true)**2)
    if normalized: mse /= np.mean(np.abs(true)**2)
    return np.sqrt(mse)


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