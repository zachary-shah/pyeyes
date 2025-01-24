from typing import Optional

import holoviews as hv
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

FULL_METRICS = [
    "L1Diff",
    "RMSE",
    "NRMSE",
    "PSNR",
    "SSIM",
]

MAPPABLE_METRICS = [
    "L1Diff",
    "L2Diff",
    "SSIM",
]

TOL = 1e-5


def L1Diff(recon: np.ndarray, true: np.ndarray, return_map=False) -> float:

    assert recon.shape == true.shape, "Input array dimensions mismatch."

    l1_diff = np.abs(recon - true)

    if return_map:
        l1_diff[l1_diff < TOL] = np.nan
        return l1_diff

    return np.mean(l1_diff)


def RMSE(recon: np.ndarray, true: np.ndarray, return_map=False) -> float:
    """
    alias for L2Diff

    Given 2 complex arrays of the same shape, recon (g) and true (f), compute the RMSE between arrays
    If normalized --> compute NRMSE, else compute RMSE
        RMSE = SQRT(|f-g|^2)
        NRMSE = SQRT(|f-g|^2 / |f|^2)
    """

    assert recon.shape == true.shape, "Input array dimensions mismatch."

    mse = np.abs(recon - true) ** 2

    if return_map:
        mse[mse < TOL] = np.nan
        return np.sqrt(mse)

    return np.sqrt(np.mean(mse))


def NRMSE(recon: np.ndarray, true: np.ndarray, return_map=False) -> float:
    """
    Given 2 complex arrays of the same shape, recon (g) and true (f), compute the RMSE between arrays
    If normalized --> compute NRMSE, else compute RMSE
        RMSE = SQRT(|f-g|^2)
        NRMSE = SQRT(|f-g|^2 / |f|^2)
    """

    assert recon.shape == true.shape, "Input array dimensions mismatch."

    nrmse = np.abs(recon - true) ** 2

    if return_map:
        nrmse /= np.sum(np.abs(true) ** 2)
        nrmse[nrmse < TOL] = np.nan
        return np.sqrt(nrmse)

    nrmse = np.mean(nrmse) / np.mean(np.abs(true) ** 2)
    return np.sqrt(nrmse)


def SSIM(recon: np.ndarray, true: np.ndarray, return_map=False) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two images.
    """

    assert recon.shape == true.shape, "Input array dimensions mismatch."

    data_range = np.abs(true).max() - np.abs(true).min()

    if return_map:
        ssim_map = compare_ssim(recon, true, full=True, data_range=data_range)[1]
        return ssim_map

    return compare_ssim(recon, true, data_range=data_range)


def PSNR(recon: np.ndarray, true: np.ndarray) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.
    """

    max_val = np.abs(true).max()

    assert recon.shape == true.shape, "Input array dimensions mismatch."

    mse = RMSE(recon, true)

    psnr = 20 * np.log10(max_val / mse)

    return psnr


METRIC_CALLABLES = {
    "L1Diff": L1Diff,
    "L2Diff": RMSE,
    "RMSE": RMSE,
    "NRMSE": NRMSE,
    "PSNR": PSNR,
    "SSIM": SSIM,
}
