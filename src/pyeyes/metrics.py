import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

FULL_METRICS = [
    "L1Diff",
    "RMSE",
    "NRMSE",
    "PSNR",
    "SSIM",
    "RelativeL1",
]

MAPPABLE_METRICS = [
    "L1Diff",
    "L2Diff",
    "SSIM",
    "RelativeL1",
    "Diff",
]

TOL = 1e-5


def diff(recon: np.ndarray, true: np.ndarray, return_map=False, isphase=False) -> float:

    assert recon.shape == true.shape, "Input array dimensions mismatch."

    if isphase:
        diff_map = np.angle(np.exp(1j * recon) / np.exp(1j * true))
    else:
        diff_map = recon - true

    if return_map:
        return diff_map

    return np.mean(diff_map)


def L1Diff(
    recon: np.ndarray, true: np.ndarray, return_map=False, isphase=False
) -> float:

    assert recon.shape == true.shape, "Input array dimensions mismatch."

    diff_map = diff(recon, true, return_map=True, isphase=isphase)

    l1_diff = np.abs(diff_map)

    if return_map:
        l1_diff[l1_diff < TOL] = np.nan
        return l1_diff

    return np.mean(l1_diff)


def RelativeL1(
    recon: np.ndarray, true: np.ndarray, return_map=False, isphase=False
) -> float:

    assert recon.shape == true.shape, "Input array dimensions mismatch."

    diff_map = diff(recon, true, return_map=True, isphase=isphase)

    rel_diff = np.abs(diff_map) / (np.abs(true) + 1e-9)

    if return_map:
        return rel_diff

    valid = np.abs(true) > TOL

    return np.mean(rel_diff[valid])


def RMSE(recon: np.ndarray, true: np.ndarray, return_map=False, isphase=False) -> float:
    """
    alias for L2Diff

    Given 2 complex arrays of the same shape, recon (g) and true (f), compute the RMSE between arrays
        RMSE = SQRT(|f-g|^2)
        NRMSE = SQRT(|f-g|^2 / |f|^2)
    """

    assert recon.shape == true.shape, "Input array dimensions mismatch."

    diff_map = diff(recon, true, return_map=True, isphase=isphase)

    mse = np.abs(diff_map) ** 2

    if return_map:
        mse = np.sqrt(mse)
        mse[mse < TOL] = np.nan
        return mse

    return np.sqrt(np.mean(mse))


def NRMSE(
    recon: np.ndarray, true: np.ndarray, return_map=False, isphase=False, eps=1e-8
) -> float:
    """
    Given 2 complex arrays of the same shape, recon (g) and true (f), compute the NRMSE between arrays
        NRMSE = SQRT(|f-g|^2 / |f|^2)
    """

    assert recon.shape == true.shape, "Input array dimensions mismatch."

    diff_map = diff(recon, true, return_map=True, isphase=isphase)

    mse = np.abs(diff_map) ** 2

    if return_map:
        nrmse = mse / ((np.abs(true) ** 2) + eps)
        nrmse[nrmse < eps] = np.nan
        nrmse[nrmse > (1 / eps)] = np.nan
        return np.sqrt(nrmse)

    nrmse = np.mean(mse) / np.mean(np.abs(true) ** 2)
    return np.sqrt(nrmse)


def SSIM(
    recon: np.ndarray,
    true: np.ndarray,
    return_map=False,
    isphase=False,
    percentiles=[0.5, 99.5],
) -> float:
    """
    Compute the Structural Similarity Index (SSIM) between two images.
    """

    assert recon.shape == true.shape, "Input array dimensions mismatch."

    if isphase:
        pass  # do nothing different, but SSIM doesn't really make sense for phase maps

    true_abs = np.abs(true)
    upper_bound = np.percentile(true_abs, percentiles[1])
    lower_bound = np.percentile(true_abs, percentiles[0])
    data_range = upper_bound - lower_bound

    if return_map:
        ssim_map = compare_ssim(recon, true, full=True, data_range=data_range)[1]
        return ssim_map

    return compare_ssim(recon, true, data_range=data_range)


def PSNR(
    recon: np.ndarray, true: np.ndarray, isphase=False, max_percentile=99.5
) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.
    """

    assert recon.shape == true.shape, "Input array dimensions mismatch."

    max_val = np.percentile(np.abs(true), max_percentile)

    mse = RMSE(recon, true, isphase=isphase)

    psnr = 20 * np.log10(max_val / mse)

    return psnr


METRIC_CALLABLES = {
    "L1Diff": L1Diff,
    "L2Diff": RMSE,
    "RMSE": RMSE,
    "NRMSE": NRMSE,
    "PSNR": PSNR,
    "SSIM": SSIM,
    "RelativeL1": RelativeL1,
    "Diff": diff,
}
