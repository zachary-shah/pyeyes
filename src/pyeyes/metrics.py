import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

FULL_METRICS = [
    "RelativeL1",
    "L1Diff",
    "RMSE",
    "NRMSE",
    "PSNR",
    "SSIM",
]

MAPPABLE_METRICS = [
    "RelativeL1",
    "L1Diff",
    "L2Diff",
    "SSIM",
    "Diff",
]

# small tol
TOL = 1e-14
ERROR_TOL = 1e-12
SCALE_TOL = 1e-8


def diff(recon: np.ndarray, true: np.ndarray, return_map=False, isphase=False) -> float:
    """Mean difference (or diff map) between recon and true; phase-aware if isphase."""
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
    """L1 difference (mean or map); phase-aware if isphase."""
    assert recon.shape == true.shape, "Input array dimensions mismatch."

    diff_map = diff(recon, true, return_map=True, isphase=isphase)

    l1_diff = np.abs(diff_map)

    if return_map:
        # prevent returning all nans
        if np.max(l1_diff) < TOL:
            return np.zeros_like(l1_diff)

        im_mask = (np.abs(true) - np.min(np.abs(true))) < SCALE_TOL * np.max(
            np.abs(true)
        )

        l1_diff[(l1_diff < TOL) & im_mask] = np.nan
        return l1_diff

    return np.mean(l1_diff)


def RelativeL1(
    recon: np.ndarray,
    true: np.ndarray,
    return_map=False,
    isphase=False,
) -> float:
    """Relative L1 (|recon-true|/|true|); mean or map; phase-aware if isphase."""
    assert recon.shape == true.shape, "Input array dimensions mismatch."

    diff_map = diff(recon, true, return_map=True, isphase=isphase)
    mx = np.max(np.abs(true))
    rel_diff = np.abs(diff_map) / (np.abs(true) + TOL * mx)
    valid = np.abs(true) > TOL * mx

    if return_map:
        if np.sum(valid) == 0:
            return np.zeros_like(rel_diff)

        rel_diff[~valid] = np.nan

        return rel_diff

    if np.sum(valid) == 0:
        return 0.0

    return np.mean(rel_diff[valid])


def RMSE(recon: np.ndarray, true: np.ndarray, return_map=False, isphase=False) -> float:
    """RMSE = sqrt(mean(|recon-true|^2)); return scalar or map; phase-aware if isphase."""
    assert recon.shape == true.shape, "Input array dimensions mismatch."

    diff_map = diff(recon, true, return_map=True, isphase=isphase)

    mse = np.abs(diff_map) ** 2

    if return_map:
        mse = np.sqrt(mse)

        # prevent returning all nans
        if np.max(mse) < TOL:
            return np.zeros_like(mse)

        mse[mse < TOL] = np.nan
        return mse

    return np.sqrt(np.mean(mse))


def NRMSE(
    recon: np.ndarray, true: np.ndarray, return_map=False, isphase=False
) -> float:
    """NRMSE = sqrt(mean(|recon-true|^2) / mean(|true|^2)); scalar or map; phase-aware if isphase."""
    assert recon.shape == true.shape, "Input array dimensions mismatch."

    diff_map = diff(recon, true, return_map=True, isphase=isphase)

    mse = np.abs(diff_map) ** 2

    if return_map:
        if np.max(np.abs(true)) < TOL:
            return np.zeros_like(mse)
        nrmse = mse / ((np.abs(true) ** 2) + (TOL**2))
        nrmse = np.sqrt(nrmse)
        nrmse[nrmse < TOL] = np.nan
        nrmse[nrmse > (1 / TOL)] = np.nan
        if np.nanmax(nrmse) < TOL:
            return np.zeros_like(nrmse)
        return nrmse

    nrmse = np.mean(mse) / (np.mean(np.abs(true) ** 2) + (TOL**2))
    return np.sqrt(nrmse)


def SSIM(
    recon: np.ndarray,
    true: np.ndarray,
    return_map=False,
    isphase=False,
    percentiles=[0.5, 99.5],
) -> float:
    """Structural similarity index (skimage); data_range from percentiles of |true|."""
    assert recon.shape == true.shape, "Input array dimensions mismatch."

    if isphase:
        pass  # do nothing different, but SSIM doesn't really make sense for phase maps

    true_abs = np.abs(true)
    upper_bound = np.percentile(true_abs, percentiles[1])
    lower_bound = np.percentile(true_abs, percentiles[0])
    data_range = upper_bound - lower_bound

    # all zeros case
    if data_range < TOL:
        if return_map:
            return np.ones_like(recon)
        else:
            return 1.0

    if return_map:
        ssim_map = compare_ssim(recon, true, full=True, data_range=data_range)[1]
        return ssim_map

    return compare_ssim(recon, true, data_range=data_range)


def PSNR(
    recon: np.ndarray, true: np.ndarray, isphase=False, max_percentile=99.5
) -> float:
    """PSNR in dB using max_percentile of |true| as peak; RMSE for error."""
    assert recon.shape == true.shape, "Input array dimensions mismatch."

    max_val = np.percentile(np.abs(true), max_percentile)

    if max_val < TOL:
        return np.inf

    rmse = RMSE(recon, true, isphase=isphase)

    if rmse < TOL:
        return np.inf

    psnr = 20 * np.log10(max_val / rmse)

    return psnr


METRIC_CALLABLES = {
    "RelativeL1": RelativeL1,
    "L1Diff": L1Diff,
    "L2Diff": RMSE,
    "RMSE": RMSE,
    "NRMSE": NRMSE,
    "PSNR": PSNR,
    "SSIM": SSIM,
    "Diff": diff,
}
