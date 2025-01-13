import os

import numpy as np


def relaxation_color_map(maptype, loLev, upLev):
    """
    Acts in two ways:
    1. Generates a colormap to be used on display, given the image type.
    2. Generates a 'clipped' image, where values are clipped to the lower level.
    """
    # Load the colormap depending on the map type
    maptype = maptype.capitalize()

    current_dir = os.path.dirname(os.path.abspath(__file__))

    if maptype in ["T1", "R1"]:
        file_path = os.path.join(current_dir, "lipari.csv")
        colortable = np.genfromtxt(file_path, delimiter=" ")
    elif maptype in ["T2", "T2*", "R2", "R2*", "T1rho", "T1ρ", "R1rho", "R1ρ"]:
        file_path = os.path.join(current_dir, "navia.csv")
        colortable = np.genfromtxt(file_path, delimiter=" ")
    else:
        raise ValueError("Expect 'T1', 'T2', 'R1', or 'R2' as maptype")

    # Flip colortable for R1 or R2 types
    if maptype[0] == "R":
        colortable = np.flipud(colortable)

    # Set the 'invalid value' color
    colortable[0, :] = 0.0

    # Modification of the image to be displayed
    eps = (upLev - loLev) / colortable.shape[0]

    def clip_for_qmap(x):
        return np.where(
            x < eps, loLev - eps, np.where(x < loLev + eps, loLev + 1.5 * eps, x)
        )

    # Apply color remapping
    lut_cmap = color_log_remap(colortable, loLev, upLev)

    return lut_cmap, clip_for_qmap


def color_log_remap(ori_cmap, loLev, upLev):
    """
    Lookup of the original color map table according to a 'log-like' curve.
    """
    assert upLev > 0, "Upper level must be positive"
    assert upLev > loLev, "Upper level must be larger than lower level"

    map_length = ori_cmap.shape[0]
    e_inv = np.exp(-1.0)
    a_val = e_inv * upLev
    m_val = max(a_val, loLev)
    b_val = (1.0 / map_length) + (a_val >= loLev) * (
        (a_val - loLev) / (2 * a_val - loLev)
    )
    b_val += 1e-7  # Ensure rounding precision

    log_cmap = np.zeros_like(ori_cmap)
    log_cmap[0, :] = ori_cmap[0, :]

    log_portion = 1.0 / (np.log(m_val) - np.log(upLev))

    for g in range(1, map_length):
        x = g * (upLev - loLev) / map_length + loLev
        if x > m_val:
            f = map_length * (
                (np.log(m_val) - np.log(x)) * log_portion * (1 - b_val) + b_val
            )
        elif loLev < a_val and x > loLev:
            f = (
                map_length
                * ((x - loLev) / (a_val - loLev) * (b_val - (1.0 / map_length)))
                + 1.0
            )
        else:
            f = 1.0 if x <= loLev else f

        log_cmap[g, :] = ori_cmap[min(map_length - 1, int(np.floor(f))), :]

    return log_cmap
