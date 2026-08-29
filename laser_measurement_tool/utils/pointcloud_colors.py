"""Shared colors for ground-coordinate point-cloud visualization and export."""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from numpy.typing import ArrayLike


ZG_HIGH_CONTRAST_CMAP = LinearSegmentedColormap.from_list(
    "zg_high_contrast",
    (
        (0.00, "#00E5FF"),
        (0.20, "#00FF66"),
        (0.45, "#E8FF00"),
        (0.65, "#FFD000"),
        (0.82, "#FF5A00"),
        (1.00, "#FF00A8"),
    ),
    N=256,
)


def map_zg_to_rgb(zg_values: ArrayLike) -> tuple[np.ndarray, float, float]:
    """Map Zg values to bright RGB colors suitable for a dark background."""
    zg = np.asarray(zg_values, dtype=np.float64).reshape(-1)
    zg_min = float(zg.min()) if len(zg) else 0.0
    zg_max = float(zg.max()) if len(zg) else 0.0
    if zg_max > zg_min:
        normalized_zg = Normalize(vmin=zg_min, vmax=zg_max, clip=True)(zg)
    else:
        normalized_zg = np.full(len(zg), 0.5, dtype=np.float64)
    rgb = np.rint(
        ZG_HIGH_CONTRAST_CMAP(normalized_zg)[:, :3] * 255.0
    ).astype(np.uint8)
    return rgb, zg_min, zg_max
