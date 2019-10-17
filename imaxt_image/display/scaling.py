from typing import Tuple

import numpy as np
from astropy.visualization import PercentileInterval, ZScaleInterval


def zscale(img: np.ndarray) -> Tuple[float, float]:
    """Determine zscale range.

    Parameters
    ----------
    img
        image array

    Returns
    -------
    zscale range
    """
    vmin, vmax = ZScaleInterval(krej=10).get_limits(img.ravel())
    return vmin, vmax


def percentile(img: np.ndarray, percentile: int) -> Tuple[float, float]:
    """Determine percentile range.

    Calculates the range (vmin, vmax) so that a percentile
    of the pixels is within those values.

    Parameters
    ----------
    img
        image array
    percentile
        Percentile value

    Returns
    -------
    percentile range
    """
    p = PercentileInterval(percentile)
    vmin, vmax = p.get_limits(img.ravel())
    return vmin, vmax
