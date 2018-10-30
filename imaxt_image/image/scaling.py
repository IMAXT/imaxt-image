"""Utility functions to scale images."""

from typing import Tuple

import numpy as np
from astropy.visualization import PercentileInterval, ZScaleInterval


def zscale(img: np.ndarray) -> Tuple[float, float]:
    """Determine zscale range.

    Parameters
    ----------
    img
        Image array

    Returns
    -------
    float, float
        Tuple containing the zscale range
    """
    vmin, vmax = ZScaleInterval(krej=10).get_limits(img)
    return vmin, vmax


def percentile(img: np.ndarray, percentile: int) -> Tuple[float, float]:
    """Determine percentile range.

    Calculates the range (vmin, vmax) so that a percentile
    of the pixels is within those values.

    Parameters
    ----------
    img
        Image array
    percentile
        Percentile value

    Returns
    -------
    float, float
        Tuple containing the percentile range
    """
    p = PercentileInterval(percentile)
    vmin, vmax = p.get_limits(img)
    return vmin, vmax
