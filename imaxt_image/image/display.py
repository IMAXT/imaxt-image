"""Utility functions to display images."""

import math

import matplotlib.pyplot as plt
import numpy as np

from imaxt_image.image.scaling import zscale


def imshow(img: np.ndarray):
    """Display an image.

    Parameters
    ----------
    img
        Image array
    """
    vmin, vmax = zscale(img)
    plt.imshow(img, vmin=vmin, vmax=vmax, origin='lower')
    plt.axis('off')


def multishow(img: np.ndarray, orientation: str = 'portrait'):
    """Display all channels from a slice.

    Parameters
    ----------
    img
        Image array
    orientation
        Orientation of sublots
    """
    n = img.shape[0]
    n_rows = math.floor(math.sqrt(n))
    n_cols = math.ceil(math.sqrt(n))
    if orientation == 'landscape':
        n_rows, n_cols = n_cols, n_rows
    for c in range(n):
        plt.subplot(n_cols * 100 + n_rows * 10 + c + 1)
        imshow(img[c])
    plt.tight_layout()
