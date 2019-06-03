import warnings
from typing import Dict, Tuple

import numpy as np
import photutils
from scipy.ndimage import fourier_gaussian, shift


def enhanced_correlation(
    im0: np.ndarray, im1: np.ndarray, sigma: int = 5
) -> np.ndarray:
    r"""Calculate the cross correlation between two images.

    Given the two images $S_0$ and $S_1$ and their Fourier
    transforms $F_0$ and $F_1$ we define the enhanced cross
    correlation as

    .. math::

        P = \frac{\Psi_{10}}{\sqrt{\Psi_{00} * \Psi_{11}}}

    where $Psi_{ij} = F_i \times F_j^*$.

    The images and the cross correlation are convolved with
    a gaussian of standard deviation ``sigma``.

    Parameters
    ----------
    im0
        Reference image
    im1
        Image to register. Must be same dimensionality as ``im0``.
    sigma
        Sigma of the gaussian to convolve the images with.

    Returns
    -------
    cross correlation array
    """
    assert im0.shape == im1.shape

    src_image = np.array(im0, dtype=np.complex128, copy=False)
    target_image = np.array(im1, dtype=np.complex128, copy=False)

    # Fourier transform
    F_0 = np.fft.fftn(src_image)
    F_1 = np.fft.fftn(target_image)

    # Convolution with gaussian
    F_0 = fourier_gaussian(F_0, sigma)
    F_1 = fourier_gaussian(F_1, sigma)

    # Cross and auto-correlation
    # phi_10 = F_1 * F_0.conj()
    phi_01 = F_0 * F_1.conj()
    phi_00 = F_0 * F_0.conj()
    phi_11 = F_1 * F_1.conj()

    # Enhanced correlation
    P = phi_01 / (np.sqrt(phi_00 * phi_11) + 1e-10)
    P = fourier_gaussian(P, 5)
    enhanced_correlation = np.fft.ifftn(P)
    return enhanced_correlation


def find_maximum(im: np.ndarray, maxpeaks: int = 3, border_width: int = 20) -> Tuple:
    """Find maximum peaks in image.

    Parameters
    ----------
    im
        Image
    maxpeaks
        Number of peaks to return (brightest)

    Returns
    -------
    x and y locations of peaks
    """
    assert maxpeaks > 0
    assert border_width > 0
    mean, std = im.mean(), im.std()
    xt = yt = [None]
    for sigma in [10, 5, 3]:
        threshold = mean + sigma * std
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            peaks_tbl = photutils.find_peaks(
                im, threshold=threshold, box_size=30, border_width=border_width
            )
        if len(peaks_tbl) == 0:
            continue
        x, y, peak = peaks_tbl['x_peak'], peaks_tbl['y_peak'], peaks_tbl['peak_value']
        indx = np.argsort(peak)[-maxpeaks:]
        xt, yt = x[indx], y[indx]
        break
    if None in x:
        im0 = np.abs(im)
        if border_width > 0:
            im0[:border_width, :] = 0
            im0[-border_width:, :] = 0
            im0[:, :border_width] = 0
            im0[:, -border_width:] = 0
        y, x = np.unravel_index(np.argmax(im0), im.shape)
        xt, yt = [x], [y]
    return xt, yt


def find_shift(
    im0: np.ndarray,
    im1: np.ndarray,
    sigma: int = 5,
    overlap: Tuple[float] = (0.08, 0.12),
    full: bool = True,
    offset: bool = False,
) -> Dict[str, int]:
    """Find shift between images using cross correlation.

    Parameters
    ----------
    im0
        Reference image
    im1
        Target image
    sigma
        Standard deviation of the convolution gaussian kernel.
    overlap
        Image overlap range to exclude possible offsets
    full
        Return full dictionary of results vs offsets list
    offset
        Shift images around origin. Useful if expected shifts are close to zero.

    Returns
    -------
    Dictionary containing x, y and overlap area in pixels.

    References
    ----------
    See: http://www.sci.utah.edu/publications/SCITechReports/UUSCI-2006-020.pdf
    """
    assert overlap[0] < overlap[1]
    ysize, xsize = im0.shape
    xcorrelation = enhanced_correlation(im0, im1, sigma)
    if offset:
        xcorrelation = np.fft.fftshift(xcorrelation)

    im = xcorrelation.real
    xt, yt = find_maximum(im)

    if offset:
        xt = xt + xsize / 2
        yt = yt + ysize / 2

    for i in range(len(xt)):
        xxt = xt[i]
        yyt = yt[i]
        permutations = (
            (xxt, yyt),
            (xxt - xsize, yyt),
            (xxt, yyt - ysize),
            (xxt - xsize, yyt - ysize),
        )
        res = [None, None, None]
        for p in permutations:
            im = np.ones_like(im0)
            pixels = shift(im, (p[1], p[0])).sum()
            if (
                pixels >= xsize * ysize * overlap[0]
                and pixels <= xsize * ysize * overlap[1]
            ):
                res = [p[0], p[1], pixels / xsize / ysize]
                break
        if res[0] is not None:
            break

    if full:
        return {'x': res[0], 'y': res[1], 'overlap': res[2]}
    else:
        if res[0] is None:
            res = (0, 0)
        return [res[1], res[0]]
