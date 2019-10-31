from typing import Tuple

import numpy as np
from scipy.ndimage import fourier_gaussian, shift
from skimage.feature.register_translation import (_compute_error,
                                                  _compute_phasediff,
                                                  _upsampled_dft)

from .mutual_information import iqr


class ShiftResult(dict):
    """ Represents the shift result.

    Attributes
    ----------
    x : float
        x offset
    y : float
        y offset
    overlap: float
        percentage of overlap
    error : float
        error
    iqr : float
        information quality ratio
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())


def find_shift(
    im0: np.ndarray,
    im1: np.ndarray,
    overlap: Tuple[float] = (0.08, 0.12),
    border_width: int= 0,
    upsample_factor: int=1

) -> ShiftResult:
    """Find shift between images using cross correlation.

    Parameters
    ----------
    im0
        Reference image
    im1
        Target image
    overlap
        Image overlap range to exclude possible offsets
    border_width
        Ignore maxima around this image border width
    upsample_factor
        Upsampling factor.

    Returns
    -------
    The result represented as a ``ShiftResult`` object.

    References
    ----------
    See: http://www.sci.utah.edu/publications/SCITechReports/UUSCI-2006-020.pdf
    """
    assert overlap[0] < overlap[1]
    im0 = im0[:]
    im1 = im1[:]
    ysize, xsize = im0.shape
    offset, error, phase = register_translation(im0, im1, border_width=border_width, upsample_factor=upsample_factor)

    yyt = offset[0]
    xxt = offset[1]
    permutations = (
        (yyt, xxt),
        (yyt + ysize, xxt),
        (yyt, xxt + xsize),
        (yyt + ysize, xxt + xsize),
        (yyt - ysize, xxt),
        (yyt, xxt - xsize),
        (yyt - ysize, xxt - xsize),
    )
    res = [None, None, None]
    mi = 0
    for p in permutations:
        im = np.ones_like(im0)
        pixels = shift(im, (p[0], p[1])).sum()
        if pixels > 0:
            nmi = iqr(im0, im1, offset=(p[1], p[0]))
            if nmi > mi:
                if overlap is not None:
                    if (
                       pixels >= xsize * ysize * overlap[0]
                       and pixels <= xsize * ysize * overlap[1]
                       ):
                        res = [p[0], p[1], pixels / xsize / ysize, mi]
                        mi = nmi
                else:
                    res = [p[0], p[1], pixels / xsize / ysize, mi]
                    mi = nmi

    res = {'y': res[0], 'x': res[1], 'overlap': res[2], 'error': error, 'iqr': mi}
    return ShiftResult(res)


def register_translation(src_image, target_image, upsample_factor=1,    # noqa: C901
                         space="real", return_error=True, border_width=0):
    """
    Efficient subpixel image translation registration by cross-correlation.
    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Parameters
    ----------
    src_image : array
        Reference image.
    target_image : array
        Image to register.  Must be same dimensionality as ``src_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel.  Default is 1 (no upsampling)
    space : string, one of "real" or "fourier", optional
        Defines how the algorithm interprets input data.  "real" means data
        will be FFT'd to compute the correlation, while "fourier" data will
        bypass FFT of input data.  Case insensitive.
    return_error : bool, optional
        Returns error and phase difference if on,
        otherwise only shifts are returned

    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``target_image`` with
        ``src_image``.  Axis ordering is consistent with numpy (e.g. Z, Y, X)
    error : float
        Translation invariant normalized RMS error between ``src_image`` and
        ``target_image``.
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).

    References
    ----------
    Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
    "Efficient subpixel image registration algorithms,"
    Optics Letters 33, 156-158 (2008).

    James R. Fienup, "Invariant error metrics for image reconstruction"
    Optics Letters 36, 8352-8357 (1997).
    """
    # images must be the same shape
    if src_image.shape != target_image.shape:
        raise ValueError("Error: images must be same size for "
                         "register_translation")

    # assume complex data is already in Fourier space
    if space.lower() == 'fourier':
        src_freq = src_image
        target_freq = target_image
    # real data needs to be fft'd.
    elif space.lower() == 'real':
        src_freq = np.fft.fftn(src_image)
        target_freq = np.fft.fftn(target_image)
    else:
        raise ValueError("Error: register_translation only knows the \"real\" "
                         "and \"fourier\" values for the ``space`` argument.")

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    src_freq = fourier_gaussian(src_freq, 5)
    target_freq = fourier_gaussian(target_freq, 5)
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    norm = np.sqrt(src_freq * src_freq.conj() * target_freq * target_freq.conj()) + 1e-10
    image_product = image_product / norm
    image_product = fourier_gaussian(image_product, 5)
    cross_correlation = np.fft.ifftn(image_product)
    if border_width > 0:
        cross_correlation[:border_width, :] = 0
        cross_correlation[-border_width:, :] = 0
        cross_correlation[:, :border_width] = 0
        cross_correlation[:, -border_width:] = 0

    # Locate maximum
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                              cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    shifts = np.array(maxima, dtype=np.float64)
    shifts[shifts > midpoints] -= np.array(shape)[shifts > midpoints]

    if upsample_factor == 1:
        if return_error:
            src_amp = np.sum(np.abs(src_freq) ** 2) / src_freq.size
            target_amp = np.sum(np.abs(target_freq) ** 2) / target_freq.size
            CCmax = cross_correlation[maxima]
    # If upsampling > 1, then refine estimate with matrix multiply DFT
    else:
        # Initial shift estimate in upsampled grid
        shifts = np.round(shifts * upsample_factor) / upsample_factor
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        upsample_factor = np.array(upsample_factor, dtype=np.float64)
        normalization = (src_freq.size * upsample_factor ** 2)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor
        cross_correlation = _upsampled_dft(image_product.conj(),
                                           upsampled_region_size,
                                           upsample_factor,
                                           sample_region_offset).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)),
                                  cross_correlation.shape)
        CCmax = cross_correlation[maxima]

        maxima = np.array(maxima, dtype=np.float64) - dftshift

        shifts = shifts + maxima / upsample_factor

        if return_error:
            src_amp = _upsampled_dft(src_freq * src_freq.conj(),
                                     1, upsample_factor)[0, 0]
            src_amp /= normalization
            target_amp = _upsampled_dft(target_freq * target_freq.conj(),
                                        1, upsample_factor)[0, 0]
            target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    if return_error:
        return shifts, _compute_error(CCmax, src_amp, target_amp),\
            _compute_phasediff(CCmax)
    else:
        return shifts
