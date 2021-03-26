from typing import Tuple

import numpy as np
import xarray as xr
from scipy.ndimage import fourier_gaussian, shift

from .mutual_information import iqr
from .utils import ShiftResult, extract_overlap


from skimage.registration._phase_cross_correlation import (
    _compute_error,
    _compute_phasediff,
    _upsampled_dft,
    _masked_phase_cross_correlation,
    fft,
)


def find_shift(
    im0: np.ndarray,
    im1: np.ndarray,
    overlap: Tuple[float] = None,
    border_width: int = 0,
    upsample_factor: int = 1,
    initial_shift: Tuple[int] = None,
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
        Upsampling factor
    initial_shift
        Initial guess

    Returns
    -------
    The result represented as a ``ShiftResult`` object.

    References
    ----------
    See: http://www.sci.utah.edu/publications/SCITechReports/UUSCI-2006-020.pdf
    """

    if isinstance(im0, xr.DataArray):
        this_im0 = im0.values
    else:
        this_im0 = im0

    if isinstance(im1, xr.DataArray):
        this_im1 = im1.values
    else:
        this_im1 = im1

    if initial_shift is not None:
        im0_cut, im1_cut = extract_overlap(this_im0, this_im1, initial_shift)
    else:
        initial_shift = (0, 0)
        im0_cut, im1_cut = this_im0, this_im1

    ysize, xsize = im0.shape
    offset, error, phase = phase_cross_correlation(
        im0_cut,
        im1_cut,
        border_width=border_width,
        upsample_factor=upsample_factor,
        extra=True,
    )

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
        offset = [p[0] + initial_shift[0], p[1] + initial_shift[1]]
        pixels = shift(im, offset, mode="constant", cval=0, order=1).sum()
        if pixels > 0:
            nmi = iqr(im0_cut, im1_cut, offset=p)
            if nmi > mi:
                if overlap is not None:
                    if (
                        pixels >= xsize * ysize * overlap[0]
                        and pixels <= xsize * ysize * overlap[1]
                    ):
                        res = [offset[0], offset[1], pixels / xsize / ysize, mi]
                        mi = nmi
                else:
                    res = [offset[0], offset[1], pixels / xsize / ysize, mi]
                    mi = nmi

    res = {"y": res[0], "x": res[1], "overlap": res[2], "error": error, "iqr": mi}
    return ShiftResult(res)


def phase_cross_correlation(  # noqa: C901
    reference_image,
    moving_image,
    *,
    upsample_factor=1,
    space="real",
    return_error=True,
    reference_mask=None,
    border_width=0,
    extra=False,
    moving_mask=None,
    overlap_ratio=0.3
):
    """Efficient subpixel image translation registration by cross-correlation.

    This code gives the same precision as the FFT upsampled cross-correlation
    in a fraction of the computation time and with reduced memory requirements.
    It obtains an initial estimate of the cross-correlation peak by an FFT and
    then refines the shift estimation by upsampling the DFT only in a small
    neighborhood of that estimate by means of a matrix-multiply DFT.

    Parameters
    ----------
    reference_image : array
        Reference image.
    moving_image : array
        Image to register. Must be same dimensionality as
        ``reference_image``.
    upsample_factor : int, optional
        Upsampling factor. Images will be registered to within
        ``1 / upsample_factor`` of a pixel. For example
        ``upsample_factor == 20`` means the images will be registered
        within 1/20th of a pixel. Default is 1 (no upsampling).
        Not used if any of ``reference_mask`` or ``moving_mask`` is not None.
    space : string, one of "real" or "fourier", optional
        Defines how the algorithm interprets input data. "real" means
        data will be FFT'd to compute the correlation, while "fourier"
        data will bypass FFT of input data. Case insensitive. Not
        used if any of ``reference_mask`` or ``moving_mask`` is not
        None.
    return_error : bool, optional
        Returns error and phase difference if on, otherwise only
        shifts are returned. Has noeffect if any of ``reference_mask`` or
        ``moving_mask`` is not None. In this case only shifts is returned.
    reference_mask : ndarray
        Boolean mask for ``reference_image``. The mask should evaluate
        to ``True`` (or 1) on valid pixels. ``reference_mask`` should
        have the same shape as ``reference_image``.
    moving_mask : ndarray or None, optional
        Boolean mask for ``moving_image``. The mask should evaluate to ``True``
        (or 1) on valid pixels. ``moving_mask`` should have the same shape
        as ``moving_image``. If ``None``, ``reference_mask`` will be used.
    overlap_ratio : float, optional
        Minimum allowed overlap ratio between images. The correlation for
        translations corresponding with an overlap ratio lower than this
        threshold will be ignored. A lower `overlap_ratio` leads to smaller
        maximum translation, while a higher `overlap_ratio` leads to greater
        robustness against spurious matches due to small overlap between
        masked images. Used only if one of ``reference_mask`` or
        ``moving_mask`` is None.

    Returns
    -------
    shifts : ndarray
        Shift vector (in pixels) required to register ``moving_image``
        with ``reference_image``. Axis ordering is consistent with
        numpy (e.g. Z, Y, X)
    error : float
        Translation invariant normalized RMS error between
        ``reference_image`` and ``moving_image``.
    phasediff : float
        Global phase difference between the two images (should be
        zero if images are non-negative).

    References
    ----------
    .. Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           "Efficient subpixel image registration algorithms,"
           Optics Letters 33, 156-158 (2008). DOI: 10.1364/OL.33.000156
    .. James R. Fienup, "Invariant error metrics for image reconstruction"
           Optics Letters 36, 8352-8357 (1997). DOI: 10.1364/AO.36.008352
    .. Dirk Padfield. Masked Object Registration in the Fourier Domain.
           IEEE Transactions on Image Processing, vol. 21(5),
           pp. 2706-2718 (2012). DOI: 10.1109/TIP.2011.2181402
    .. D. Padfield. "Masked FFT registration". In Proc. Computer Vision and
           Pattern Recognition, pp. 2918-2925 (2010).
           DOI: 10.1109/CVPR.2010.5540032

    """
    if (reference_mask is not None) or (moving_mask is not None):
        return _masked_phase_cross_correlation(
            reference_image, moving_image, reference_mask, moving_mask, overlap_ratio
        )

    # images must be the same shape
    if reference_image.shape != moving_image.shape:
        raise ValueError("images must be same shape")

    # assume complex data is already in Fourier space
    if space.lower() == "fourier":
        src_freq = reference_image
        target_freq = moving_image
    # real data needs to be fft'd.
    elif space.lower() == "real":
        src_freq = fft.fftn(reference_image)
        target_freq = fft.fftn(moving_image)
    else:
        raise ValueError('space argument must be "real" of "fourier"')

    # Whole-pixel shift - Compute cross-correlation by an IFFT
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    if extra:
        norm = (
            np.sqrt(src_freq * src_freq.conj() + target_freq * target_freq.conj())
            + 1e-10
        )
        image_product = image_product / norm
        image_product = fourier_gaussian(image_product, 5)

    cross_correlation = fft.ifftn(image_product)

    if border_width > 0:
        cross_correlation[:border_width, :] = 0
        cross_correlation[-border_width:, :] = 0
        cross_correlation[:, :border_width] = 0
        cross_correlation[:, -border_width:] = 0

    # Locate maximum
    maxima = np.unravel_index(
        np.argmax(np.abs(cross_correlation)), cross_correlation.shape
    )
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
        normalization = src_freq.size * upsample_factor ** 2
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor
        cross_correlation = _upsampled_dft(
            image_product.conj(),
            upsampled_region_size,
            upsample_factor,
            sample_region_offset,
        ).conj()
        cross_correlation /= normalization
        # Locate maximum and map back to original pixel grid
        maxima = np.unravel_index(
            np.argmax(np.abs(cross_correlation)), cross_correlation.shape
        )
        CCmax = cross_correlation[maxima]

        maxima = np.array(maxima, dtype=np.float64) - dftshift

        shifts = shifts + maxima / upsample_factor

        if return_error:
            src_amp = _upsampled_dft(src_freq * src_freq.conj(), 1, upsample_factor)[
                0, 0
            ]
            src_amp /= normalization
            target_amp = _upsampled_dft(
                target_freq * target_freq.conj(), 1, upsample_factor
            )[0, 0]
            target_amp /= normalization

    # If its only one row or column the shift along that dimension has no
    # effect. We set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shifts[dim] = 0

    if return_error:
        return (
            shifts,
            _compute_error(CCmax, src_amp, target_amp),
            _compute_phasediff(CCmax),
        )
    else:
        return shifts
