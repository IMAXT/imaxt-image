import pytest
import scipy.misc
from scipy.ndimage import shift

from imaxt_image.registration.cross_correlation import (enhanced_correlation,
                                                        find_maximum,
                                                        find_shift)


def test_enhanced_correlation():
    im = scipy.misc.face(gray=True)
    im2 = shift(im, (100, 100))

    xcorr = enhanced_correlation(im, im2)
    assert xcorr.real.max() > 0.001

    with pytest.raises(AssertionError):
        enhanced_correlation(im, im2[:100, :100])


def test_find_maximum():
    im = scipy.misc.face(gray=True)
    im2 = shift(im, (100, 100))

    xcorr = enhanced_correlation(im, im2)
    x, y = find_maximum(xcorr.real)
    assert x == 924
    assert y == 668


def test_find_shift():
    im = scipy.misc.face(gray=True)
    im2 = shift(im, (100, 100))

    res = find_shift(im, im2, overlap=(0.6, 0.8))
    assert res['x'] == -100
    assert res['y'] == -100
    assert res['overlap'] > 0.7

    res = find_shift(im, im, offset=(100, 100), overlap=(0.6, 0.8))
    assert res['x'] == 0
    assert res['y'] == 0
    assert res['overlap'] > 0.7
