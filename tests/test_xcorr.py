import scipy.misc
from scipy.ndimage import shift

from imaxt_image.registration.cross_correlation import find_shift


def test_find_shift():
    im = scipy.misc.face(gray=True)
    im2 = shift(im, (100, 100))

    res = find_shift(im, im2, overlap=(0.6, 0.8))
    assert res['x'] == -100
    assert res['y'] == -100
    assert res['overlap'] > 0.7

    res = find_shift(im, im, overlap=(0.6, 1.0))
    assert res['x'] == 0
    assert res['y'] == 0
    assert res['overlap'] > 0.7
