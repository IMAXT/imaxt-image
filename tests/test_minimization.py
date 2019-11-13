import numpy as np
import pytest
import scipy.misc
from scipy.ndimage import shift

from imaxt_image.registration.minimization import find_overlap_conf


def test_find_overlap_conf():
    im = scipy.misc.face(gray=True)
    im2 = shift(im, (100, 100))
    conf = np.ones_like(im)

    res = find_overlap_conf(im, im2, conf, conf, desp=(-100, -100))
    assert round(res.x) == -100
    assert round(res.y) == -100

    res = find_overlap_conf(im, im2, conf, conf, desp=(-90, -90))
    assert round(res.x) == -100
    assert round(res.y) == -100

    res = find_overlap_conf(im, im, conf, conf, desp=(0, 0))
    assert round(res.x) == 0
    assert round(res.y) == 0
