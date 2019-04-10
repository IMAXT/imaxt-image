import numpy as np
import pytest

from imaxt_image.merfish import browse
from imaxt_image.merfish.display import prepare_image, rebin


def test_rebin():
    arr = np.random.random((100, 100))
    out = rebin(arr, scale=4)

    assert out.shape == (25, 25)


def test_prepare_image():
    arr = np.random.random((100, 100))
    out = prepare_image(arr)

    assert out.shape == (25, 25)
    assert out.mean() == pytest.approx(0.0, abs=0.1)
    assert out.std() == pytest.approx(1.0, abs=0.1)
