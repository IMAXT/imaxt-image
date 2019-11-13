from pathlib import Path

import numpy as np
import pytest

from imaxt_image.io.tiffimage import TiffImage
from imaxt_image.misc import bytescale


def test_bytescale():
    img = np.random.randn(100, 100)
    im = bytescale(img)
    assert im.dtype == np.uint8

    with pytest.raises(ValueError):
        bytescale(img, high=0, low=1)

    with pytest.raises(ValueError):
        bytescale(img, cmax=0, cmin=1)

    im = bytescale(img * 0)
    assert im.dtype == np.uint8

    im = bytescale(im)
    assert im.dtype == np.uint8
