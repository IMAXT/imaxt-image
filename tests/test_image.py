from pathlib import Path

import numpy as np
import pytest

from imaxt_image.display.scaling import percentile, zscale
from imaxt_image.io.tiffimage import TiffImage


def imread(img):
    return np.zeros((4, 4, 128, 128))


def test_zscale():
    im = imread(None)
    vmin, vmax = zscale(im)
    assert vmin == 0.0
    assert vmax == 0.0


def test_percentile():
    im = imread(None)
    vmin, vmax = percentile(im, 50)
    assert vmin == 0.0
    assert vmax == 0.0


# def test_imshow(mocker):
#     mocker.patch.object(plt, 'imshow', autospec=True)
#     mocker.patch.object(plt, 'axis', autospec=True)
#     im = imread(None)
#     imshow(im)
#     plt.imshow.assert_called_once()


# def test_multishow(mocker):
#     mocker.patch.object(plt, 'imshow', autospec=True)
#     mocker.patch.object(plt, 'subplot', autospec=True)
#     mocker.patch.object(plt, 'axis', autospec=True)
#     mocker.patch.object(plt, 'tight_layout', autospec=True)
#     im = imread(None)
#     multishow(im, orientation='landscape')
#     assert plt.imshow.call_count == im.shape[0]
#     assert plt.subplot.call_count == im.shape[0]

#     multishow(im, orientation='portrait')
#     assert plt.imshow.call_count == im.shape[0] * 2
#     assert plt.subplot.call_count == im.shape[0] * 2


def test_image():
    im = TiffImage('tests/16bit.s.tif')
    assert im.shape == (10, 10)

    arr = im.asarray()
    assert arr.shape == (10, 10)


def test_image_notfound():
    with pytest.raises(FileNotFoundError):
        TiffImage('tests/notfound.tif')


def test_image_pathlib():
    im = TiffImage(Path('tests/16bit.s.tif'))
    assert im.shape == (10, 10)


def test_metadata():
    im = TiffImage(Path('tests/16bit.s.tif'))
    assert im.metadata is None
