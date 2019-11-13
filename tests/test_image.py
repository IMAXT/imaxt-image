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


def test_ome_tiff():
    im = TiffImage(Path('tests/tubhiswt_C1.ome.tif'))
    assert 'OME' in im.metadata.as_dict()
    assert 'OME' in im.metadata.description


def test_dask():
    im = TiffImage(Path('tests/16bit.s.tif'))
    assert im.ndim == 2
    assert im.dtype == np.dtype('int16')

    da = im.to_dask()
    assert da.shape == (10, 10)

    xarr = im.to_xarray()
    assert len(xarr) == 10
