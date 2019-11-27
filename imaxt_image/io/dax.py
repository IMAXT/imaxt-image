from pathlib import Path

import dask.array as da
import numpy as np
import yaml
from dask import delayed


@delayed
def _read_dax(daxname, image_width, image_height, frame_number):
    startByte = frame_number * image_height * image_width * 2
    endByte = startByte + 2 * (image_height * image_width)

    fh = open(daxname, 'rb')
    fh.seek(startByte)
    data = fh.read(endByte - startByte)
    img = np.frombuffer(data, dtype='int16')
    img = img.reshape((image_height, image_width))
    return img


def read_dax(filename: Path) -> da.Array:
    """Read a DAX file

    Parameters
    ----------
    filename
        Full path to file name

    Returns
    -------
    data array
    """
    infname = f'{filename}'.replace('.dax', '.inf')
    conf = yaml.safe_load(open(infname).read().replace('=', ':'))
    image_width = conf['x_end']
    image_height = conf['y_end']
    images = []
    for i in range(conf['number of frames']):
        im = _read_dax(filename, image_width, image_height, i)
        im = da.from_delayed(im, shape=(image_width, image_height), dtype='uint16')
        images.append(im)
    return da.stack(images)
