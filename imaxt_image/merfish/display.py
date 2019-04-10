from functools import partial
from pathlib import Path

import holoviews as hv
import numpy as np
import zarr
from holoviews.operation.datashader import regrid

from imaxt_image.image.scaling import zscale


def rebin(arr: np.ndarray, scale: int) -> np.ndarray:
    """Rebin array.

    Parameters
    ----------
    arr
        data array
    scale
        scale factor to downsize image

    Returns
    -------
    rebinned array
    """
    shape = [*map(lambda x: x // scale, arr.shape)]
    sh = (shape[0], arr.shape[0] // shape[0], shape[1], arr.shape[1] // shape[1])
    return arr.reshape(sh).mean(-1).mean(1)


def prepare_image(img: np.ndarray, scale: int = 4) -> np.ndarray:
    """Return scaled and rebinned image for display.

    Parameters
    ----------
    img
        image array
    scale
        scale to downsize input image

    Returns
    -------
    rescale image array
    """
    if scale is not None:
        img = rebin(img, scale)
    vmin, vmax = zscale(img)
    img = (img.clip(vmin, vmax) - img.mean()) / img.std()
    return img


def load_image(
    fov: int,
    data: zarr.Array = None,
    channel: str = 'microbeads',
    cycle: int = 0,
    z: int = 0,
) -> hv.Image:
    """Get an image from the cube

    Parameters
    ----------
    fov
        field of view
    data
        array
    channel
        channel
    cycle
        cycle
    z
        optical slice

    Returns
    -------
    image to display
    """
    gcycle = data[f'fov={fov}/z={z}/cycle={cycle}']
    names = [v[0] for v in gcycle.groups() if 'bit' in v[0]]
    if 'bit=0' in channel:
        channel = names[0]
    elif 'bit=1' in channel:
        channel = names[1]
    img = prepare_image(data[f'fov={fov}/z={z}/cycle={cycle}/{channel}/raw'][:])
    img = hv.Image(img, bounds=(0, 0, 2048, 2048), vdims='Intensity')
    return img


def browse(path: Path, colormap: str = 'Viridis') -> hv.Layout:
    """Display a browsable MerFISH image cube in the Notebook.

    Parameters
    ----------
    path
        location of data in Zarr format
    colormap
         colormap to use

    Returns
    -------
    layout
    """
    data = zarr.open(f'{path}', 'r')
    dmaps = [
        hv.DynamicMap(
            partial(load_image, channel=channel, data=data), kdims=['fov', 'cycle', 'z']
        )
        for channel in ['nuclei', 'microbeads', 'bit=0', 'bit=1']
    ]
    plots = [
        regrid(dmap)
        .redim.range(
            fov=(0, data.attrs['fov'] - 1),
            cycle=(0, data.attrs['cycles'] - 1),
            z=(0, data.attrs['planes'] - 1),
        )
        .opts(cmap=colormap)
        for dmap in dmaps
    ]
    layout = hv.Layout(plots).cols(2)
    return layout
