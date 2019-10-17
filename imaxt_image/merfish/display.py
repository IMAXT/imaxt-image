from functools import lru_cache, partial
from pathlib import Path
from typing import List, Tuple

import dask.array as da
import holoviews as hv
import param
import zarr
from holoviews.operation.datashader import regrid

from imaxt_image.display.scaling import zscale


class zscale_filter(hv.Operation):

    normalize = param.Boolean(default=False)

    def _process(self, element, key=None):
        xs = element.dimension_values(0, expanded=False)
        ys = element.dimension_values(1, expanded=False)

        # setting flat=False will preserve the matrix shape
        data = element.dimension_values(2, flat=False)

        if self.p.normalize:
            dr = data.ravel()
            data = (data - dr.mean()) / dr.std() * 2 ** 16

        vmin, vmax = zscale(data.ravel())

        new_data = data.clip(vmin, vmax)

        label = element.label
        # make an exact copy of the element with all settings, just with different data and label:
        element = element.clone((xs, ys, new_data), label=label)
        return element


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
    if 'bit' in channel:
        names = [v[0] for v in gcycle.groups() if 'bit' in v[0]]
        if '0' in channel:
            channel = names[0]
        else:
            channel = names[1]
    img = da.from_zarr(data[f'fov={fov}/z={z}/cycle={cycle}/{channel}/raw'])
    ysize, xsize = img.shape
    img = hv.Image((range(ysize), range(xsize), img), vdims='Intensity', label=channel)
    return img


def display_raw(path: Path) -> hv.Layout:
    """Display a browsable MerFISH image cube in the Notebook.

    Parameters
    ----------
    path
        location of data in Zarr format

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
    dmaps = [zscale_filter(dmap, normalize=False) for dmap in dmaps]
    plots = [
        regrid(dmap).redim.range(
            fov=(0, data.attrs['fov'] - 1),
            cycle=(0, data.attrs['cycles'] - 1),
            z=(0, data.attrs['planes'] - 1),
        )
        for dmap in dmaps
    ]
    layout = hv.Layout(plots).cols(2)
    return layout


@lru_cache(maxsize=128)
def get_bit_images(
    path, fov: int, plane: int, imgtype: str = 'raw'
) -> List[zarr.Array]:
    """Create stack of images from a field of view.

    Given a field of view, extracts a stack of images containing
    the bits images in the correct order.

    Parameters
    ----------
    fov
        Field of view.
    plane
        Slice
    imgtype
        Type of image to return (raw, raw_offset, bkg, seg)

    Returns
    -------
    Stack of arrays containing all images from the field of view
    """
    data = zarr.open(f'{path}', 'r')
    plane = data[f'fov={fov}/z={plane}']
    j = 0
    bits = []
    for cycle in plane:
        try:
            bits.append(plane[f'{cycle}/bit={j}/{imgtype}'])
            bits.append(plane[f'{cycle}/bit={j+1}/{imgtype}'])
            j = j + 2
        except KeyError:
            pass
    return bits


def load_bit_image(
    fov: int, z: int, bit: int, *, path: str, imgtype: str = 'raw'
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
    bits = get_bit_images(path, fov, z, imgtype=imgtype)
    img = da.from_zarr(bits[bit])
    xsize, ysize = img.shape
    img = hv.Image((range(ysize), range(xsize), img), vdims='Intensity')
    return img


def display_bits(path: Path, imgtype: str = 'raw', zscale=False) -> hv.DynamicMap:
    """Display a browsable MerFISH bit image cube in the Notebook.

    Parameters
    ----------
    path
        location of data in Zarr format
    imgtype
        type of bit image to display (raw, raw_offset, bkg, seg)
    zscale
         perform z-scaling

    Returns
    -------
    image
    """
    data = zarr.open(f'{path}', 'r')
    dmap = hv.DynamicMap(partial(load_bit_image, path=path), kdims=['fov', 'z', 'bit'])

    if zscale:
        dmap = zscale_filter(dmap)

    plots = regrid(dmap).redim.range(
        fov=(0, data.attrs['fov'] - 1), z=(0, data.attrs['planes'] - 1), bit=(0, 15)
    )

    return regrid(plots)


def cutout(
    path: Path,
    coords: Tuple[int],
    fov: int,
    z: int,
    imgtype: str = 'raw',
    size: int = 20,
    marker: int = 0,
) -> hv.Layout:
    """Return cutouts around a position.

    Parameters
    ----------
    path
        location of data in Zarr format
    coords
        coordinates of centre
    fov
        field of view
    z
        slice
    imgtype
        type of image
    size
        cutout size in pixels
    marker
        size of optional marker in the centre

    Returns
    -------
    layout
    """
    imgs = get_bit_images(path, fov, z, imgtype=imgtype)
    a = slice(coords[1] - size, coords[1] + size)
    b = slice(coords[0] - size, coords[0] + size)
    plots = [hv.Image(im[a, b], label=im.name) for im in imgs]
    if marker > 0:
        m = hv.Points((0, 0)).opts(marker='o', size=marker, fill_alpha=0)
        plots = [p * m for p in plots]
    layout = hv.Layout(plots)
    return layout
