from pathlib import Path

import holoviews as hv
import numpy as np
import param
from astropy.table import Table
from holoviews import dim, opts
from holoviews.operation.datashader import datashade, dynspread, regrid

from imaxt_image.image import TiffImage, zscale


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


class CubeViewer:
    def __init__(self, data_dir, imgtype='cubes'):
        self.data_dir = Path(data_dir)
        self.imgtype = imgtype
        self.cubes = [f.name for f in (self.data_dir / self.imgtype).glob('*.tif')]

    @property
    def channels(self):
        image = TiffImage(self.data_dir / self.imgtype / self.cubes[0]).asarray()
        return image.shape[0]

    def load_image(self, channel, cube):
        image = TiffImage(self.data_dir / self.imgtype / cube).asarray()
        img = image[channel][0:1000,0:1000]
        xsize, ysize = img.shape
        img = hv.Image(
            img, bounds=(0, 0, img.shape[1], img.shape[0]), vdims='Intensity'
        )
        return img

    def display(self, xaxis=None, yaxis=None, colorbar=True, toolbar='below'):
        image_stack = hv.DynamicMap(self.load_image, kdims=['channel', 'cube'])
        image_stack = zscale_filter(image_stack)
        # Apply regridding in case data is large and set a global Intensity range
        regridded = (
            regrid(image_stack)
            .redim.range(channel=(0, self.channels - 1))
            .redim.values(cube=self.cubes)
        )
        display_obj = regridded.opts(
            plot={
                'Image': dict(
                    colorbar=colorbar,
                    toolbar=toolbar,
                    xaxis=xaxis,
                    yaxis=yaxis,
                    aspect='equal',
                )
            }
        )
        display_obj.data_dir = self.data_dir
        return display_obj


class Catalogue:
    def __init__(self, data_dir, filename):
        self.data_dir = Path(data_dir)
        self.df = Table.read(self.data_dir / 'catalogue' / filename).to_pandas()

    def display(self):
        points = hv.Points(
            (self.df['X'], self.df['Y'], np.log10(self.df['flux_25'])), vdims='z'
        )
        res = points.opts(size=0.25)  # dynspread(datashade(points))
        res.opts(aspect='equal').opts(opts.Points(color='z'))
        return res

    @classmethod
    def from_view(cls, view):
        key, val = list(view.data.items())[-1]
        filename = key[1].replace('.tif', '.fits')
        return cls(view.data_dir, filename)
