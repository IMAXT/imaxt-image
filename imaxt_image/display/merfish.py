from functools import partial
from pathlib import Path

import numpy as np

from imaxt_image.image import TiffImage
from imaxt_image.image.scaling import zscale

try:
    import holoviews as hv
    from holoviews.operation.datashader import regrid
    HAS_HOLOVIEWS = True
except:
    HAS_HOLOVIEWS = False

#PATH=Path("/data/apm56_a/db663/Imaxt_Samples/for IoA/test_merFISH_data")

def load_image(slice=0, channel=0, cycle=0, fov=0, path=None):
    filename = path / f'merFISH_merged_{cycle:02d}_0{fov:02d}.tif'
    image = TiffImage(filename)
    img = image.asarray()
    im1 = img[slice,channel].astype('float')
    vmin, vmax = zscale(im1)
    im1 = im1.clip(vmin, vmax)
    im1 = hv.Image(im1, bounds=(0, 0, 2048, 2048), vdims='Intensity')
    return im1

def display(df):
    if not HAS_HOLOVIEWS:
        return "Need holoviews to display images in Notebook"
    path = Path(df.path[0])
    image_stack = hv.DynamicMap(partial(load_image, path=path),
                    kdims=['slice', 'channel', 'cycle', 'fov'])
    regridded = regrid(image_stack).redim.range(slice=(0,3), channel=(0,3), cycle=(0,int(df.ncycles[0]-1)), fov=(0,int(df.nfovs[0]-1)))
    return regridded
