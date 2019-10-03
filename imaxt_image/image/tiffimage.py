"""Read and write TIFF files."""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from distributed.protocol import dask_deserialize, dask_serialize

from imaxt_image.external.tifffile import TiffFile, TiffWriter
from imaxt_image.image.metadata import Metadata
from imaxt_image.image.omexmlmetadata import OMEXMLMetadata


class TiffImage:
    """Main class to operate with TIFF images.

    Parameters
    ----------
    path
        Full path to image including filename.

    Examples
    --------
    >>> from imaxt.image import TiffImage
    >>> im = TiffImage('image.tif')
    >>> im.shape
    .... (4, 4, 2048, 2048)
    >>> arr = im.asarray()
    >>> arr1 = arr[0, 0]
    """

    def __init__(self, path: Union[str, Path]) -> None:
        if not isinstance(path, Path):
            path = Path(path)
        self.path = path
        if self.path.exists():
            self.tiff = TiffFile(f'{path}')
        else:
            raise FileNotFoundError(f'File not found: {self.path}')

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return size of image.
        """
        return self.tiff.series[0].pages.shape

    @property
    def dtype(self):
        """Return type of data.
        """
        return self.tiff.series[0].pages.dtype

    def to_dask(self, chunks=None):
        """Return a dask.array representation of the data.
        """
        chunks = chunks or self.shape
        return da.from_array(self, chunks=chunks)

    def to_xarray(self, dims=None, use_dask=True) -> xr.DataArray:
        """Return a DataArray representation of the data.

        Parameters
        ----------
        dims
            Dimension names associated with this array.
        use_dask
            Use Dask to lazyload the array.

        Returns
        -------
        Xarray DataArray
        """
        coords = tuple([range(d) for d in self.shape])
        if use_dask:
            arr = self.to_dask()
        else:
            arr = self.asarray()
        return xr.DataArray(arr, dims=dims, coords=coords)

    def asarray(self, out=None, maxworkers=1) -> np.ndarray:
        """Return TIFF image as numpy array.

        This functions returns the full content of the file as
        a multidimensional array.
        """
        return self.tiff.asarray(out=out, maxworkers=maxworkers)

    def __getitem__(self, item):
        return self.asarray()[item]

    @property
    def metadata(self) -> Metadata:
        """Return image metadata.

        Currently only returns ImageJ or OME-TIFF metadata.
        """
        description = self.tiff.pages[0].description
        if 'OME' not in description:
            return None
        return Metadata(description)

    @property
    def omemetadata(self) -> OMEXMLMetadata:
        description = self.tiff.pages[0].description
        if 'OME' not in description:
            return None
        return OMEXMLMetadata(self.tiff.pages[0].description)

    @staticmethod
    def write(img: np.ndarray, filename: Union[str, Path], compress: int = 0):
        """Write image array to filename.

        Parameters
        ----------
        img
            image array
        filename
            name of the file
        compress
            compression level
        """
        with TiffWriter(f'{filename}') as writer:
            writer.save(img, compress=compress)

    def close(self):
        self.tiff.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


@dask_serialize.register(TiffImage)
def serialize(image: TiffImage) -> Tuple[Dict, List[bytes]]:
    header = {}
    frames = [f'{image.path}'.encode()]
    return header, frames


@dask_deserialize.register(TiffImage)
def deserialize(header: Dict, frames: List[bytes]) -> TiffImage:
    return TiffImage(frames[0].decode())
