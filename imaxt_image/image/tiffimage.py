"""Read and write TIFF files."""

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

from imaxt_image.external.tifffile import TiffFile


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
    def shape(self) -> Tuple[int, ...]:
        """Return size of image.
        """
        return self.tiff.series[0].pages.shape

    def asarray(self) -> np.ndarray:
        """Return TIFF image as numpy array.

        This functions returns the full content of the file as
        a multidimensional array.
        """
        return self.tiff.asarray(out='memmap')

    @property
    def metadata(self) -> Dict:
        """Return image metadata.

        Currently only returns ImageJ or OME-TIFF metadata.
        """
        if self.tiff.is_ome:
            return self.tiff.ome_metadata
        elif self.tiff.is_imagej:
            return self.tiff.imagej_metadata
