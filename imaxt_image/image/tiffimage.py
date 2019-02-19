"""Read and write TIFF files."""

from pathlib import Path
from typing import Tuple, Union

import numpy as np

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
    def shape(self) -> Tuple[int, ...]:
        """Return size of image.
        """
        return self.tiff.series[0].pages.shape

    def asarray(self, out=None) -> np.ndarray:
        """Return TIFF image as numpy array.

        This functions returns the full content of the file as
        a multidimensional array.
        """
        return self.tiff.asarray(out=out)

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
