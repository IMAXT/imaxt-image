"""The image module contains functions to read TIFF files and to operate on them.
"""
import warnings

from imaxt_image.display.scaling import percentile, zscale  # noqa
from imaxt_image.io.tiffimage import TiffImage  # noqa

warnings.warn(
    'The image submodule is deprecated and will be removed in future versions. '
    'Please use the imaxt_image.display and imaxt_image.io modules instead!'
)
