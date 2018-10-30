from typing import ByteString

import numpy as np

import tif_lzw


def decompress(data: ByteString) -> np.ndarray:
    """LZW decompress data bits.

    Parameters
    ----------
    data
        Byte string to decompress.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    return tif_lzw.decode(arr, 100000000)
