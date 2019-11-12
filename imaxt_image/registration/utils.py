import numpy as np
from scipy.ndimage import shift


class ShiftResult(dict):
    """Represents the shift result.

    Attributes
    ----------
    x : float
        x offset
    y : float
        y offset
    overlap: float
        percentage of overlap
    error : float
        error
    iqr : float
        information quality ratio
    mi : float
        mutual information
    avg_flux : float
        average flux over overlapping area

    Notes
    -----
    Not all algorithms return all variables.
    """

    __slots__ = ['x', 'y', 'overlap', 'error', 'iqr', 'mi', 'avg_flux']

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join(
                [k.rjust(m) + ': ' + repr(v) for k, v in sorted(self.items())]
            )
        else:
            return self.__class__.__name__ + '()'

    def __dir__(self):
        return list(self.keys())


def extract_overlap(im1, im2, offset, conf1=None, conf2=None):
    """[summary]

    Parameters
    ----------
    im1 : [type]
        [description]
    im2 : [type]
        [description]
    offset : [type]
        [description]
    conf1 : [type], optional
        [description], by default None
    conf2 : [type], optional
        [description], by default None

    Returns
    -------
    [type]
        [description]
    """
    if offset == (0, 0):
        if (conf1 is not None) and (conf2 is not None):
            res = (im1, im2, conf1, conf2)
        else:
            res = (im1, im2)
        return res

    c = shift(
        np.ones_like(im1), (offset[0], offset[1]), mode='constant', cval=0, order=1
    )
    mask = c > 0
    nrows = sum(c.sum(axis=1) > 0)
    im1_out = im1[mask].reshape(nrows, -1).astype('float32')
    if conf1 is not None:
        conf1_out = conf1[mask].reshape(nrows, -1)

    c = shift(
        np.ones_like(im2), (-offset[0], -offset[1]), mode='constant', cval=0, order=1
    )
    mask = c > 0
    nrows = sum(c.sum(axis=1) > 0)
    im2_out = im2[mask].reshape(nrows, -1).astype('float32')
    if conf2 is not None:
        conf2_out = conf2[mask].reshape(nrows, -1)

    if (conf1 is not None) and (conf2 is not None):
        res = (im1_out, im2_out, conf1_out, conf2_out)
    else:
        res = (im1_out, im2_out)
    return res
