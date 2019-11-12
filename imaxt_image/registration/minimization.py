from typing import List

import numpy as np
from scipy.ndimage import gaussian_filter, shift
from scipy.optimize import minimize

from .mutual_information import mutual_information
from .utils import ShiftResult, extract_overlap


def compare_imgs(desp, im1, im2, co1, co2, do_print):
    """[summary]

    Parameters
    ----------
    desp : [type]
        [description]
    im1 : [type]
        [description]
    im2 : [type]
        [description]
    co1 : [type]
        [description]
    co2 : [type]
        [description]
    do_print : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    im2_desp = shift(im2, desp, mode='constant', cval=0, order=1)
    co2_desp = shift(co2, desp, mode='constant', cval=0, order=1)

    co_all = co2_desp * co1

    s = np.sum(co_all)
    if s == 0.0:
        return np.inf

    err = np.sum((co_all * (im2_desp - im1) ** 2) / (im1 + 0.001)) / s

    if do_print:  # pragma: nocover
        t_s = ''
        for t in desp:
            if np.abs(t) < 1e-3:
                t_s += ' {0:.1e}'.format(t)
            else:
                t_s += ' {0:.3f}'.format(t)
        print(t_s + ' {0:f}'.format(err))

    return err


def find_overlap_conf(im0, im1, conf0, conf1, desp: List[int], ftol=0.1, debug=False):
    """[summary]

    This calculates the displacementes that minimize the squared difference
    between images.

    Parameters
    ----------
    img_ref : [type]
        [description]
    conf_ref : [type]
        [description]
    img_obj : [type]
        [description]
    conf_obj : [type]
        [description]
    desp : []
        [description]

    Returns
    -------
    [type]
        [description]
    """
    im0_cut, im1_cut, conf0_cut, conf1_cut = extract_overlap(
        im0, im1, desp, conf0, conf1
    )

    im0_cut = gaussian_filter(im0_cut, sigma=3)
    im1_cut = gaussian_filter(im1_cut, sigma=3)

    res = minimize(
        compare_imgs,
        [0.0, 0.0],
        args=(im0_cut, im1_cut, conf0_cut, conf1_cut, debug),
        method='Powell',
        options={'ftol': ftol},
    )

    mi = mutual_information(im0_cut, im1_cut, res['x'])
    avg_flux = (im0_cut * conf0_cut).sum() / conf0_cut.sum()

    res = {
        'y': desp[0] + res['x'][0],
        'x': desp[1] + res['x'][1],
        'mi': mi,
        'avg_flux': avg_flux,
    }
    return ShiftResult(res)
