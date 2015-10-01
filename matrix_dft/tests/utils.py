import numpy as np
import logging
_log = logging.getLogger('matrixdft')

def complexinfo(a, extra_message=None):
    """ Print some info about the sum of real and imaginary parts of an array
    """

    if extra_message:
        _log.debug("\t", extra_message)
    re = a.real.copy()
    im = a.imag.copy()
    _log.debug("\t%.2e  %.2g  =  re.sum im.sum" % (re.sum(), im.sum()))
    _log.debug("\t%.2e  %.2g  =  abs(re).sum abs(im).sum" % (abs(re).sum(), abs(im).sum()))


def euclid2(s, c=None):
    """ Compute Euclidean distance between points across an 2d ndarray

    Paramters
    ----------
    s : tuple
        shape of array
    c : tuple
        coordinates of point to measure distance from

    """

    if c is None:
        c = (0.5*float(s[0]),  0.5*float(s[1]))

    y, x = np.indices(s)
    r2 = (x - c[0])**2 + (y - c[1])**2

    return r2

def makedisk(s=None, c=None, r=None, inside=1.0, outside=0.0, grey=None, t=None):
    """ Create a 2D ndarray containing a uniform circular aperture with
    given center and size
    """


    # fft style or sft asymmetric style - center = nx/2, ny/2

    disk = np.where(euclid2(s, c=c) <= r*r, inside, outside)
    return disk

def make_parity_test_array(npix):
    arr = makedisk(s=(npix, npix), r=(npix / 4.0))
    arr[int(npix/2):,int(npix/2):] = 0.0
    return arr
