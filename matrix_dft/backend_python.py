"""
    MatrixDFT: Matrix-based discrete Fourier transforms for computing PSFs.

    See Soummer et al. 2007 JOSA

    The main user interface in this module is a class MatrixFourierTransform.
    Internally this will call one of several subfunctions depending on the
    specified centering type. These have to do with where the (0, 0) element of
    the Fourier transform is located, i.e. where the PSF center ends up.

        - 'FFTSTYLE' centered on one pixel
        - 'SYMMETRIC' centerd on crosshairs between middle pixel
        - 'ADJUSTABLE', always centered in output array depending on
          whether it is even or odd

    'ADJUSTABLE' is the default.

    This module was originally called "Slow Fourier Transform", and this
    terminology still appears in some places in the code.  Note that this is
    'slow' only in the sense that if you perform the exact same calculation as
    an FFT, the FFT algorithm is much faster. However this algorithm gives you
    much more flexibility in choosing array sizes and sampling, and often lets
    you replace "fast calculations on very large arrays" with "relatively slow
    calculations on much smaller ones".

    Example
    -------
    mf = matrixDFT.MatrixFourierTransform()
    result = mf.perform(pupilArray, focalplane_size, focalplane_npix)

    History
    -------
    Code originally by A. Sivaramakrishnan
    2010-11-05 Revised normalizations for flux conservation consistent
        with Soummer et al. 2007. Updated documentation.  -- M. Perrin
    2011-2012: Various enhancements, detailed history not kept, sorry.
    2012-05-18: module renamed SFT.py -> matrixDFT.py
    2012-09-26: minor big fixes
    2015-01-21: Eliminate redundant code paths, correct parity flip,
                PEP8 formatting pass (except variable names)-- J. Long

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import math
from cmath import exp

import logging
_log = logging.getLogger('poppy')

from .constants import (CENTERING_CHOICES, FFTSTYLE,
                        SYMMETRIC, ADJUSTABLE)

def matrix_multiply(matrix_x, matrix_y):
    return [
        [sum(a * b for a, b in zip(x_row, y_col)) for y_col in zip(*matrix_y)]
        for x_row in matrix_x
    ]

def matrix_dft(plane, nlamD, npix,
               offset=None, inverse=False, centering=FFTSTYLE):
    """Perform a matrix discrete Fourier transform with selectable
    output sampling and centering.

    Where parameters can be supplied as either scalars or 2-tuples, the first
    element of the 2-tuple is used for the Y dimension and the second for the
    X dimension. This ordering matches that of numpy.ndarray.shape attributes
    and that of Python indexing.

    To achieve exact correspondence to the FFT set nlamD and npix to the size
    of the input array in pixels and use 'FFTSTYLE' centering. (n.b. When
    using `numpy.fft.fft2` you must `numpy.fft.fftshift` the input pupil both
    before and after applying fft2 or else it will introduce a checkerboard
    pattern in the signs of alternating pixels!)

    Parameters
    ----------
    plane : 2D ndarray
        2D array (either real or complex) representing the input image plane or
        pupil plane to transform.
    nlamD : float or 2-tuple of floats (nlamDY, nlamDX)
        Size of desired output region in lambda / D units, assuming that the
        pupil fills the input array (corresponds to 'm' in
        Soummer et al. 2007 4.2). This is in units of the spatial frequency that
        is just Nyquist sampled by the input array.) If given as a tuple,
        interpreted as (nlamDY, nlamDX).
    npix : int or 2-tuple of ints (npixY, npixX)
        Number of pixels per side side of destination plane array (corresponds
        to 'N_B' in Soummer et al. 2007 4.2). This will be the # of pixels in
        the image plane for a forward transformation, in the pupil plane for an
        inverse. If given as a tuple, interpreted as (npixY, npixX).
    inverse : bool, optional
        Is this a forward or inverse transformation? (Default is False,
        implying a forward transformation.)
    centering : {'FFTSTYLE', 'SYMMETRIC', 'ADJUSTABLE'}, optional
        What type of centering convention should be used for this FFT?

        * ADJUSTABLE (the default) For an output array with ODD size n,
          the PSF center will be at the center of pixel (n-1)/2. For an output
          array with EVEN size n, the PSF center will be in the corner between
          pixel (n/2-1, n/2-1) and (n/2, n/2)
        * FFTSTYLE puts the zero-order term in a single pixel.
        * SYMMETRIC spreads the zero-order term evenly between the center
          four pixels

    offset : 2-tuple of floats (offsetY, offsetX)
        For ADJUSTABLE-style transforms, an offset in pixels by which the PSF
        will be displaced from the central pixel (or cross). Given as
        (offsetY, offsetX).
    """
    centering = centering.upper()

    try:
        npupY, npupX = len(plane), len(plane[0])
    except TypeError as e:
        raise TypeError("The `plane` argument must be an iterable of "
                         "iterables of the same length (e.g. 2D array)")

    slice_lengths = map(len, plane)
    assert len(set(slice_lengths)) == 1, (
        "Second dimension doesn't have the same length for all rows"
    )

    try:
        npixY, npixX = map(int, npix)
    except TypeError:
        # only a single, scalar npix value supplied (square array)
        try:
            npixY = npixX = int(npix)
        except TypeError:
            raise TypeError(
                "'npix' must be supplied as a scalar (for square arrays) or "
                "as a 2-tuple of ints (npixY, npixX)"
            )

    # make sure these are integer values
    if npixX != int(npixX) or npixY != int(npixY):
        raise TypeError("'npix' must be supplied as integer value(s)")

    try:
        nlamDY, nlamDX = float(nlamD[0]), float(nlamD[1])
    except TypeError:
        # only a single, scalar npix value supplied
        try:
            nlamDY = nlamDX = float(nlamD)
        except TypeError:
            raise TypeError(
                "'nlamD' must be supplied as a scalar (for square arrays) or "
                "as a 2-tuple of floats (nlamDY, nlamDX)"
            )

    if offset is not None and centering != constants.ADJUSTABLE:
        raise ValueError("Cannot provide an offset to non-adjustable centering")
    elif offset is None:
        offsetY, offsetX = 0.0, 0.0
    else:
        try:
            offsetY, offsetX = map(float, offset)
        except ValueError:
            raise ValueError(
                "'offset' must be supplied as a 2-tuple with "
                "(y_offset, x_offset) as floating point values"
            )

    # In the following: X and Y are coordinates in the input plane
    #                   U and V are coordinates in the output plane

    if inverse:
        dX = nlamDX / float(npupX)
        dY = nlamDY / float(npupY)
        dU = 1.0 / float(npixX)
        dV = 1.0 / float(npixY)
    else:
        dU = nlamDX / float(npixX)
        dV = nlamDY / float(npixY)
        dX = 1.0 / float(npupX)
        dY = 1.0 / float(npupY)

    if centering == FFTSTYLE:
        Xs = [(i - npupX / 2.0) * dX for i in xrange(npupX)]
        Ys = [(i - npupY / 2.0) * dY for i in xrange(npupY)]
        Us = [(i - npixX / 2.0) * dU for i in xrange(npixX)]
        Vs = [(i - npixY / 2.0) * dV for i in xrange(npixY)]
    elif centering == ADJUSTABLE:
        Xs = [(i - npupX / 2.0 - offsetX + 0.5) * dX for i in xrange(npupX)]
        Ys = [(i - npupY / 2.0 - offsetY + 0.5) * dY for i in xrange(npupY)]
        Us = [(i - npixX / 2.0 - offsetX + 0.5) * dU for i in xrange(npixX)]
        Vs = [(i - npixY / 2.0 - offsetY + 0.5) * dV for i in xrange(npixY)]
    elif centering == SYMMETRIC:
        Xs = [(i - npupX / 2.0 + 0.5) * dX for i in xrange(npupX)]
        Ys = [(i - npupY / 2.0 + 0.5) * dY for i in xrange(npupY)]
        Us = [(i - npixX / 2.0 + 0.5) * dU for i in xrange(npixX)]
        Vs = [(i - npixY / 2.0 + 0.5) * dV for i in xrange(npixY)]
    else:
        raise ValueError("Invalid centering style")

    XU = [[a * b for b in Us] for a in Xs]  # outer(Xs, Us)
    YV = [[a * b for b in Vs] for a in Ys]  # outer(Ys, Vs)

    if inverse:
        expYV = []
        for row in YV:
            new_row = []
            for val in row:
                new_row.append(exp(-2.0 * math.pi * -1.0j * val))
            expYV.append(new_row)
        expYV_T = list(zip(*expYV))  # transpose nested lists

        expXU = []
        for row in XU:
            new_row = []
            for val in row:
                new_row.append(exp(-2.0 * math.pi * -1.0j * val))
            expXU.append(new_row)
        t1 = matrix_multiply(expYV, plane)
        t2 = matrix_multiply(t1, expXU)
    else:
        expXU = []
        for row in XU:
            new_row = []
            for val in row:
                new_row.append(exp(-2.0 * math.pi * 1.0j * val))
            expXU.append(new_row)

        expYV = []
        for row in YV:
            new_row = []
            for val in row:
                new_row.append(exp(-2.0 * math.pi * 1.0j * val))
            expYV.append(new_row)
        expYV_T = list(zip(*expYV))  # transpose nested lists
        expYV = np.asarray(expYV_T)
        t1 = matrix_multiply(expYV, plane)
        t2 = matrix_multiply(t1, expXU)

    norm_coeff = math.sqrt((nlamDY * nlamDX) / (npupY * npupX * npixY * npixX))
    # element-wise multiply with a lambda and list comp
    result = [map(lambda x: norm_coeff * x, row) for row in t2]
    return result

def matrix_idft(*args, **kwargs):
    kwargs['inverse'] = True
    return matrix_dft(*args, **kwargs)

matrix_idft.__doc__ = matrix_dft.__doc__.replace(
    'Perform a matrix discrete Fourier transform',
    'Perform an inverse matrix discrete Fourier transform'
)
