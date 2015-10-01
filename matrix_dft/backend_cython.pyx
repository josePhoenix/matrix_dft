# cython: profile=True
from __future__ import division
import numpy as np
import cython
cimport numpy as np

# For now, only implement for complex128
DTYPE = np.complex128
ctypedef np.complex128_t DTYPE_t


from .constants import (CENTERING_CHOICES, FFTSTYLE,
                        SYMMETRIC, ADJUSTABLE)

def matrix_multiply(matrix_x, matrix_y):
    return [
        [sum(a * b for a, b in zip(x_row, y_col)) for y_col in zip(*matrix_y)]
        for x_row in matrix_x
    ]

# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef inline _matrix_multiply(np.ndarray[DTYPE_t, ndim=2] matrix_x, np.ndarray[DTYPE_t, ndim=2] matrix_y):
#     cdef int matrix_x_rows = matrix_x.shape[0]
#     cdef int matrix_x_cols = matrix_x.shape[1]
#     cdef int matrix_y_rows = matrix_y.shape[0]
#     cdef int matrix_y_cols = matrix_y.shape[1]
#     cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros((matrix_x_rows, matrix_y_cols), dtype=DTYPE)
#     cdef int i, j, k
#     cdef DTYPE_t cell_val
#     for i in range(matrix_x_rows):
#         for j in range(matrix_y_cols):
#             cell_val = 0.0
#             for k in range(matrix_y_rows):
#                 cell_val += matrix_x[i, k] * matrix_y[k, j]
#             result[i, j] = cell_val
#     return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline _matrix_multiply(np.ndarray[DTYPE_t, ndim=2] matrix_x, np.ndarray[DTYPE_t, ndim=2] matrix_y):
    cdef int matrix_x_rows = matrix_x.shape[0]
    cdef int matrix_x_cols = matrix_x.shape[1]
    cdef int matrix_y_rows = matrix_y.shape[0]
    cdef int matrix_y_cols = matrix_y.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] result = np.zeros((matrix_x_rows, matrix_y_cols), dtype=DTYPE)
    cdef int i, j, k
    for i in range(matrix_x_rows):
        for k in range(matrix_y_rows):  # i, j, k -> i, k, j to exploit locality
            for j in range(matrix_y_cols):
                result[i, j] = result[i, j] + matrix_x[i, k] * matrix_y[k, j]
    return result


# @cython.boundscheck(False)
# @cython.wraparound(False)
# cdef np.ndarray[DTYPE_t, ndim=2] matmatprod(
#     np.ndarray[DTYPE_t, ndim=2] A,
#     np.ndarray[DTYPE_t, ndim=2] B):
#     '''
#     Matrix-matrix multiplication.
#     '''
#     cdef:
#         int i, j, k
#         int A_n = A.shape[0]
#         int A_m = A.shape[1]
#         int B_n = B.shape[0]
#         int B_m = B.shape[1]
#         np.ndarray[DTYPE_t, ndim=2] C
#
#     # Are matrices conformable?
#     assert A_m == B_n, \
#         'Non-conformable shapes.'
#
#     # Initialize the results matrix.
#     C = np.zeros((A_n, B_m))
#     for i in xrange(A_n):
#         for j in xrange(B_m):
#             for k in xrange(A_m):
#                 C[i, j] += A[i, k] * B[k, j]
#     return C

def _matrix_dft(np.ndarray plane,
                double nlamDY, double nlamDX,
                int npixY, int npixX,
                double offsetY, double offsetX,
                bint inverse):
    cdef int npupY = plane.shape[0]
    cdef int npupX = plane.shape[1]

    cdef double dX, dY, dU, dV
    if inverse:
        dX = nlamDX / <double>npupX
        dY = nlamDY / <double>npupY
        dU = 1.0 / <double>npixX
        dV = 1.0 / <double>npixY
    else:
        dU = nlamDX / <double>npixX
        dV = nlamDY / <double>npixY
        dX = 1.0 / <double>npupX
        dY = 1.0 / <double>npupY

    cdef np.ndarray Xs = np.arange(npupX, dtype=DTYPE)
    Xs = (Xs - (<double>npupX / 2.0) - offsetX) * dX
    cdef np.ndarray Ys = (np.arange(npupY, dtype=DTYPE) - float(npupY) / 2.0 - offsetY) * dY

    cdef np.ndarray Us = (np.arange(npixX, dtype=DTYPE) - float(npixX) / 2.0 - offsetX) * dU
    cdef np.ndarray Vs = (np.arange(npixY, dtype=DTYPE) - float(npixY) / 2.0 - offsetY) * dV

    cdef np.ndarray XU = np.outer(Xs, Us)
    cdef np.ndarray YV = np.outer(Ys, Vs)
    cdef np.ndarray expYV, expXU, t1, t2

    if inverse == 1:
        expYV = np.exp(-2.0 * np.pi * -1j * YV).T
        expXU = np.exp(-2.0 * np.pi * -1j * XU)
        t1 = _matrix_multiply(expYV, plane)
        t2 = _matrix_multiply(t1, expXU)
    else:
        expXU = np.exp(-2.0 * np.pi * 1j * XU)
        expYV = np.exp(-2.0 * np.pi * 1j * YV).T
        t1 = _matrix_multiply(expYV, plane)
        t2 = _matrix_multiply(t1, expXU)
    assert npupY != 0, "npupY == 0?"
    assert npupX != 0, "npupX == 0?"

    assert nlamDY != 0, "nlamDY == 0?"
    assert nlamDX != 0, "nlamDX == 0?"

    assert npixY != 0, "npixY == 0?"
    assert npixX != 0, "npixX == 0?"
    cdef double numerator = (nlamDY * nlamDX)
    cdef double denominator = (<double>npupY * <double>npupX * <double>npixY * <double>npixX)
    assert denominator != 0
    cdef double norm_coeff = np.sqrt(numerator / denominator)
    return norm_coeff * t2

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
    if not centering in CENTERING_CHOICES:
        raise ValueError("Invalid centering style")

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

    if offset is not None and centering != ADJUSTABLE:
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
    if centering in (ADJUSTABLE, SYMMETRIC):
        # 0.5 pixel offset for ADJUSTABLE and SYMMETRIC, vs. 0 for FFTSTYLE
        # subtracted here because offsetX and offsetY are subtracted later,
        # and we want to *add* 0.5 px
        offsetX -= 0.5
        offsetY -= 0.5

    plane = np.asarray(plane, dtype=np.complex128)
    return _matrix_dft(plane, nlamDY, nlamDX, npixY, npixX, offsetY, offsetX, inverse)

def matrix_idft(*args, **kwargs):
    """Perform an inverse matrix discrete Fourier transform with selectable
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
    kwargs['inverse'] = True
    return matrix_dft(*args, **kwargs)
