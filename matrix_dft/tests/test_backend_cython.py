import numpy as np

import os
from .. import constants
from .. import backend_cython
from . import utils

import logging
_log = logging.getLogger('matrixdft')

def test_MFT_flux_conservation(centering=constants.FFTSTYLE, precision=0.01):
    """
    Test that MFTing a circular aperture is properly normalized to be
    flux conserving.

    This test is limited primarily by how finely the arrays are
    sampled. The function implements tests to better than 1% or 0.1%,
    selectable using the 'precision' argument.

    Parameters
    -----------

    outdir : path
        Directory path to output diagnostic FITS files. If not
        specified, files will not be written.
    precision : float, either 0.01 or 0.001
        How precisely to expect flux conservation; it will not be
        strictly 1.0 given any finite array size. This function uses
        predetermined MFT array sizes based on the desired precision
        level of the test.
    """

    # Set up constants for either a more precise test or a less precise but much
    # faster test:
    print("Testing MFT flux conservation for centering = "+centering)
    if precision ==0.001:
        npupil = 800
        npix = 4096
        u = 400    # of lam/D. Must be <= the Nyquist frequency of the pupil sampling or there
                   #           will be aliased copies of the PSF.
    elif precision==0.01:
        npupil = 400
        npix = 2048
        u = 200    # of lam/D. Must be <= the Nyquist frequency of the pupil sampling or there
                   #           will be aliased copies of the PSF.
    else:
        raise NotImplementedError('Invalid value for precision.')

    # Create pupil
    ctr = (float(npupil)/2.0, float(npupil)/2.0 )
    pupil = utils.makedisk(s=(npupil, npupil), c=ctr, r=float(npupil)/2.0001, t=np.float64, grey=0)
    pupil /= np.sqrt(pupil.sum())

    # MFT setup style and execute
    a = backend_cython.matrix_dft(pupil, u, npix, centering=centering)

    pre = (np.abs(pupil)**2).sum()    # normalized area of input pupil, should be 1 by construction
    post = (np.abs(a)**2).sum()       #
    ratio = post / pre
    print "Pre-FFT  total: "+str( pre)
    print "Post-FFT total: "+str( post )
    print "Ratio:          "+str( ratio)

    assert np.abs(1.0 - ratio) < precision

def test_MFT_fluxconsv_all_types(centering=None, **kwargs):
    for centering_type in constants.CENTERING_CHOICES:
        test_MFT_flux_conservation(centering=centering_type, **kwargs)

def test_DFT_rect(centering=constants.FFTSTYLE, npix=None, sampling=10.,
                  nlamd=None):
    """
    Test matrix DFT, including non-square arrays, in both the
    forward and inverse directions.

    This is an exact equivalent (in Python) of Marshall Perrin's
    test_matrix_DFT in matrix_dft.pro (in IDL)
    They should give identical results. However, this function doesn't actually
    check that since that would require having IDL...
    Instead it just checks that the sizes of the output arrays
    are as requested.

    """

    _log.info("Testing DFT, style = "+centering)

    npupil = 156
    pctr = int(npupil / 2)
    s = (npupil, npupil)

    # make things rectangular:
    if nlamd is None and npix is None:
        nlamd = (10, 20)
        npix = [val * sampling for val in nlamd]
    elif npix is None:
        npix = [val * sampling for val in nlamd]
    elif nlamd is None:
        nlamd = [val / sampling for val in npix]
    u = nlamd
    _log.info("Requested sampling in pixels: "+str(npix))
    _log.info("Requested sampling in lam/D units: "+str(u))

    ctr = (float(npupil)/2.0 , float(npupil)/2.0)
    pupil = utils.makedisk(s=s, c=ctr, r=float(npupil)/2.0001, t=np.float64, grey=0)

    pupil[0:60, 0:60] = 0
    pupil[0:10] = 0

    pupil /= np.sqrt(pupil.sum())

    _log.info("performing {} MFT with pupil shape: {} "
              "nlamd: {} npix: {}".format(centering, pupil.shape, nlamd, npix))
    a = backend_cython.matrix_dft(pupil, nlamd, npix, centering=centering)

    _log.info('Shape of MFT result: '+str(a.shape))
    assert( a.shape[0] == npix[0] )
    assert( a.shape[1] == npix[1] )

    pre = (np.abs(pupil) ** 2).sum()
    post = (np.abs(a) ** 2).sum()
    ratio = post / pre

    # multiply post by this to make them equal
    calcr = 1. / (1.0 * u[0] * u[1] * npix[0] * npix[1])
    _log.info( "Pre-FFT  total: {}".format(pre))
    _log.info( "Post-FFT total: {}".format(post))
    _log.info( "Ratio:          {}".format(ratio))

    asf = a.real.copy()
    cpsf = a * a.conjugate()
    psf = cpsf.real.copy()

    # Inverse transform:
    pupil2 = backend_cython.matrix_idft(a, u, npupil, centering=centering)
    pupil2r = (pupil2 * pupil2.conjugate()).real

    assert(pupil2.shape[0] == pupil.shape[0])
    assert(pupil2.shape[1] == pupil.shape[1])

    _log.info("Post-inverse FFT total: {}".format(np.abs(pupil2r).sum()))
    _log.info("Post-inverse pupil max: {}".format(pupil2r.max()))


def test_DFT_rect_adj():
    """
    Repeat DFT rectangle check, but for adjustable FFT centering
    """
    test_DFT_rect(centering='ADJUSTABLE')

def test_DFT_center(npix=100):
    centering='ADJUSTABLE'

    npupil = 156
    pctr = int(npupil/2)
    npix = 1024
    u = 100    # of lam/D
    s = (npupil, npupil)

    ctr = (float(npupil)/2.0, float(npupil)/2.0 )
    pupil = utils.makedisk(s=s, c=ctr, r=float(npupil)/2.0001, t=np.float64, grey=0)

    pupil /= np.sqrt(pupil.sum())

    a = backend_cython.matrix_dft(pupil, u, npix, centering=centering)

    pre = (np.abs(pupil) ** 2).sum()
    post = (np.abs(a) ** 2).sum()
    ratio = post / pre

    # multiply post by this to make them equal
    calcr = 1. / (u ** 2 * npix ** 2)
    _log.info( "Pre-FFT  total: {}".format(pre))
    _log.info( "Post-FFT total: {}".format(post))
    _log.info( "Ratio:          {}".format(ratio))

    utils.complexinfo(a, extra_message="mft1 asf")
    asf = a.real.copy()
    cpsf = a * a.conjugate()
    psf = cpsf.real.copy()

def test_DFT_rect_fov_sampling(fov_npix=(500,1000)):
    """ Test that we can create a rectangular FOV which nonetheless
    is properly sampled in both the X and Y directions as desired.
    In this case specifically we test that we can get the a symmetric
    PSF (same pixel scale in both X and Y) even when the overall FOV
    is rectangular. This tests some of the low level normalizations and
    scaling factors within the matrixDFT code.


    """
    # pupil array: 200 wide, 100 high
    npix_pupil_x = 200
    npix_pupil_y = 100
    # 1 meter diameter aperture centered at y=50, x=100
    x_center, y_center = 100, 50
    # 250 pixel radius -> 50 px / 1 m
    meters_to_pix = 50.0 / 1.0
    radius = 0.5 * meters_to_pix # 0.5 m * 500 px/m
    pupil = utils.makedisk(s=(npix_pupil_y, npix_pupil_x), c=(x_center, y_center), r=radius)

    # total extent of pupil array in meters:
    extent_x, extent_y = npix_pupil_x / meters_to_pix, npix_pupil_y / meters_to_pix
    wavelength = 1e-6
    _RADIANStoARCSEC = 206265
    lamD_y = wavelength / extent_y * _RADIANStoARCSEC
    lamD_x = wavelength / extent_x * _RADIANStoARCSEC
    det_fov_lamD = 1.0 / lamD_y, 2.0 / lamD_x

    # MFT pupil -> img
    img = backend_cython.matrix_dft(
        pupil,
        (extent_y / lamD_y, extent_x / lamD_x),
        (npix_pupil_y, npix_pupil_x),
        centering=constants.ADJUSTABLE
    )
    img = np.asarray(img)
    intensity = np.abs(img)

    delta = 25  # take center - 25 px to center + 25 px for comparison
    x_from, x_to = npix_pupil_x / 2 - delta, npix_pupil_x / 2 + delta
    cut_h = intensity[npix_pupil_y / 2,x_from:x_to]
    y_from, y_to = npix_pupil_y / 2 - delta, npix_pupil_y / 2 + delta
    cut_v = intensity[y_from:y_to, npix_pupil_x / 2]

    assert(np.all(np.abs(cut_h-cut_v) < 1e-12))

def test_check_invalid_centering():
    """ intentionally invalid CENTERING option to test the error message part of the code.
    """
    try:
        import pytest
    except:
        poppy._log.warning('Skipping test test_check_invalid_centering because pytest is not installed.')
        return # We can't do this test if we don't have the pytest.raises function.

    # MFT setup style and execute
    plane = np.zeros((10, 10))
    with pytest.raises(ValueError) as excinfo:
        mft = backend_cython.matrix_dft(plane, 10, 10, centering='some garbage value')
    assert excinfo.value.message == "Invalid centering style"

def test_parity_MFT_forward_inverse(npix=512, centering=constants.ADJUSTABLE):
    """ Test that transforming from a pupil, to an image, and back to the pupil
    leaves you with the same pupil as you had in the first place.

    In other words it doesn't flip left/right or up/down etc.

    See https://github.com/mperrin/webbpsf/issues/35

    **  See also: test_fft.test_parity_FFT_forward_inverse() for a  **
    **  parallel function to this.                                  **


    """

    pupil = utils.make_parity_test_array(npix)
    nlamD = npix
    mftout = backend_cython.matrix_dft(pupil, nlamD, npix, centering=centering)
    imftout = backend_cython.matrix_idft(mftout, nlamD, npix, centering=centering)
    imftout = np.asarray(imftout)

    # for checking the overall parity it's sufficient to check the intensity.
    # we can have arbitrarily large differences in phase for regions with
    # intensity =0, so don't check the complex field or phase here.
    absdiff = np.abs(imftout) - np.abs(pupil)
    max_absdiff = np.max(absdiff)
    assert (max_absdiff < 1e-10)

def test_MFT_FFT_equivalence():
    """ Test that the MFT transform is numerically equivalent to the
    FFT, if calculated on the correct sampling. """

    centering = 'FFTSTYLE' # needed if you want near-exact agreement!

    imgin = utils.make_parity_test_array(npix=256)

    npix = imgin.shape
    nlamD = np.asarray(imgin.shape)
    mftout = backend_cython.matrix_dft(imgin, nlamD, npix, centering=centering)
    mftout = np.asarray(mftout)

    fftout = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(imgin)))
    fftout /= np.sqrt(imgin.shape[0] * imgin.shape[1])

    norm_factor = np.abs(mftout).sum()

    absdiff = np.abs(mftout-fftout) / norm_factor

    assert(np.all(absdiff < 1e-10))
