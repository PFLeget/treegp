from __future__ import print_function
import numpy as np
import treegp
import os
import fitsio

from treegp_test_helper import timer


def make_average_sky(ra, dec, gp=False):
    """Create a smooth mean function on the sky.

    :param ra: Right ascension in degrees
    :param dec: Declination in degrees
    :param gp: If True, add Gaussian random field noise
    :return: params array
    """
    # Simple quadratic function centered on (ra=180, dec=0)
    average = 0.02 + 1e-4 * ((ra - 180) ** 2 + dec**2)
    params = average.copy()

    if gp:
        # Add some random scatter
        params += np.random.normal(0, 0.01, size=len(ra))

    return params


@timer
def test_meanify_healpix_basic():
    """Test basic functionality of meanify_healpix."""
    np.random.seed(42)

    # Generate random sky coordinates in a patch
    n_points = 10000
    ra = np.random.uniform(170, 190, size=n_points)
    dec = np.random.uniform(-10, 10, size=n_points)
    coord = np.column_stack((ra, dec))

    # Generate parameter values
    params = make_average_sky(ra, dec, gp=True)

    # Create meanify_healpix instance
    mh = treegp.meanify_healpix(bin_spacing=120.0)

    # Add data
    mh.add_field(coord, params)

    # Compute mean
    mh.meanify()

    # Check outputs exist and have correct shapes
    assert hasattr(mh, "coords0")
    assert hasattr(mh, "params0")
    assert hasattr(mh, "wrms0")
    assert mh.coords0.shape[1] == 2
    assert len(mh.params0) == len(mh.coords0)
    assert len(mh.wrms0) == len(mh.coords0)

    # Check that we got some valid pixels
    assert len(mh.params0) > 0

    # Check that mean values are reasonable
    expected = make_average_sky(mh.coords0[:, 0], mh.coords0[:, 1], gp=False)
    np.testing.assert_allclose(expected, mh.params0, atol=0.05)


@timer
def test_meanify_healpix_nside():
    """Test nside parameter override."""
    np.random.seed(42)

    # Test with explicit nside
    mh = treegp.meanify_healpix(nside=64)
    assert mh.nside == 64

    # Test that non-power-of-2 raises error
    try:
        mh = treegp.meanify_healpix(nside=100)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


@timer
def test_meanify_healpix_multiple_fields():
    """Test adding multiple fields incrementally."""
    np.random.seed(42)

    mh = treegp.meanify_healpix(bin_spacing=240.0)

    # Add data in multiple chunks (simulating multiple exposures)
    n_fields = 10
    n_points = 1000

    for _ in range(n_fields):
        ra = np.random.uniform(175, 185, size=n_points)
        dec = np.random.uniform(-5, 5, size=n_points)
        coord = np.column_stack((ra, dec))
        params = make_average_sky(ra, dec, gp=True)
        mh.add_field(coord, params)

    mh.meanify()

    # Check that we accumulated data
    assert len(mh.params0) > 0

    # With more data, mean should be closer to true value
    expected = make_average_sky(mh.coords0[:, 0], mh.coords0[:, 1], gp=False)
    np.testing.assert_allclose(expected, mh.params0, atol=0.05)


@timer
def test_meanify_healpix_save_results():
    """Test saving results to FITS file."""
    np.random.seed(42)

    n_points = 5000
    ra = np.random.uniform(170, 190, size=n_points)
    dec = np.random.uniform(-10, 10, size=n_points)
    coord = np.column_stack((ra, dec))
    params = make_average_sky(ra, dec, gp=True)

    mh = treegp.meanify_healpix(bin_spacing=180.0)
    mh.add_field(coord, params)
    mh.meanify()

    # Save results
    output_file = os.path.join("outputs", "mean_gp_healpix_test.fits")
    mh.save_results(name_output=output_file)

    # Read back and verify
    assert os.path.exists(output_file)

    with fitsio.FITS(output_file) as f:
        data = f["average_solution"].read()
        assert "COORDS0" in data.dtype.names
        assert "PARAMS0" in data.dtype.names
        assert "WRMS0" in data.dtype.names
        assert "NSIDE" in data.dtype.names
        assert "PIXEL_SIZE_ARCSEC" in data.dtype.names

    # Clean up
    os.remove(output_file)


@timer
def test_meanify_healpix_nan_handling():
    """Test that NaN values are properly filtered."""
    np.random.seed(42)

    n_points = 1000
    ra = np.random.uniform(170, 190, size=n_points)
    dec = np.random.uniform(-10, 10, size=n_points)
    coord = np.column_stack((ra, dec))
    params = make_average_sky(ra, dec, gp=False)

    # Insert some NaN values
    params[::10] = np.nan

    mh = treegp.meanify_healpix(bin_spacing=240.0)
    mh.add_field(coord, params)
    mh.meanify()

    # Check no NaN in output
    assert np.all(np.isfinite(mh.params0))
    assert np.all(np.isfinite(mh.wrms0))


@timer
def test_meanify_healpix_empty():
    """Test behavior with no data."""
    mh = treegp.meanify_healpix(bin_spacing=120.0)
    mh.meanify()

    assert len(mh.coords0) == 0
    assert len(mh.params0) == 0
    assert len(mh.wrms0) == 0


@timer
def test_meanify_healpix_resolution():
    """Test that bin_spacing to nside conversion is reasonable."""
    # Approximate pixel size should be close to bin_spacing

    # 120 arcsec -> nside ~1024
    mh1 = treegp.meanify_healpix(bin_spacing=120.0)
    # Pixel size should be within factor of 2 of requested
    assert 60 < mh1.pixel_size_arcsec < 240

    # 60 arcsec -> nside ~2048
    mh2 = treegp.meanify_healpix(bin_spacing=60.0)
    assert mh2.nside > mh1.nside

    # 240 arcsec -> nside ~512
    mh3 = treegp.meanify_healpix(bin_spacing=240.0)
    assert mh3.nside < mh1.nside


if __name__ == "__main__":
    test_meanify_healpix_basic()
    test_meanify_healpix_nside()
    test_meanify_healpix_multiple_fields()
    test_meanify_healpix_save_results()
    test_meanify_healpix_nan_handling()
    test_meanify_healpix_empty()
    test_meanify_healpix_resolution()
