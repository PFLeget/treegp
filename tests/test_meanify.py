from __future__ import print_function
import warnings
import numpy as np
import copy
import treegp
import os
import fitsio
import glob

from treegp_test_helper import timer
from treegp_test_helper import get_correlation_length_matrix
from treegp_test_helper import make_2d_grf


def make_average(coord=None, gp=True):
    if coord is None:
        x = np.linspace(0, 2048, 10)
        x, y = np.meshgrid(x, x)
        x = x.reshape(len(x) ** 2)
        y = y.reshape(len(y) ** 2)
    else:
        x = coord[:, 0]
        y = coord[:, 1]

    average = 0.02 + 5e-8 * (x - 1024) ** 2 + 5e-8 * (y - 1024) ** 2
    params = copy.deepcopy(average)

    if gp:
        from scipy.spatial.distance import pdist, squareform

        dists = squareform(pdist(np.array([x, y]).T))
        cov = 0.03**2 * np.exp(-0.5 * dists**2 / 300.0**2)

        # avoids to print warning from numpy when generated uge gaussian random fields.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params += np.random.multivariate_normal([0] * len(x), cov)

    return np.array([x, y]).T, params


@timer
def test_meanify():
    np.random.seed(42)
    nfields = 300
    ndata = 500
    coords = []
    fields = []

    for n in range(nfields):
        x = np.random.uniform(0, 2048, size=ndata)
        y = np.random.uniform(0, 2048, size=ndata)
        coord = np.array([x, y]).T
        coord, field = make_average(coord=coord, gp=True)
        coords.append(coord)
        fields.append(field)

    meanify = treegp.meanify(bin_spacing=40.0, statistics="mean")
    for n in range(nfields):
        meanify.add_field(coords[n], fields[n])
    meanify.meanify()
    meanify.save_results(name_output=os.path.join("outputs", "mean_gp_stat_mean.fits"))
    coord, param = make_average(coord=meanify.coords0, gp=False)
    np.testing.assert_allclose(param, meanify.params0, atol=2e-1)


@timer
def test_gpinterp_meanify():
    optimizer = ["log-likelihood", "anisotropic"]
    npoints = [600, 2000]
    noise = 0.01
    sigma = 2.0
    size = 0.5
    g1 = 0.2
    g2 = 0.2
    ker = "AnisotropicRBF"

    # Generate 2D gaussian random fields.
    L = get_correlation_length_matrix(size, g1, g2)
    invL = np.linalg.inv(L)
    kernel = "%f**2*%s" % ((sigma, ker))
    kernel += "(invLam={0!r})".format(invL)
    kernel_skl = treegp.eval_kernel(kernel)

    for n, opt in enumerate(optimizer):
        x, y, y_err = make_2d_grf(kernel_skl, noise=noise, seed=44, npoints=npoints[n])
        # add mean function
        coords0, y0 = make_average(coord=x, gp=False)
        y += y0

        # Do gp interpolation without hyperparameters
        # fitting (truth is put initially).
        gp = treegp.GPInterpolation(
            kernel=kernel,
            optimizer=opt,
            normalize=True,
            nbins=21,
            min_sep=0.0,
            max_sep=3.0,
            p0=[0.5, 0, 0],
            average_fits=os.path.join("inputs", "mean_gp_stat_mean.fits"),
        )
        gp.initialize(x, y, y_err=y_err)
        gp.solve()
        # test if found hyperparameters are close the true hyperparameters.
        np.testing.assert_allclose(kernel_skl.theta, gp.kernel.theta, atol=5e-1)

        # Predict at same position as the simulated data.
        # Predictions are strictily equal to the input data
        # in the case of no noise. With noise you should expect
        # to have a pull distribution with mean function arround 0
        # with a std<1 (you use the same data to train and validate, and
        # the data are well sample compared to the input correlation
        # length).
        y_predict, y_cov = gp.predict(x, return_cov=True)
        y_std = np.sqrt(np.diag(y_cov))
        pull = y - y_predict
        pull /= np.sqrt(y_err**2 + y_std**2)
        mean_pull = np.mean(pull)
        std_pull = np.std(pull)

        # Test that mean of the pull is close to zeros and std of the pull bellow 1.
        np.testing.assert_allclose(
            0.0, mean_pull, atol=3.0 * (std_pull) / np.sqrt(npoints[n])
        )
        if std_pull > 1.0:
            raise ValueError(
                "std_pull is > 1. Current value std_pull = %f" % (std_pull)
            )


@timer
def test_meanify_streaming():
    """Test that streaming mode (with bounds) matches legacy mode."""

    # --- SETUP DATA ---
    np.random.seed(42)
    N = 20000
    bounds = (0, 1000, 0, 1000)
    spacing = 53.7  # Arbitrary spacing to test rounding logic

    # Random data + edge cases
    coords = np.random.uniform(0, 1000, size=(N, 2))
    edge_cases = np.array([[0, 0], [1000, 1000], [500, 1000], [1000, 500]])
    coords = np.vstack([coords, edge_cases])
    params = np.random.normal(100, 10, size=len(coords))

    # --- RUN LEGACY (no bounds -> uses scipy binned_statistic) ---
    legacy = treegp.meanify(bin_spacing=spacing, statistics="mean")
    legacy.add_field(coords, params)
    legacy.meanify(
        lu_min=bounds[0], lu_max=bounds[1], lv_min=bounds[2], lv_max=bounds[3]
    )
    assert not legacy._use_streaming, "Legacy mode should not use streaming"

    # --- RUN STREAMING (with bounds -> uses O(1) accumulators) ---
    stream = treegp.meanify(bin_spacing=spacing, statistics="mean", bounds=bounds)
    stream.add_field(coords, params)
    stream.meanify()
    assert stream._use_streaming, "Streaming mode should use streaming"

    # --- ASSERTIONS ---

    # 1. Check Shapes
    assert (
        legacy._xedge.shape == stream._xedge.shape
    ), f"X-Edge Shape mismatch: {legacy._xedge.shape} vs {stream._xedge.shape}"
    assert (
        legacy._yedge.shape == stream._yedge.shape
    ), f"Y-Edge Shape mismatch: {legacy._yedge.shape} vs {stream._yedge.shape}"
    assert (
        legacy._average.shape == stream._average.shape
    ), f"Average Map Shape mismatch: {legacy._average.shape} vs {stream._average.shape}"

    # 2. Check Edges (Strict Equality)
    np.testing.assert_allclose(
        legacy._xedge, stream._xedge, rtol=1e-12, err_msg="X-Edge values do not match"
    )
    np.testing.assert_allclose(
        legacy._yedge, stream._yedge, rtol=1e-12, err_msg="Y-Edge values do not match"
    )

    # 3. Check computed values match
    np.testing.assert_allclose(
        legacy.params0,
        stream.params0,
        rtol=1e-10,
        err_msg="params0 values do not match",
    )
    np.testing.assert_allclose(
        legacy.coords0,
        stream.coords0,
        rtol=1e-10,
        err_msg="coords0 values do not match",
    )


@timer
def test_meanify_median_uses_legacy():
    """Test that median statistics always uses legacy mode (even with bounds)."""
    np.random.seed(42)
    coords = np.random.uniform(0, 1000, size=(1000, 2))
    params = np.random.normal(100, 10, size=1000)

    # Median with bounds should still use legacy (streaming only supports mean)
    m = treegp.meanify(
        bin_spacing=100.0, statistics="median", bounds=(0, 1000, 0, 1000)
    )
    assert not m._use_streaming, "Median should use legacy mode"
    m.add_field(coords, params)
    m.meanify()
    assert m.params0 is not None


@timer
def test_meanify_backward_compat():
    """Test MeanifyStream alias for backward compatibility."""
    np.random.seed(42)
    coords = np.random.uniform(0, 1000, size=(1000, 2))
    params = np.random.normal(100, 10, size=1000)

    # MeanifyStream is now an alias - should work with bounds
    stream = treegp.MeanifyStream(bin_spacing=100.0, bounds=(0, 1000, 0, 1000))
    stream.add_field(coords, params)
    stream.meanify()
    assert stream._use_streaming, "MeanifyStream should use streaming mode"
    assert stream.params0 is not None


@timer
def test_robust_helpers():
    """Test the biweight and MAD-clipped median helpers used by meanify."""
    from treegp.meanify import biweight, median_clipped

    # Empty sample -> nan, without any warning (binned_statistic_2d probes
    # the callable with an empty array for bins without data).
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(biweight(np.array([])))
        assert np.isnan(median_clipped(np.array([])))

    # Degenerate samples.
    assert biweight(np.array([3.0])) == 3.0
    assert median_clipped(np.array([3.0])) == 3.0
    np.testing.assert_allclose(biweight(np.ones(5) * 7.0), 7.0)
    np.testing.assert_allclose(median_clipped(np.ones(5) * 7.0), 7.0)

    # Gaussian sample with 10% of strong asymmetric outliers.
    np.random.seed(42)
    sample = np.random.normal(0.0, 1.0, size=1000)
    sample[:100] += 8.0
    m_mean = np.mean(sample)
    m_median = np.median(sample)
    m_biweight = biweight(sample)
    m_clipped = median_clipped(sample)
    assert abs(m_mean) > 0.6
    assert abs(m_median) > 0.1
    assert abs(m_biweight) < 0.1
    assert abs(m_clipped) < 0.1
    assert abs(m_biweight) < abs(m_median)
    assert abs(m_clipped) < abs(m_median)

    # No outliers: both are close to the mean and the clipping is mild.
    sample = np.random.normal(0.0, 1.0, size=1000)
    np.testing.assert_allclose(biweight(sample), np.mean(sample), atol=0.05)
    np.testing.assert_allclose(median_clipped(sample), np.mean(sample), atol=0.1)

    # Unknown statistic raises.
    try:
        treegp.meanify(statistics="not_a_stat")
    except ValueError:
        pass
    else:
        raise AssertionError("meanify should reject an unknown statistic")


@timer
def test_meanify_robust_statistics():
    """Test biweight and median_clipped statistics in meanify."""
    np.random.seed(42)
    npoints = 20000
    coords = np.random.uniform(0, 1000, size=(npoints, 2))
    params = np.random.normal(100.0, 10.0, size=npoints)
    # 10% of strong asymmetric outliers.
    params[: npoints // 10] += 100.0

    def run(statistics):
        m = treegp.meanify(bin_spacing=100.0, statistics=statistics)
        m.add_field(coords, params)
        m.meanify(lu_min=0, lu_max=1000, lv_min=0, lv_max=1000)
        return m

    m_mean = run("mean")
    err_mean = np.max(np.abs(m_mean.params0 - 100.0))
    assert err_mean > 5.0  # mean is pulled by the outliers

    for statistics in ["biweight", "median_clipped"]:
        # Streaming is only supported for "mean", even when bounds are given.
        m = treegp.meanify(
            bin_spacing=100.0, statistics=statistics, bounds=(0, 1000, 0, 1000)
        )
        assert not m._use_streaming, "%s should use legacy mode" % statistics

        m = run(statistics)
        assert np.all(np.isfinite(m.params0))
        assert np.all(m.wrms0 == 0.0)
        assert len(m.params0) == len(m_mean.params0)
        np.testing.assert_allclose(m.coords0, m_mean.coords0)
        np.testing.assert_allclose(m.params0, 100.0, atol=err_mean / 2.0)
        m.save_results(
            name_output=os.path.join("outputs", "mean_gp_stat_%s.fits" % statistics)
        )

    # Tuning parameters are forwarded.
    m = treegp.meanify(statistics="median_clipped", clip_nsigma=2.0, clip_maxiters=2)
    assert m.clip_nsigma == 2.0 and m.clip_maxiters == 2
    m = treegp.meanify(statistics="biweight", biweight_c=9.0)
    assert m.biweight_c == 9.0


def _quadratic_truth(coords):
    return (
        0.02 + 5e-8 * (coords[:, 0] - 500.0) ** 2 + 5e-8 * (coords[:, 1] - 500.0) ** 2
    )


@timer
def test_meanify_counts_and_read_results():
    """Per-bin counts in both modes, and round trip through the fits file."""
    np.random.seed(42)
    npoints = 5000
    coords = np.random.uniform(0, 1000, size=(npoints, 2))
    params = np.random.normal(100.0, 10.0, size=npoints)
    params[:10] = np.nan  # dropped by add_field
    bounds = (0, 1000, 0, 1000)

    legacy = treegp.meanify(bin_spacing=100.0, statistics="biweight")
    legacy.add_field(coords, params)
    legacy.meanify(*bounds)

    stream = treegp.meanify(bin_spacing=100.0, statistics="mean", bounds=bounds)
    stream.add_field(coords, params)
    stream.meanify()

    for m in [legacy, stream]:
        assert m._count.shape == m._average.shape
        assert m._count.sum() == npoints - 10
        assert m._count.dtype.kind == "i"
    np.testing.assert_array_equal(legacy._count, stream._count)

    # Round trip.
    name = os.path.join("outputs", "mean_gp_counts.fits")
    legacy.save_results(name_output=name)
    loaded = treegp.meanify.read_results(name)
    np.testing.assert_array_equal(loaded._count, legacy._count)
    np.testing.assert_allclose(loaded._average, legacy._average, equal_nan=True)
    np.testing.assert_allclose(loaded._wrms, legacy._wrms, equal_nan=True)
    np.testing.assert_allclose(loaded._u0, legacy._u0)
    np.testing.assert_allclose(loaded._v0, legacy._v0)
    np.testing.assert_allclose(loaded._xedge, legacy._xedge, atol=1e-9)
    np.testing.assert_allclose(loaded._yedge, legacy._yedge, atol=1e-9)
    np.testing.assert_allclose(loaded.coords0, legacy.coords0)
    np.testing.assert_allclose(loaded.params0, legacy.params0)
    np.testing.assert_allclose(loaded.wrms0, legacy.wrms0)
    np.testing.assert_allclose(loaded.bin_spacing, legacy._xedge[1] - legacy._xedge[0])

    # Files written before counts existed load with _count = None.
    legacy._count = None
    name_old = os.path.join("outputs", "mean_gp_nocounts.fits")
    legacy.save_results(name_output=name_old)
    with fitsio.FITS(name_old) as f:
        assert "_COUNT" not in f["average_solution"].get_colnames()
    loaded = treegp.meanify.read_results(name_old)
    assert loaded._count is None
    for kwargs in [dict(min_count=1), dict(weight_by_count=True)]:
        try:
            loaded.smooth(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("smooth should require counts for %s" % kwargs)
    loaded.smooth()  # unweighted smoothing still works
    assert np.all(np.isfinite(loaded.params0))


@timer
def test_meanify_smooth():
    """smooth() reduces the noise of the mean function on a quadratic field."""
    np.random.seed(42)
    npoints = 40000
    coords = np.random.uniform(0, 1000, size=(npoints, 2))
    params = _quadratic_truth(coords) + np.random.normal(0.0, 0.05, size=npoints)

    def run():
        m = treegp.meanify(bin_spacing=20.0, statistics="mean")
        m.add_field(coords, params)
        m.meanify(0, 1000, 0, 1000)
        return m

    m = run()
    nbins_raw = len(m.params0)
    err_raw = np.std(m.params0 - _quadratic_truth(m.coords0))
    average_raw = m._average.copy()

    out = m.smooth(window=5, order=2)
    assert out is m
    np.testing.assert_array_equal(m._average_raw, average_raw)
    assert m._average.shape == average_raw.shape
    assert len(m.params0) == len(m.coords0) == len(m.wrms0) == nbins_raw
    err_smooth = np.std(m.params0 - _quadratic_truth(m.coords0))
    assert err_smooth < 0.6 * err_raw
    # A second call keeps the original raw grid.
    m.smooth(window=3, order=1)
    np.testing.assert_array_equal(m._average_raw, average_raw)

    # min_count masks low-count bins, which stay empty unless fill_empty.
    m2 = run()
    min_count = 12
    expected = np.sum((m2._count >= min_count) & np.isfinite(m2._average))
    m2.smooth(window=5, order=2, min_count=min_count)
    # Masked bins stay empty; a few kept bins next to the border may also
    # lack enough valid neighbours for the fit and drop out.
    assert expected - 5 <= len(m2.params0) <= expected < nbins_raw
    kept = np.isfinite(m2._average)
    assert np.all(m2._count[kept] >= min_count)
    m3 = run()
    m3.smooth(window=5, order=2, min_count=min_count, fill_empty=True)
    assert len(m3.params0) >= 0.99 * nbins_raw
    assert np.all(np.isfinite(m3.wrms0))
    err_filled = np.std(m3.params0 - _quadratic_truth(m3.coords0))
    assert err_filled < 0.6 * err_raw

    # Count weighting runs and is also less noisy than the raw map.
    m4 = run()
    m4.smooth(window=5, order=2, weight_by_count=True)
    err_w = np.std(m4.params0 - _quadratic_truth(m4.coords0))
    assert err_w < 0.6 * err_raw

    # Chained read / smooth / save.
    name = os.path.join("outputs", "mean_gp_smooth.fits")
    m5 = run()
    m5.save_results(name_output=name)
    smoothed = treegp.meanify.read_results(name).smooth(window=5, order=2)
    np.testing.assert_allclose(
        smoothed.params0, run().smooth(window=5, order=2).params0
    )
    name2 = os.path.join("outputs", "mean_gp_smooth2.fits")
    smoothed.save_results(name_output=name2)
    reloaded = treegp.meanify.read_results(name2)
    np.testing.assert_allclose(reloaded.params0, smoothed.params0)

    # smooth() before meanify() is an error.
    try:
        treegp.meanify().smooth()
    except RuntimeError:
        pass
    else:
        raise AssertionError("smooth should require meanify() first")


if __name__ == "__main__":
    test_meanify()
    test_gpinterp_meanify()
    test_meanify_streaming()
    test_meanify_median_uses_legacy()
    test_meanify_backward_compat()
    test_robust_helpers()
    test_meanify_robust_statistics()
    test_meanify_counts_and_read_results()
    test_meanify_smooth()
