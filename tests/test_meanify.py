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
        x, y, y_err = make_2d_grf(kernel_skl, noise=noise, seed=42, npoints=npoints[n])
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
    legacy.meanify(lu_min=bounds[0], lu_max=bounds[1], lv_min=bounds[2], lv_max=bounds[3])
    assert not legacy._use_streaming, "Legacy mode should not use streaming"

    # --- RUN STREAMING (with bounds -> uses O(1) accumulators) ---
    stream = treegp.meanify(bin_spacing=spacing, statistics="mean", bounds=bounds)
    stream.add_field(coords, params)
    stream.meanify()
    assert stream._use_streaming, "Streaming mode should use streaming"

    # --- ASSERTIONS ---

    # 1. Check Shapes
    assert legacy._xedge.shape == stream._xedge.shape, (
        f"X-Edge Shape mismatch: {legacy._xedge.shape} vs {stream._xedge.shape}"
    )
    assert legacy._yedge.shape == stream._yedge.shape, (
        f"Y-Edge Shape mismatch: {legacy._yedge.shape} vs {stream._yedge.shape}"
    )
    assert legacy._average.shape == stream._average.shape, (
        f"Average Map Shape mismatch: {legacy._average.shape} vs {stream._average.shape}"
    )

    # 2. Check Edges (Strict Equality)
    np.testing.assert_allclose(
        legacy._xedge, stream._xedge, rtol=1e-12, err_msg="X-Edge values do not match"
    )
    np.testing.assert_allclose(
        legacy._yedge, stream._yedge, rtol=1e-12, err_msg="Y-Edge values do not match"
    )

    # 3. Check computed values match
    np.testing.assert_allclose(
        legacy.params0, stream.params0, rtol=1e-10, err_msg="params0 values do not match"
    )
    np.testing.assert_allclose(
        legacy.coords0, stream.coords0, rtol=1e-10, err_msg="coords0 values do not match"
    )


@timer
def test_meanify_median_uses_legacy():
    """Test that median statistics always uses legacy mode (even with bounds)."""
    np.random.seed(42)
    coords = np.random.uniform(0, 1000, size=(1000, 2))
    params = np.random.normal(100, 10, size=1000)

    # Median with bounds should still use legacy (streaming only supports mean)
    m = treegp.meanify(bin_spacing=100.0, statistics="median", bounds=(0, 1000, 0, 1000))
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


if __name__ == "__main__":
    test_meanify()
    test_gpinterp_meanify()
    test_meanify_streaming()
    test_meanify_median_uses_legacy()
    test_meanify_backward_compat()
