import numpy as np
import treegp
import copy

from treegp_test_helper import timer
from treegp_test_helper import get_correlation_length_matrix
from treegp_test_helper import make_2d_grf

from treegp.empirical_2pcf import (
    _shift_and_bin,
    _threshold,
    _corr2power,
    _power2corr,
    _apod,
    _adaptive_moments,
)


def make_elliptical_gaussian(npix, size, g1, g2):
    """Elliptical gaussian map in treecorr TwoD layout ([iy, ix]),
    centered on pixel npix//2, using the Leget et al. 2021 shear
    parametrization (size = major axis, in pixels)."""
    L = get_correlation_length_matrix(size, g1, g2)
    invL = np.linalg.inv(L)
    lag = np.arange(npix) - npix // 2
    dx, dy = np.meshgrid(lag, lag)
    arg = invL[0, 0] * dx**2 + 2.0 * invL[0, 1] * dx * dy + invL[1, 1] * dy**2
    return np.exp(-0.5 * arg)


def make_gp(npoints=2000, noise=0.3, white_noise=0.0, seed=42):
    """Generate a 2d GRF with a known anisotropic kernel and return
    an initialized and solved GPInterpolation using empirical-2pcf."""
    L = get_correlation_length_matrix(2.0, 0.2, 0.2)
    invLam = np.linalg.inv(L)
    kernel = 2.0**2 * treegp.AnisotropicRBF(invLam=invLam)
    X, y, y_err = make_2d_grf(kernel, noise=noise, seed=seed, npoints=npoints)
    gp = treegp.GPInterpolation(
        optimizer="empirical-2pcf",
        normalize=True,
        white_noise=white_noise,
        max_sep=6.0,
        pixel_size=0.5,
    )
    gp.initialize(X, y, y_err=y_err)
    gp.solve()
    return gp, X, y, y_err, noise


@timer
def test_empirical_2pcf_gp():
    gp, X, y, y_err, noise = make_gp()

    # No hyperparameters are fitted: the kernel is a tabulated
    # correlation function with an empty theta.
    assert isinstance(gp.kernel, treegp.EmpiricalCorrelationKernel)
    assert len(gp.kernel.theta) == 0

    # The zero lag of the cleaned correlation function is the
    # variance of the field (up to noise in the measured 2-pcf).
    np.testing.assert_allclose(gp.kernel.xi0, np.var(y) - noise**2, rtol=3e-1)

    y_predict, y_cov = gp.predict(X, return_cov=True)
    y_std = np.sqrt(np.diag(y_cov))

    # Check that the GP interpolation catches a good fraction of
    # the field variance.
    residuals = y - y_predict
    assert np.var(residuals) < 0.5 * np.var(y)

    # Pull distribution should have a mean of 0 and a std < 1
    # (as the interpolation is better than the noise).
    pull = residuals / np.sqrt(gp._y_err**2 + y_std**2)
    mean_pull = np.mean(pull)
    std_pull = np.std(pull)
    assert np.abs(mean_pull) < 3.0 * std_pull / np.sqrt(len(y))
    assert std_pull < 1.0


@timer
def test_empirical_2pcf_eigenvalue_clipping():
    # The tabulated kernel is not guaranteed to be positive
    # semi-definite between arbitrary points. By default the negative
    # eigenvalues of the covariance matrix are clipped to zero
    # (equivalent to the singular value clipping of Gomes et al. 2025);
    # without the clipping, the Cholesky decomposition fails on this
    # data set.
    gp, X, y, y_err, noise = make_gp()
    K = gp.kernel(X)
    eigenvalues = np.linalg.eigvalsh(K)
    assert np.all(eigenvalues > -1e-10)

    gp.kernel.clip_eigenvalues = False
    K_raw = gp.kernel(X)
    assert np.min(np.linalg.eigvalsh(K_raw)) < 0.0
    gp._alpha = None
    np.testing.assert_raises(np.linalg.LinAlgError, gp.predict, X)


@timer
def test_empirical_2pcf_extrapolation():
    gp, X, y, y_err, noise = make_gp()

    # Far from the data (beyond max_sep), the kernel is zero, so the GP
    # returns the mean of the field with a variance equal to the zero
    # lag of the correlation function.
    np.random.seed(42)
    X_far = np.random.uniform(
        np.max(X) + 6.0 * gp._optimizer.max_sep,
        np.max(X) + 12.0 * gp._optimizer.max_sep,
        size=20,
    ).reshape((10, 2))
    y_far, y_cov_far = gp.predict(X_far, return_cov=True)
    np.testing.assert_allclose(y_far, np.mean(y), atol=1e-10)
    np.testing.assert_allclose(
        np.sqrt(np.diag(y_cov_far)), np.sqrt(gp.kernel.xi0), atol=1e-10
    )


@timer
def test_empirical_2pcf_introspection():
    gp, X, y, y_err, noise = make_gp()

    npix = gp._optimizer.npix
    xi, xi_clean, distance, pixel_size = gp.return_empirical_2pcf()
    assert xi.shape == (npix, npix)
    assert xi_clean.shape == (npix, npix)
    assert distance.shape == (npix * npix, 2)
    assert pixel_size == 0.5
    np.testing.assert_allclose(xi.flatten(), gp._optimizer._2pcf, atol=1e-10)
    np.testing.assert_allclose(xi_clean.flatten(), gp._optimizer._2pcf_fit, atol=1e-10)

    # plot_fitted_kernel uses _2pcf, _2pcf_fit and _2pcf_dist, so it
    # works for the empirical-2pcf optimizer.
    import matplotlib

    matplotlib.use("Agg")
    gp.plot_fitted_kernel()

    # return_2pcf is only meaningful for the two-pcf and anisotropic
    # optimizers.
    np.testing.assert_raises(NotImplementedError, gp.return_2pcf)

    # return_empirical_2pcf is only available for the empirical-2pcf
    # optimizer.
    gp_iso = treegp.GPInterpolation(optimizer="two-pcf")
    np.testing.assert_raises(NotImplementedError, gp_iso.return_empirical_2pcf)


@timer
def test_empirical_kernel_orientation():
    # Build an analytic anisotropic correlation function, with a
    # correlation length larger along x than along y, on a grid in
    # treecorr TwoD layout, i.e. indexed [iy, ix].
    npix = 20
    pixel_size = 1.0
    lag = (np.arange(npix) - npix // 2) * pixel_size
    dx, dy = np.meshgrid(lag, lag)
    xi_grid = np.exp(-0.5 * (dx**2 / 16.0 + dy**2 / 1.0))

    kernel = treegp.EmpiricalCorrelationKernel(lag, lag, xi_grid)

    # A separation along x should be evaluated with the long
    # correlation length, a separation along y with the short one.
    X = np.array([[3.0, 0.0], [0.0, 0.0], [0.0, 3.0]])
    K = kernel(X)
    np.testing.assert_allclose(K[0, 1], np.exp(-0.5 * 9.0 / 16.0), atol=1e-10)
    np.testing.assert_allclose(K[2, 1], np.exp(-0.5 * 9.0 / 1.0), atol=1e-10)
    np.testing.assert_allclose(K, K.T, atol=1e-10)
    np.testing.assert_allclose(np.diag(K), kernel.diag(X), atol=1e-10)
    np.testing.assert_allclose(kernel.xi0, 1.0, atol=1e-10)

    # Cross covariance between two sets of points.
    X2 = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [10.0, -3.0]])
    HT = kernel(X2, Y=X)
    assert HT.shape == (len(X2), len(X))
    np.testing.assert_allclose(HT[0, 1], np.exp(-0.5 * 1.0 / 16.0), atol=1e-10)
    np.testing.assert_allclose(HT[1, 1], np.exp(-0.5 * 1.0 / 1.0), atol=1e-10)

    # Beyond the grid, the kernel is zero.
    X_far = np.array([[100.0, 100.0]])
    np.testing.assert_allclose(kernel(X_far, Y=X), 0.0, atol=1e-10)

    # sklearn kernel API with an empty theta.
    assert len(kernel.theta) == 0
    kernel_clone = kernel.clone_with_theta(kernel.theta)
    np.testing.assert_allclose(kernel_clone(X), K, atol=1e-10)
    kernel_copy = copy.deepcopy(kernel)
    np.testing.assert_allclose(kernel_copy(X), K, atol=1e-10)
    assert kernel.is_stationary()

    # Only 2d coordinates are supported.
    np.testing.assert_raises(ValueError, kernel, np.array([[1.0], [2.0]]))


@timer
def test_empirical_2pcf_helpers():
    # _shift_and_bin conserves the sum (up to the 1/4 normalization)
    # and moves the zero lag from the intersection of the 4 central
    # pixels to the center of pixel N//2.
    np.random.seed(42)
    raw = np.random.uniform(size=(40, 40))
    binned = _shift_and_bin(raw)
    assert binned.shape == (20, 20)
    np.testing.assert_allclose(np.sum(binned), np.sum(raw) / 4.0, atol=1e-10)
    raw_peak = np.zeros((40, 40))
    raw_peak[19:21, 19:21] = 1.0
    binned_peak = _shift_and_bin(raw_peak)
    assert binned_peak[10, 10] == 1.0
    np.testing.assert_allclose(np.sum(binned_peak), 1.0, atol=1e-10)
    np.testing.assert_raises(ValueError, _shift_and_bin, np.zeros((39, 39)))
    np.testing.assert_raises(ValueError, _shift_and_bin, np.zeros((40, 20)))

    # _power2corr is the inverse of _corr2power for a symmetric map.
    npix = 20
    lag = np.arange(npix) - npix // 2
    dx, dy = np.meshgrid(lag, lag)
    xi = np.exp(-0.5 * (dx**2 + dy**2) / 9.0)
    np.testing.assert_allclose(_power2corr(_corr2power(xi)), xi, atol=1e-10)

    # _threshold zeroes the elements below the threshold and keeps
    # the ones above.
    p = np.ones((20, 20))
    p[10, 10] = 1e4
    p_thresh = _threshold(p, n_sigma=3.0)
    assert p_thresh[10, 10] == 1e4
    assert np.sum(p_thresh != 0.0) == 1
    # An all-zero power spectrum is returned unchanged.
    np.testing.assert_allclose(
        _threshold(np.zeros((20, 20))), np.zeros((20, 20)), atol=1e-10
    )

    # _apod is 1 at zero lag and goes to zero at the edge of the grid
    # (the 4-term Blackman-Harris window is 6e-5 at its edge).
    window = _apod(xi)
    np.testing.assert_allclose(window[npix // 2, npix // 2], 1.0, atol=1e-10)
    np.testing.assert_allclose(window[npix // 2, 0], 0.0, atol=1e-4)

    # The Hann window is exactly zero at its edge, 1/2 at half radius,
    # and gentler (larger) than Blackman-Harris in the interior. (Right
    # at the edge Blackman-Harris is larger: it floors at 6e-5 while
    # Hann goes to zero quadratically.)
    window_hann = _apod(xi, window="hann")
    np.testing.assert_allclose(window_hann[npix // 2, npix // 2], 1.0, atol=1e-10)
    np.testing.assert_allclose(window_hann[npix // 2, 0], 0.0, atol=1e-10)
    np.testing.assert_allclose(
        window_hann[npix // 2, npix // 2 + npix // 4], 0.5, atol=1e-10
    )
    interior = _apod(xi, r_max=0.9 * (npix // 2)) > 0.0
    assert np.all(window_hann[interior] >= window[interior])

    # An elliptical window with g1 > 0 (major axis along x) is wider
    # along x than along y, and reduces exactly to the isotropic
    # window at g1 = g2 = 0.
    window_ell = _apod(xi, window="hann", g1=0.4, g2=0.0)
    d = npix // 4
    assert window_ell[npix // 2, npix // 2 + d] > window_ell[npix // 2 + d, npix // 2]
    np.testing.assert_allclose(_apod(xi, g1=0.0, g2=0.0), _apod(xi), atol=1e-14)

    # A smaller apodization radius reaches zero earlier.
    window_small = _apod(xi, r_max=npix // 4)
    np.testing.assert_allclose(window_small[npix // 2, npix // 2], 1.0, atol=1e-10)
    np.testing.assert_allclose(
        window_small[npix // 2, npix // 2 + npix // 4 + 1 :], 0.0, atol=1e-10
    )
    # A larger radius leaves the window non-zero at the grid edge.
    window_large = _apod(xi, r_max=npix, window="hann")
    assert window_large[npix // 2, 0] > 0.1


@timer
def test_adaptive_moments():
    # Adaptive moments recover the shear of an analytic elliptical
    # gaussian in the get_correlation_length_matrix convention.
    for g1_true, g2_true in [(0.0, 0.0), (0.3, 0.0), (0.0, -0.2), (0.2, 0.2)]:
        f = make_elliptical_gaussian(64, 6.0, g1_true, g2_true)
        g1, g2 = _adaptive_moments(f)
        np.testing.assert_allclose([g1, g2], [g1_true, g2_true], atol=1e-2)

    # An all-zero (or all-negative, clipped to zero) map returns no
    # anisotropy.
    assert _adaptive_moments(np.zeros((32, 32))) == (0.0, 0.0)
    assert _adaptive_moments(-np.ones((32, 32))) == (0.0, 0.0)


@timer
def test_empirical_2pcf_anisotropic_apod():
    L = get_correlation_length_matrix(2.0, 0.2, 0.2)
    invLam = np.linalg.inv(L)
    kernel = 2.0**2 * treegp.AnisotropicRBF(invLam=invLam)
    X, y, y_err = make_2d_grf(kernel, noise=0.3, seed=42, npoints=2000)

    def run(**kwargs):
        gp = treegp.GPInterpolation(
            optimizer="empirical-2pcf",
            normalize=True,
            max_sep=6.0,
            pixel_size=0.5,
            **kwargs,
        )
        gp.initialize(X, y, y_err=y_err)
        gp.solve()
        return gp

    # Auto mode: the field was generated with positive (g1, g2), so the
    # measured anisotropy of its 2-pcf must have positive components,
    # and the applied shear is the measured one times apod_g_scale.
    gp = run(apod_anisotropy="auto", apod_g_scale=0.5)
    g1_m, g2_m = gp._optimizer._apod_g_measured
    assert g1_m > 0.0
    assert g2_m > 0.0
    np.testing.assert_allclose(
        gp._optimizer._apod_g_applied, [0.5 * g1_m, 0.5 * g2_m], atol=1e-12
    )
    assert gp._optimizer._xi_clean_pass1 is not None
    y_predict = gp.predict(X)
    assert np.var(y - y_predict) < 0.5 * np.var(y)

    # apod_g_scale = 0 in auto mode applies an isotropic window, so the
    # final map is identical to the first pass.
    gp = run(apod_anisotropy="auto", apod_g_scale=0.0)
    np.testing.assert_allclose(
        gp._optimizer._xi_clean, gp._optimizer._xi_clean_pass1, atol=1e-14
    )

    # Manual mode: the given shear is applied directly, nothing is
    # measured.
    gp = run(apod_anisotropy=(0.3, 0.1))
    assert gp._optimizer._apod_g_measured is None
    np.testing.assert_allclose(gp._optimizer._apod_g_applied, [0.3, 0.1], atol=1e-12)
    y_predict = gp.predict(X)
    assert np.var(y - y_predict) < 0.5 * np.var(y)


@timer
def test_empirical_2pcf_validation():
    # Only 2d fields are supported.
    X = np.random.uniform(-10, 10, 100).reshape((100, 1))
    y = np.random.normal(size=100)
    np.testing.assert_raises(ValueError, treegp.empirical_2pcf, X, y, np.zeros_like(y))

    # max_sep must span at least 2 pixels.
    X = np.random.uniform(-10, 10, 200).reshape((100, 2))
    np.testing.assert_raises(
        ValueError,
        treegp.empirical_2pcf,
        X,
        y,
        np.zeros_like(y),
        1.0,
        2.0,
    )

    # Unknown optimizer is rejected.
    np.testing.assert_raises(ValueError, treegp.GPInterpolation, optimizer="gomes25")

    # Unknown apodization window and non-positive apodization radius
    # are rejected.
    X2d = np.random.uniform(-10, 10, 200).reshape((100, 2))
    np.testing.assert_raises(
        ValueError,
        treegp.empirical_2pcf,
        X2d,
        y,
        np.zeros_like(y),
        apod_window="tukey",
    )
    np.testing.assert_raises(
        ValueError,
        treegp.empirical_2pcf,
        X2d,
        y,
        np.zeros_like(y),
        apod_radius=-1.0,
    )

    # Invalid apod_anisotropy and apod_g_scale values are rejected.
    for bad in ["hsm", (0.1, 0.2, 0.3), (0.8, 0.7)]:
        np.testing.assert_raises(
            ValueError,
            treegp.empirical_2pcf,
            X2d,
            y,
            np.zeros_like(y),
            apod_anisotropy=bad,
        )
    np.testing.assert_raises(
        ValueError,
        treegp.empirical_2pcf,
        X2d,
        y,
        np.zeros_like(y),
        apod_g_scale=-0.5,
    )

    # The empirical-2pcf optimizer builds its own kernel: passing one
    # is rejected.
    np.testing.assert_raises(
        ValueError,
        treegp.GPInterpolation,
        kernel="2.0**2 * AnisotropicVonKarman(scale_length=[1.0, 1.0])",
        optimizer="empirical-2pcf",
    )

    # An EmpiricalCorrelationKernel has no hyperparameters: it is
    # rejected by the fitting optimizers (also inside a composite
    # kernel), but allowed with optimizer="none".
    kernel_str = (
        "EmpiricalCorrelationKernel(array([-1., 0., 1.]), array([-1., 0., 1.]), "
        "array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]))"
    )
    for opt in ["two-pcf", "anisotropic", "log-likelihood"]:
        np.testing.assert_raises(
            ValueError, treegp.GPInterpolation, kernel=kernel_str, optimizer=opt
        )
        np.testing.assert_raises(
            ValueError,
            treegp.GPInterpolation,
            kernel="2.0**2 * " + kernel_str,
            optimizer=opt,
        )
    gp_none = treegp.GPInterpolation(kernel=kernel_str, optimizer="none")
    assert isinstance(gp_none.kernel_template, treegp.EmpiricalCorrelationKernel)

    # y_err = 0 runs (unweighted 2-point correlation function).
    L = get_correlation_length_matrix(2.0, 0.2, 0.2)
    invLam = np.linalg.inv(L)
    kernel = 2.0**2 * treegp.AnisotropicRBF(invLam=invLam)
    X, y, _ = make_2d_grf(kernel, noise=None, seed=42, npoints=1000)
    gp = treegp.GPInterpolation(
        optimizer="empirical-2pcf",
        normalize=True,
        white_noise=0.7,
        max_sep=6.0,
        pixel_size=0.5,
    )
    gp.initialize(X, y)
    gp.solve()
    y_predict = gp.predict(X)
    assert np.var(y - y_predict) < 0.5 * np.var(y)

    # The apodization window and radius are threaded through
    # GPInterpolation, and the interpolation still works with them.
    gp = treegp.GPInterpolation(
        optimizer="empirical-2pcf",
        normalize=True,
        white_noise=0.7,
        max_sep=6.0,
        pixel_size=0.5,
        apod_window="hann",
        apod_radius=4.0,
    )
    gp.initialize(X, y)
    gp.solve()
    assert gp._optimizer.apod_window == "hann"
    assert gp._optimizer.apod_radius == 4.0
    y_predict = gp.predict(X)
    assert np.var(y - y_predict) < 0.5 * np.var(y)


if __name__ == "__main__":
    test_empirical_2pcf_gp()
    test_empirical_2pcf_eigenvalue_clipping()
    test_empirical_2pcf_extrapolation()
    test_empirical_2pcf_introspection()
    test_empirical_kernel_orientation()
    test_empirical_2pcf_helpers()
    test_adaptive_moments()
    test_empirical_2pcf_anisotropic_apod()
    test_empirical_2pcf_validation()
