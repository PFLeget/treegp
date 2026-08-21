"""
.. module:: empirical_2pcf
"""

import copy
import warnings
import numpy as np
import treecorr

from scipy import fft

from .kernels import EmpiricalCorrelationKernel
from .two_pcf import get_correlation_length_matrix


def _blackman_harris(r, r_max=1.0):
    """Return Blackman-Harris function of variable r with
    peak at 1.0 and going to zero at r_max.

    :param r:     Radii where to evaluate the window. (ndarray)
    :param r_max: Radius where the window reaches zero. [default: 1.]
    """
    upi = np.pi * r / r_max
    out = (
        0.35875
        + 0.48829 * np.cos(upi)
        + 0.14128 * np.cos(2 * upi)
        + 0.01168 * np.cos(3 * upi)
    )
    return np.where(np.abs(upi) <= np.pi, out, 0.0)


def _hann(r, r_max=1.0):
    """Return Hann function of variable r with peak at 1.0 and going
    to zero at r_max. Gentler taper (less bias on the correlation
    function) than Blackman-Harris, at the price of more spectral
    leakage.

    :param r:     Radii where to evaluate the window. (ndarray)
    :param r_max: Radius where the window reaches zero. [default: 1.]
    """
    out = 0.5 * (1.0 + np.cos(np.pi * r / r_max))
    return np.where(np.abs(r) <= r_max, out, 0.0)


APOD_WINDOWS = {
    "blackman-harris": _blackman_harris,
    "hann": _hann,
}


def _apod(corr, r_max=None, window="blackman-harris", g1=0.0, g2=0.0):
    """Return an apodization window with the same shape as the given
    2d correlation function, equal to 1 at zero lag (pixel N//2) and
    going to zero at r_max pixels from it.

    The window can be made elliptical using the (g1, g2) shear
    parametrization of Leget et al. (2021) (same convention as
    get_correlation_length_matrix): the window reaches zero at r_max
    along the major axis, whose direction is phi = 0.5 arctan2(g2, g1)
    from the x (column) axis, and at r_max * q along the minor axis,
    with q = (1 - g) / (1 + g). At g1 = g2 = 0 the window is isotropic.

    :param corr:   2d correlation function, zero lag at pixel N//2.
                   (N, N) ndarray
    :param r_max:  Radius (in pixels) where the window reaches zero.
                   N//2 (the grid edge) if not given. A radius beyond
                   the grid edge gives a gentler taper but leaves the
                   window non-zero at the edge, reintroducing some
                   spectral leakage. [default: None]
    :param window: Name of the window function, one of "blackman-harris"
                   or "hann". [default: "blackman-harris"]
    :param g1, g2: Shear applied to the isotropic window.
                   [default: 0., 0.]
    """
    if r_max is None:
        r_max = corr.shape[0] // 2
    yx = np.indices(corr.shape)
    ctr = np.array(corr.shape) // 2
    yx -= ctr[:, np.newaxis, np.newaxis]
    dy = yx[0]
    dx = yx[1]
    # Dimensionless elliptical radius, equal to 1 on the ellipse with
    # semi-major axis r_max. Reduces to rad / r_max at g1 = g2 = 0.
    L = get_correlation_length_matrix(r_max, g1, g2)
    invL = np.linalg.inv(L)
    r_ell = np.sqrt(
        invL[0, 0] * dx**2 + 2.0 * invL[0, 1] * dx * dy + invL[1, 1] * dy**2
    )
    return APOD_WINDOWS[window](r_ell, 1.0)


def _adaptive_moments(xi_map, n_iter=30, tol=1e-6):
    """Measure the anisotropy of a 2d correlation function map using
    HSM-like adaptive weighted second moments: the weight function is an
    elliptical gaussian that is iterated until it matches the measured
    moments (Hirata & Seljak 2003). Negative values of the map are
    clipped to zero, and the map is assumed to be centered on pixel
    N//2 (no centroid iteration).

    Returns the measured anisotropy as a (g1, g2) shear, in the same
    convention as get_correlation_length_matrix.

    :param xi_map: 2d correlation function, zero lag at pixel N//2.
                   (N, N) ndarray
    :param n_iter: Maximum number of iterations. [default: 30]
    :param tol:    Relative tolerance on the moment matrix for
                   convergence. [default: 1e-6]
    """
    f = np.clip(xi_map, 0.0, None)
    if np.all(f == 0.0):
        return 0.0, 0.0

    yx = np.indices(f.shape)
    ctr = np.array(f.shape) // 2
    yx = yx - ctr[:, np.newaxis, np.newaxis]
    dy = yx[0].astype(float)
    dx = yx[1].astype(float)

    # Start from an isotropic gaussian weight.
    sigma = f.shape[0] / 8.0
    M = np.array([[sigma**2, 0.0], [0.0, sigma**2]])
    converged = False
    for _ in range(n_iter):
        invM = np.linalg.inv(M)
        arg = invM[0, 0] * dx**2 + 2.0 * invM[0, 1] * dx * dy + invM[1, 1] * dy**2
        w = np.exp(-0.5 * arg)
        norm = np.sum(w * f)
        if norm <= 0.0:
            return 0.0, 0.0
        mxx = np.sum(w * f * dx * dx) / norm
        myy = np.sum(w * f * dy * dy) / norm
        mxy = np.sum(w * f * dx * dy) / norm
        # The factor 2 makes the iteration converge to the true
        # covariance for a gaussian map: the weighted moments measure
        # (C^-1 + M^-1)^-1, whose fixed point after doubling is M = C.
        M_new = 2.0 * np.array([[mxx, mxy], [mxy, myy]])
        if np.max(np.abs(M_new - M)) < tol * np.trace(M_new):
            M = M_new
            converged = True
            break
        M = M_new
    if not converged:
        warnings.warn(
            "Adaptive moments did not converge after %i iterations; "
            "using the last iterate." % (n_iter)
        )

    # Moments give the distortion chi; convert it to the shear g used
    # by get_correlation_length_matrix.
    trace = M[0, 0] + M[1, 1]
    chi1 = (M[0, 0] - M[1, 1]) / trace
    chi2 = 2.0 * M[0, 1] / trace
    chi = np.hypot(chi1, chi2)
    if chi == 0.0:
        return 0.0, 0.0
    if chi >= 1.0:
        # Degenerate (essentially 1d) map; cap just below 1.
        chi = 1.0 - 1e-12
    q = np.sqrt((1.0 - chi) / (1.0 + chi))
    g = (1.0 - q) / (1.0 + q)
    return g * chi1 / chi, g * chi2 / chi


def _shift_and_bin(corr_func):
    """Bin a 2d correlation function 2x2 in such a way that the zero
    lag, located at the intersection of the 4 central pixels of the
    input (treecorr TwoD convention), ends up at the center of pixel
    N//2 of the output, where N is the output side length.

    :param corr_func: 2d correlation function from treecorr TwoD binning,
                      with even side length. (2N, 2N) ndarray
    """
    if corr_func.shape[0] != corr_func.shape[1]:
        raise ValueError(
            "corr_func must be square. Current shape: %s" % (str(corr_func.shape))
        )
    if corr_func.shape[0] % 2 != 0:
        raise ValueError(
            "corr_func side length must be even. Current shape: %s"
            % (str(corr_func.shape))
        )
    # Roll points astride zero lag into origin corner
    s = corr_func.shape[0] // 2
    corr = np.roll(
        corr_func, -(s - 1), axis=0
    )  # This places the 2 mirror-image low-f pixels at 0,1
    corr = np.sum(corr.reshape(s, 2, -1), axis=1)  # Bin by 2 in y
    corr = np.roll(corr, -(s - 1), axis=1)
    corr = np.sum(corr.reshape(-1, s, 2), axis=2)  # Bin by 2 in x
    # Roll DC back to center from its current position at (0,0)
    s = corr.shape[0] // 2
    corr = np.roll(corr, s, axis=0)
    corr = np.roll(corr, s, axis=1)
    return corr / 4.0


def _threshold(p, n_sigma=3.0):
    """Keep only the elements of the power spectrum p that are above
    n_sigma times the noise, where the noise is estimated from the
    16-50-84 percentiles of the non-zero elements of p. Elements below
    the threshold are set to zero.

    :param p:       2d power spectrum. (ndarray)
    :param n_sigma: Multiple of the noise below which elements
                    are zeroed out. [default: 3.]
    """
    nonzero = p[p != 0.0]
    if len(nonzero) == 0:
        return p
    hml = np.percentile(nonzero, (16.0, 50.0, 84.0))
    thresh = hml[1] + n_sigma * 0.5 * (hml[2] - hml[0])
    return np.where(p > thresh, p, 0.0)


def _corr2power(corr):
    """Get power spectrum from a (square, even side length) 2d correlation
    function. Both have their origin at pixel N//2 along the first axis.

    :param corr: 2d correlation function, zero lag at pixel N//2. (N, N) ndarray
    """
    s = corr.shape[0] // 2
    tmp = np.roll(corr, -s, axis=0)
    tmp = np.roll(tmp, -s, axis=1)
    pk = fft.rfft2(tmp)
    return np.roll(pk.real, s, axis=0)


def _power2corr(pk):
    """Get 2d correlation function from a power spectrum, origin at
    pixel N//2 in the power spectrum first axis. Inverse of _corr2power.

    :param pk: 2d power spectrum from _corr2power. (N, N//2+1) ndarray
    """
    s = pk.shape[0] // 2
    corr = fft.irfft2(np.roll(pk, -s, axis=0))
    corr = np.roll(corr, s, axis=0)
    corr = np.roll(corr, s, axis=1)
    return corr


class empirical_2pcf(object):
    """
    Build a gaussian process kernel directly from the measured anisotropic
    2d 2-point correlation function, following Gomes et al. (2025)
    (AJ 170:361, doi:10.3847/1538-3881/ae1a7b). No hyperparameters are
    fitted: the measured correlation function, cleaned by apodization and
    thresholding of its Fourier power spectrum, is tabulated and used
    as the kernel.

    :param X:               Coordinates of the field. (n_samples, 2)
    :param y:               Values of the field. (n_samples)
    :param y_err:           Error of y. (n_samples)
    :param max_sep:         Maximum separation in each coordinate of the
                            2d correlation function grid (half width of the
                            grid), in the same units as X. Rounded up to an
                            integer number of pixels. Computed automatically
                            (half of the field diagonal) if not given.
                            [default: None]
    :param pixel_size:      Pixel size of the 2d correlation function grid,
                            in the same units as X. Computed automatically
                            (twice the mean separation between points) if
                            not given. [default: None]
    :param power_threshold: Signal-to-noise threshold below which Fourier
                            modes of the measured correlation function are
                            set to zero. [default: 2.5]
    :param apodize:         Whether to apodize the measured correlation
                            function before taking its Fourier transform.
                            [default: True]
    :param apod_window:     Name of the apodization window, one of
                            "blackman-harris" or "hann". Hann is a gentler
                            taper (less bias on the correlation function)
                            at the price of more spectral leakage.
                            [default: "blackman-harris"]
    :param apod_radius:     Radius where the apodization window reaches
                            zero, in the same units as X. max_sep (the
                            grid edge) if not given. A radius beyond
                            max_sep gives a gentler taper but leaves the
                            window non-zero at the grid edge,
                            reintroducing some spectral leakage.
                            [default: None]
    :param apod_anisotropy: Anisotropy of the apodization window, using
                            the (g1, g2) shear parametrization of Leget
                            et al. (2021). None gives an isotropic
                            window. A (g1, g2) tuple applies the given
                            shear: the window reaches zero at
                            apod_radius along the major axis (direction
                            0.5 arctan2(g2, g1) from the x axis) and at
                            apod_radius * q along the minor axis, with
                            q = (1 - g) / (1 + g). "auto" measures
                            (g1, g2) on the correlation function itself:
                            a first cleaning pass is done with the
                            isotropic window, the anisotropy of its
                            output is measured with adaptive weighted
                            second moments, and the raw correlation
                            function is re-cleaned with the matched
                            elliptical window. Ignored if apodize is
                            False. [default: None]
    :param apod_g_scale:    Factor multiplying the measured (g1, g2)
                            before building the elliptical window in
                            "auto" mode, to soften (< 1) or exaggerate
                            (> 1) the anisotropy of the taper.
                            [default: 1.]
    """

    def __init__(
        self,
        X,
        y,
        y_err,
        max_sep=None,
        pixel_size=None,
        power_threshold=2.5,
        apodize=True,
        apod_window="blackman-harris",
        apod_radius=None,
        apod_anisotropy=None,
        apod_g_scale=1.0,
    ):
        self.ndim = np.shape(X)[1]
        if self.ndim != 2:
            raise ValueError(
                "empirical-2pcf supports only 2d modeling. Current ndim: %i"
                % (self.ndim)
            )
        if apod_window not in APOD_WINDOWS:
            raise ValueError(
                "Only %s are supported for apod_window. Current value: %s"
                % (sorted(APOD_WINDOWS), apod_window)
            )
        if apod_radius is not None and apod_radius <= 0:
            raise ValueError(
                "apod_radius must be positive. Current value: %s" % (apod_radius)
            )
        if apod_anisotropy is not None and not (
            isinstance(apod_anisotropy, str) and apod_anisotropy == "auto"
        ):
            apod_anisotropy = np.asarray(apod_anisotropy, dtype=float)
            if apod_anisotropy.shape != (2,):
                raise ValueError(
                    "apod_anisotropy must be None, 'auto', or a (g1, g2) pair. "
                    "Current value: %s" % (apod_anisotropy)
                )
            if np.hypot(apod_anisotropy[0], apod_anisotropy[1]) >= 1.0:
                raise ValueError(
                    "The norm of the apod_anisotropy (g1, g2) shear must be "
                    "lower than one. Current value: %s" % (apod_anisotropy)
                )
        if not np.isfinite(apod_g_scale) or apod_g_scale < 0:
            raise ValueError(
                "apod_g_scale must be finite and non-negative. "
                "Current value: %s" % (apod_g_scale)
            )
        self.X = X
        self.y = y
        self.y_err = y_err
        self.power_threshold = power_threshold
        self.apodize = apodize
        self.apod_window = apod_window
        self.apod_radius = apod_radius
        self.apod_anisotropy = apod_anisotropy
        self.apod_g_scale = apod_g_scale

        size_x = np.max(X[:, 0]) - np.min(X[:, 0])
        size_y = np.max(X[:, 1]) - np.min(X[:, 1])
        rho = float(len(X[:, 0])) / (size_x * size_y)
        # if max_sep is None, set max_sep to half of the size of the
        # given field.
        if max_sep is None:
            max_sep = np.sqrt(size_x**2 + size_y**2) / 2.0
        # if pixel_size is None, set it to twice the mean separation
        # between points, so treecorr bins (half a pixel) match the
        # mean point separation.
        if pixel_size is None:
            pixel_size = 2.0 * np.sqrt(1.0 / rho)

        half_npix = int(np.ceil(max_sep / pixel_size))
        if half_npix < 2:
            raise ValueError(
                "max_sep must span at least 2 pixels. "
                "Current max_sep: %f, pixel_size: %f" % (max_sep, pixel_size)
            )
        # Final grid is (npix, npix) with zero lag at the center of
        # pixel npix//2 in each axis, covering [-max_sep, max_sep].
        self.npix = 2 * half_npix
        self.pixel_size = pixel_size
        self.max_sep = half_npix * pixel_size

    def comp_2pcf(self, X, y, y_err):
        """
        Estimate the anisotropic 2d 2-point correlation function
        using TreeCorr.

        TreeCorr TwoD binning puts zero lag at the intersection of the 4
        central pixels, so the correlation function is measured at half
        the requested pixel size and then binned 2x2 so that zero lag
        ends up at the center of pixel npix//2.

        :param X:  Coordinates of the field. (n_samples, 2)
        :param y:  Values of the field. (n_samples)
        :param y_err: Error of y. (n_samples)
        """
        if np.sum(y_err) == 0:
            w = None
        else:
            w = 1.0 / y_err**2

        cat = treecorr.Catalog(x=X[:, 0], y=X[:, 1], k=(y - np.mean(y)), w=w)
        kk = treecorr.KKCorrelation(
            max_sep=self.max_sep,
            nbins=2 * self.npix,
            bin_type="TwoD",
            bin_slop=0,
        )
        kk.process(cat)
        return _shift_and_bin(kk.xi)

    def _clean_pass(self, xi, g1=0.0, g2=0.0):
        """Single cleaning pass: apodize (if requested) with the given
        window shear, threshold the Fourier power spectrum, and
        transform back.

        :param xi:     Measured 2d correlation function, zero lag at
                       pixel npix//2. (npix, npix) ndarray
        :param g1, g2: Shear applied to the apodization window.
                       [default: 0., 0.]
        """
        if self.apodize:
            if self.apod_radius is None:
                r_max = None
            else:
                r_max = self.apod_radius / self.pixel_size
            window = _apod(xi, r_max=r_max, window=self.apod_window, g1=g1, g2=g2)
            pk = _corr2power(xi * window)
        else:
            pk = _corr2power(xi)
        pk = _threshold(pk, n_sigma=self.power_threshold)
        if np.all(pk == 0.0):
            raise RuntimeError(
                "All Fourier modes of the measured 2-point correlation "
                "function are below the power threshold. The field might "
                "not have significant correlations; try lowering "
                "power_threshold (current value: %f)." % (self.power_threshold)
            )
        return _power2corr(pk)

    def clean(self, xi):
        """
        Clean the measured 2d 2-point correlation function by apodizing
        it (if requested) and keeping only the Fourier modes above
        power_threshold times the noise. The surviving power is
        positive, so the cleaned correlation function is positive
        semi-definite on its grid.

        With apod_anisotropy="auto", a first pass is cleaned with the
        isotropic window, the anisotropy of its output is measured with
        adaptive weighted second moments, and the raw correlation
        function is re-cleaned with the matched elliptical window
        (scaled by apod_g_scale). The measured and applied shears are
        stored in _apod_g_measured and _apod_g_applied, and the first
        pass in _xi_clean_pass1.

        :param xi: Measured 2d correlation function, zero lag at
                   pixel npix//2. (npix, npix) ndarray
        """
        self._apod_g_measured = None
        self._xi_clean_pass1 = None
        g1, g2 = 0.0, 0.0
        if self.apodize and self.apod_anisotropy is not None:
            if isinstance(self.apod_anisotropy, str):
                # "auto": isotropic pass, measure, elliptical re-clean.
                xi_pass1 = self._clean_pass(xi)
                g1_m, g2_m = _adaptive_moments(xi_pass1)
                self._xi_clean_pass1 = xi_pass1
                self._apod_g_measured = (g1_m, g2_m)
                g1 = self.apod_g_scale * g1_m
                g2 = self.apod_g_scale * g2_m
                g_norm = np.hypot(g1, g2)
                if g_norm >= 0.9:
                    warnings.warn(
                        "The scaled apodization shear norm (%f) was capped "
                        "at 0.9." % (g_norm)
                    )
                    g1 *= 0.9 / g_norm
                    g2 *= 0.9 / g_norm
            else:
                g1, g2 = self.apod_anisotropy
        self._apod_g_applied = (g1, g2)
        return self._clean_pass(xi, g1=g1, g2=g2)

    def optimizer(self, kernel):
        """
        Build the gaussian process kernel from the measured 2d 2-point
        correlation function. Contrary to the other optimizers, no
        hyperparameters are fitted and the given kernel is ignored: the
        cleaned measured correlation function is returned as a tabulated
        EmpiricalCorrelationKernel.

        :param kernel: sklearn.gaussian_process kernel. (ignored)
        """
        xi = self.comp_2pcf(self.X, self.y, self.y_err)
        xi_clean = self.clean(xi)

        lag = (np.arange(self.npix) - self.npix // 2) * self.pixel_size
        kernel_out = EmpiricalCorrelationKernel(lag, lag, xi_clean)

        # Keep diagnostics around. dx varies along the columns of the
        # treecorr TwoD maps, so the flattened distances match the
        # flattened correlation functions.
        dx, dy = np.meshgrid(lag, lag)
        self._xi = xi
        self._xi_clean = xi_clean
        self._2pcf = xi.flatten()
        self._2pcf_fit = xi_clean.flatten()
        self._2pcf_dist = np.array([dx.flatten(), dy.flatten()]).T
        self._2pcf_mask = np.ones(self.npix**2, dtype=bool)
        self._kernel = copy.deepcopy(kernel_out)
        return kernel_out
