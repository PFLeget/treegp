"""
.. module:: gp_interp
"""

import treegp
import numpy as np
import copy
import warnings

from .kernels import eval_kernel
from .kernels import EmpiricalCorrelationKernel
from .grid_gp import GridConvolutionGP

from sklearn.neighbors import KNeighborsRegressor
from scipy.linalg import cholesky, cho_solve


def _kernel_contains(kernel, kernel_class):
    """Return whether the given (possibly composite) sklearn kernel is,
    or contains, an instance of kernel_class."""
    if isinstance(kernel, kernel_class):
        return True
    return any(
        isinstance(param, kernel_class)
        for param in kernel.get_params(deep=True).values()
    )


class GPInterpolation(object):
    """
    An interpolator that uses 2-point correlation function informations
    or Maximum Likelihood informations to do a gaussian process to interpolate
    a single surface.

    :param kernel:       A string that can be `eval`ed to make a
                         sklearn.gaussian_process.kernels.Kernel object.  The reprs of
                         sklearn.gaussian_process.kernels will work, as well as the repr of a
                         custom treegp VonKarman object.  [default: 'RBF(1)']
    :param optimizer:    Indicates which techniques to use for optimizing the kernel. Five options
                         are available. "none" does not optimize hyperparameters and used the one
                         given in the kernel. "two-pcf" optimize the kernel on the 1d 2-point
                         correlation function estimate by treecorr. "anisotropic" optimize the kernel
                         on the 2d 2-point correlation function estimate by treecorr.
                         "two-pcf" and "anisotropic" implement Leget et al. 2021
                         (A&A 650, A81, arXiv:2103.09881).
                         "log-likelihood" used the classical maximum likelihood method.
                         "empirical-2pcf" implements Gomes et al. 2025 (AJ 170:361,
                         doi:10.3847/1538-3881/ae1a7b): the measured 2d 2-point correlation
                         function, cleaned by apodization and thresholding of its Fourier
                         power spectrum, is used directly as a tabulated kernel; no
                         hyperparameters are fitted: the kernel argument must be left to
                         its default (an error is raised otherwise), and min_sep and
                         nbins are ignored (the grid is controlled by max_sep and
                         pixel_size). Conversely, an EmpiricalCorrelationKernel has no
                         hyperparameters, so it is rejected by the fitting optimizers
                         and can only be used with "empirical-2pcf" or "none".
                         As the tabulated kernel is not guaranteed to be
                         positive semi-definite between arbitrary points, the
                         covariance is repaired: the spectral solve method is
                         positive semi-definite by construction (the negative
                         Fourier modes of the padded kernel image are clipped
                         to zero), and the direct solve method adds an
                         escalating diagonal jitter until the Cholesky
                         factorization succeeds (both equivalent in spirit to
                         the singular value clipping of Gomes et al. 2025).
                         If the factorization still fails with a LinAlgError,
                         increase white_noise.
    :param normalize:    Whether to normalize the interpolation parameters to have a mean of 0.
                         Normally, the parameters being interpolated are not mean 0, so you would
                         want this to be True, but if your parameters have an a priori mean of 0,
                         then subtracting off the realized mean would be invalid.  [default: True]
    :param white_noise:  A float value that indicate the ammount of white noise that you want to
                         use during the gp interpolation. This is an additional uncorrelated noise
                         added to the error of the interpolated parameters. [default: 0.]
    :param n_neighbors:  Number of neighbors to used for interpolating the spatial average using
                         a KNeighbors interpolation. Used only if average_fits is not None. [defaulf: 4]
    :param nbins:        Number of bins (if 1D correlation function) of the square root of the number
                         of bins (if 2D correlation function) used in TreeCorr to compute the
                         2-point correlation function. [default: 20]
    :param min_sep:      Minimum separation between pairs when computing 2-point correlation
                         function. In the same units as the coordinates of the field.
                         Compute automaticaly if it is not given. [default: None]
    :param max_sep:      Maximum separation between pairs when computing 2-point correlation
                         function. In the same units as the coordinates of the field.
                         Compute automaticaly if it is not given. [default: None]
    :param average_fits: A fits file that have the spatial average functions of the interpolated parameter
                         build in it. Build using meanify output across different
                         exposures. See meanify documentation. [default: None]
    :param pixel_size:   Pixel size of the 2d correlation function grid used by the
                         "empirical-2pcf" optimizer, in the same units as the coordinates
                         of the field. Computed automatically (twice the mean separation
                         between points) if it is not given. [default: None]
    :param power_threshold: Signal-to-noise threshold below which Fourier modes of the
                         measured 2-point correlation function are set to zero. Used only
                         by the "empirical-2pcf" optimizer. [default: 2.5]
    :param apodize:      Whether to apodize the measured 2-point correlation function
                         before taking its Fourier transform. Used only by the
                         "empirical-2pcf" optimizer. [default: True]
    :param apod_window:  Name of the apodization window, one of "blackman-harris" or
                         "hann". Hann is a gentler taper (less bias on the correlation
                         function) at the price of more spectral leakage. Used only by
                         the "empirical-2pcf" optimizer. [default: "blackman-harris"]
    :param apod_radius:  Radius where the apodization window reaches zero, in the same
                         units as the coordinates of the field. max_sep (the grid edge)
                         if it is not given. A radius beyond max_sep gives a gentler
                         taper but leaves the window non-zero at the grid edge,
                         reintroducing some spectral leakage. Used only by the
                         "empirical-2pcf" optimizer. [default: None]
    :param apod_anisotropy: Anisotropy of the apodization window, using the (g1, g2)
                         shear parametrization of Leget et al. 2021 (same convention as
                         get_correlation_length_matrix). None gives an isotropic window.
                         A (g1, g2) tuple applies the given shear: the window reaches
                         zero at apod_radius along the major axis (direction
                         0.5 arctan2(g2, g1) from the x axis) and at apod_radius * q
                         along the minor axis, with q = (1 - g) / (1 + g). "auto"
                         measures (g1, g2) on the correlation function itself: a first
                         cleaning pass is done with the isotropic window, the anisotropy
                         of its output is measured with adaptive weighted second
                         moments, and the raw correlation function is re-cleaned with
                         the matched elliptical window. Ignored if apodize is False.
                         Used only by the "empirical-2pcf" optimizer. [default: None]
    :param apod_g_scale: Factor multiplying the measured (g1, g2) before building the
                         elliptical window when apod_anisotropy="auto", to soften (< 1)
                         or exaggerate (> 1) the anisotropy of the taper. Used only by
                         the "empirical-2pcf" optimizer. [default: 1.]
    :param solve_method: How to solve for the gaussian process weights and evaluate
                         predictions with the "empirical-2pcf" optimizer (other
                         optimizers only accept "direct"). "spectral" exploits the grid
                         structure of the empirical kernel (see GridConvolutionGP): the
                         training covariance is never built, the solve is a
                         preconditioned conjugate gradient where each product costs one
                         FFT, and predictions sample a precomputed mean field, so time
                         and memory stay far below the O(N^3) / O(N^2) of the dense
                         path (minutes and several GB at N ~ 10^4). "direct" is the
                         dense path: covariance matrix plus Cholesky factorization.
                         Predictions with return_cov=True always use the dense path.
                         If None, defaults to "spectral" for the "empirical-2pcf"
                         optimizer and "direct" otherwise. [default: None]
    :param spread_upsample: Integer upsampling factor of the spreading grid used by
                         solve_method="spectral", with respect to the correlation
                         function grid pixels. Controls the (second order) smoothing
                         of the kernel by the bilinear spreading. [default: 2]
    :param cg_rtol:      Relative tolerance of the conjugate gradient solve of
                         solve_method="spectral". [default: 1e-7]
    :param cg_maxiter:   Maximum number of conjugate gradient iterations of
                         solve_method="spectral". [default: 500]
    """

    def __init__(
        self,
        kernel="RBF(1)",
        optimizer="two-pcf",
        normalize=True,
        p0=[3000.0, 0.0, 0.0],
        white_noise=0.0,
        n_neighbors=4,
        average_fits=None,
        indice_meanify=None,
        nbins=20,
        min_sep=None,
        max_sep=None,
        pixel_size=None,
        power_threshold=2.5,
        apodize=True,
        apod_window="blackman-harris",
        apod_radius=None,
        apod_anisotropy=None,
        apod_g_scale=1.0,
        solve_method=None,
        spread_upsample=2,
        cg_rtol=1e-7,
        cg_maxiter=500,
    ):
        self.normalize = normalize
        self.optimizer = optimizer
        self.white_noise = white_noise
        self.n_neighbors = n_neighbors
        self.nbins = nbins
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.pixel_size = pixel_size
        self.power_threshold = power_threshold
        self.apodize = apodize
        self.apod_window = apod_window
        self.apod_radius = apod_radius
        self.apod_anisotropy = apod_anisotropy
        self.apod_g_scale = apod_g_scale
        if solve_method is None:
            solve_method = "spectral" if optimizer == "empirical-2pcf" else "direct"
        if solve_method not in ["spectral", "direct"]:
            raise ValueError(
                "Only spectral and direct are supported for solve_method. "
                "Current value: %s" % (solve_method)
            )
        if solve_method == "spectral" and optimizer != "empirical-2pcf":
            raise ValueError(
                "solve_method='spectral' is only available for the "
                "empirical-2pcf optimizer. Current optimizer: %s" % (optimizer)
            )
        self.solve_method = solve_method
        self.spread_upsample = spread_upsample
        self.cg_rtol = cg_rtol
        self.cg_maxiter = cg_maxiter
        self._engine = None

        if self.optimizer == "anisotropic":
            self.robust_fit = True
        else:
            self.robust_fit = False

        self.p0_robust_fit = p0
        self.indice_meanify = indice_meanify

        if isinstance(kernel, str):
            self.kernel_template = eval_kernel(kernel)
        else:
            raise TypeError(
                "kernel should be a string a list or a numpy.ndarray of string"
            )

        if self.optimizer not in [
            "anisotropic",
            "two-pcf",
            "empirical-2pcf",
            "log-likelihood",
            "none",
        ]:
            raise ValueError(
                "Only anisotropic, two-pcf, empirical-2pcf, log-likelihood and none "
                "are supported for optimizer. Current value: %s" % (self.optimizer)
            )

        if self.optimizer == "empirical-2pcf" and kernel != "RBF(1)":
            raise ValueError(
                "The empirical-2pcf optimizer builds its own "
                "EmpiricalCorrelationKernel from the measured 2-point "
                "correlation function, so the kernel argument is ignored and "
                "should be left to its default value. Current value: %s" % (kernel)
            )
        if self.optimizer != "none" and _kernel_contains(
            self.kernel_template, EmpiricalCorrelationKernel
        ):
            raise ValueError(
                "EmpiricalCorrelationKernel has no hyperparameters to fit, so "
                "it can only be used with optimizer='empirical-2pcf' (where it "
                "is built automatically) or optimizer='none'. "
                "Current optimizer: %s" % (self.optimizer)
            )

        if average_fits is not None:
            import fitsio

            average = fitsio.read(average_fits)
            X0 = average["COORDS0"][0]
            y0 = average["PARAMS0"][0]
        else:
            X0 = None
            y0 = None
        self._X0 = X0
        self._y0 = y0

        self._alpha = None
        self._fact = None

    def _fit(self, kernel, X, y, y_err):
        """Update the Kernel with data.

        :param kernel: sklearn.gaussian_process kernel.
        :param X:  Coordinates of the field.  (n_samples, 1 or 2)
        :param y:  Values of the field.  (n_samples)
        :param y_err: Error of y. (n_samples)
        """
        self._alpha = None
        self._fact = None
        self._engine = None
        if self.optimizer != "none":
            # Hyperparameters estimation using 2-point correlation
            # function information.
            if self.optimizer in ["two-pcf", "anisotropic"]:
                anisotropic = self.optimizer == "anisotropic"
                self._optimizer = treegp.two_pcf(
                    X,
                    y,
                    y_err,
                    self.min_sep,
                    self.max_sep,
                    nbins=self.nbins,
                    anisotropic=anisotropic,
                    robust_fit=self.robust_fit,
                    p0=self.p0_robust_fit,
                )
                kernel = self._optimizer.optimizer(kernel)
            # Kernel built directly from the measured 2d 2-point correlation
            # function (Gomes et al. 2025). No hyperparameters are fitted and
            # the given kernel is ignored.
            if self.optimizer == "empirical-2pcf":
                self._optimizer = treegp.empirical_2pcf(
                    X,
                    y,
                    y_err,
                    max_sep=self.max_sep,
                    pixel_size=self.pixel_size,
                    power_threshold=self.power_threshold,
                    apodize=self.apodize,
                    apod_window=self.apod_window,
                    apod_radius=self.apod_radius,
                    apod_anisotropy=self.apod_anisotropy,
                    apod_g_scale=self.apod_g_scale,
                )
                kernel = self._optimizer.optimizer(kernel)
                if self.solve_method == "spectral":
                    self._engine = GridConvolutionGP(
                        self._optimizer._xi_clean,
                        self._optimizer.pixel_size,
                        upsample=self.spread_upsample,
                        cg_rtol=self.cg_rtol,
                        cg_maxiter=self.cg_maxiter,
                    )
            # Hyperparameters estimation using maximum likelihood fit.
            if self.optimizer == "log-likelihood":
                self._optimizer = treegp.log_likelihood(X, y, y_err)
                kernel = self._optimizer.optimizer(kernel)
        return kernel

    def predict(self, X, return_cov=False):
        """Predict responses to given coordinates.

        :param X:  The coordinates at which to interpolate.  (n_samples, 1 or 2).
        :returns:  Regressed parameters  (n_samples)
        """
        y_init = copy.deepcopy(self._y)
        y_err = copy.deepcopy(self._y_err)

        if self._engine is not None and not return_cov:
            # Spectral solve and predict (see GridConvolutionGP): the
            # weights are solved once by conjugate gradient, the mean
            # field is computed once as a grid convolution, and each
            # predict call only samples it, so its cost is independent
            # of the number of training points. return_cov=True falls
            # through to the dense path below.
            if self._engine._alpha is None:
                self._engine.solve(
                    self._X,
                    y_init - self._mean - self._spatial_average,
                    y_err,
                )
            y_interp = self._engine.predict(X)
            y_interp += self._mean + self._build_average_meanify(X)
            return y_interp

        y_interp, y_cov = self.return_gp_predict(
            y_init - self._mean - self._spatial_average,
            self._X,
            X,
            self.kernel,
            y_err=y_err,
            return_cov=return_cov,
        )
        y_interp = y_interp.T
        spatial_average = self._build_average_meanify(X)
        y_interp += self._mean + spatial_average
        if return_cov:
            return y_interp, y_cov
        else:
            return y_interp

    def return_gp_predict(self, y, X1, X2, kernel, y_err, return_cov=False):
        """Compute interpolation with gaussian process for a given kernel.

        :param y:      Values of the field.  (n_samples)
        :param X1:     The coodinates of the field.  (n_samples, 1 or 2)
        :param X2:     The coordinates at which to interpolate.  (n_samples, 1 or 2)
        :param kernel: sklearn.gaussian_process kernel.
        :param y_err:  Error of y. (n_samples)
        """
        HT = kernel.__call__(X2, Y=X1)
        if self._alpha is None:
            if isinstance(kernel, EmpiricalCorrelationKernel):
                # The PSD repair of the tabulated kernel is done at the
                # factorization level below (escalating diagonal
                # jitter), which is much cheaper than clipping the
                # eigenvalues of the covariance matrix, so bypass the
                # clipping when building the training covariance.
                clip = kernel.clip_eigenvalues
                kernel.clip_eigenvalues = False
                try:
                    K = kernel.__call__(X1)
                finally:
                    kernel.clip_eigenvalues = clip
            else:
                K = kernel.__call__(X1)
            K[np.diag_indices_from(K)] += y_err**2
            try:
                self._fact = cholesky(K, lower=True)
            except np.linalg.LinAlgError as e:
                if not _kernel_contains(kernel, EmpiricalCorrelationKernel):
                    raise np.linalg.LinAlgError(
                        "Cholesky decomposition of the covariance matrix failed "
                        "(%s). The kernel might not be positive definite; "
                        "increasing white_noise can help." % (str(e))
                    )
                # A tabulated correlation function is not guaranteed to
                # be positive semi-definite between arbitrary points:
                # add an escalating diagonal jitter until the
                # factorization succeeds.
                scale = np.mean(np.diagonal(K))
                jitter = 1e-6 * scale
                added = 0.0
                fact = None
                for _ in range(8):
                    K[np.diag_indices_from(K)] += jitter - added
                    added = jitter
                    try:
                        fact = cholesky(K, lower=True)
                        break
                    except np.linalg.LinAlgError:
                        jitter *= 10.0
                if fact is None:
                    raise np.linalg.LinAlgError(
                        "Cholesky decomposition of the covariance matrix "
                        "failed even with a diagonal jitter of %.3e (%s). "
                        "Increasing white_noise can help." % (added, str(e))
                    )
                warnings.warn(
                    "The covariance matrix of the tabulated kernel is not "
                    "positive semi-definite: a diagonal jitter of %.3e "
                    "(%.2f%% of its mean diagonal) was added to make the "
                    "Cholesky factorization succeed." % (added, 100.0 * added / scale)
                )
                self._fact = fact
            self._alpha = cho_solve((self._fact, True), y)
        y_predict = np.dot(HT, self._alpha.reshape((len(self._alpha), 1))).T[0]
        if return_cov:
            v = cho_solve((self._fact, True), HT.T)
            y_cov = kernel.__call__(X2) - HT.dot(v)
            return y_predict, y_cov
        else:
            return y_predict, None

    def initialize(self, X, y, y_err=None):
        """Initialize both the interpolator to some state prefatory to any solve iterations and
        initialize the field values for use with this interpolator.

        :param X:     The coodinates of the field.  (n_samples, 1 or 2)
        :param y:     Values of the field.  (n_samples)
        :param y_err: Error of y. (n_samples)
        """
        self.kernel = copy.deepcopy(self.kernel_template)

        self._X = X
        self._y = y
        if y_err is None:
            y_err = np.zeros_like(y)
        self._y_err = y_err

        if self._X0 is None:
            self._X0 = np.zeros_like(self._X)
            self._y0 = np.zeros_like(self._y)
        self._spatial_average = self._build_average_meanify(X)

        if self.white_noise > 0:
            y_err = np.sqrt(copy.deepcopy(self._y_err) ** 2 + self.white_noise**2)
        self._y_err = y_err

        if self.normalize:
            self._mean = np.mean(y - self._spatial_average)
        else:
            self._mean = 0.0
        # Initialize alpha to None so that we know to recompute it if we change the
        # input data.
        self._alpha = None
        self._fact = None
        if self._engine is not None:
            self._engine.reset()

    def _build_average_meanify(self, X):
        """Compute spatial average from meanify output for a given coordinate using KN interpolation.
        If no average_fits was given, return array of 0.

        :param X: Coordinates of training coordinates where to interpolate. (n_samples, 1 or 2)
        """
        if np.sum(np.equal(self._X0, 0)) != len(self._X0[:, 0]) * len(self._X0[0]):
            neigh = KNeighborsRegressor(n_neighbors=self.n_neighbors)
            neigh.fit(self._X0, self._y0)
            average = neigh.predict(X)
            if self.indice_meanify is not None:
                average = average[:, self.indice_meanify]
            return average
        else:
            return np.zeros((len(X[:, 0])))

    def solve(self):
        """Set up this GPInterp object.
        Solve for hyperparameters if requested using 2-point correlation
        function method or maximum likelihood.
        """
        self._init_theta = []
        kernel = copy.deepcopy(self.kernel)
        self._init_theta.append(kernel.theta)
        self.kernel = self._fit(
            self.kernel,
            self._X,
            self._y - self._mean - self._spatial_average,
            self._y_err,
        )

    def return_2pcf(self):
        """
        Return 2-point correlation function and its variance using Bootstrap.
        """
        if self.optimizer == "empirical-2pcf":
            raise NotImplementedError(
                "return_2pcf is not available for the empirical-2pcf optimizer. "
                "Use return_empirical_2pcf instead."
            )
        anisotropic = self.optimizer == "anisotropic"
        pcf = treegp.two_pcf(
            self._X,
            self._y - self._mean - self._spatial_average,
            self._y_err,
            self.min_sep,
            self.max_sep,
            nbins=self.nbins,
            anisotropic=anisotropic,
        )
        xi, xi_weight, distance, coord, mask = pcf.return_2pcf()
        return xi, xi_weight, distance, coord, mask

    def return_empirical_2pcf(self):
        """
        Return the measured and cleaned 2d 2-point correlation functions
        used as kernel by the empirical-2pcf optimizer, the lag coordinates
        of the grid, and the pixel size of the grid.

        Returns xi, xi_clean, distance, pixel_size where xi and xi_clean
        are (npix, npix) arrays with zero lag at pixel npix//2, and
        distance is a (npix*npix, 2) array of the (dx, dy) lags matching
        the flattened correlation functions.
        """
        if self.optimizer != "empirical-2pcf":
            raise NotImplementedError(
                "return_empirical_2pcf is only available for the "
                "empirical-2pcf optimizer. Current optimizer: %s" % (self.optimizer)
            )
        if not hasattr(self, "_optimizer"):
            raise RuntimeError("solve() must be called before return_empirical_2pcf.")
        return (
            self._optimizer._xi,
            self._optimizer._xi_clean,
            self._optimizer._2pcf_dist,
            self._optimizer.pixel_size,
        )

    def return_log_likelihood(self, theta=None):
        """
        Return of log likehood of gaussian process
        for given hyperparameters.

        :param theta: Array of hyperparamters. (default: None)
        """
        kernel = copy.deepcopy(self.kernel)
        if theta is not None:
            kernel = kernel.clone_with_theta(theta)
        logl = treegp.log_likelihood(
            self._X, self._y - self._mean - self._spatial_average, self._y_err
        )
        log_likelihood = logl.log_likelihood(kernel)
        return log_likelihood

    def plot_fitted_kernel(self):

        if self.optimizer in ["none", "log-likehood", "two-pcf"]:
            raise NotImplementedError(
                "This method is only available for anisotropic optimizer"
            )
        import os

        if os.getenv("GITHUB_ACTIONS") == "true":
            # Code is running under GitHub Actions
            import matplotlib

            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        EXT = [
            np.min(self._optimizer._2pcf_dist[:, 0]),
            np.max(self._optimizer._2pcf_dist[:, 0]),
            np.min(self._optimizer._2pcf_dist[:, 1]),
            np.max(self._optimizer._2pcf_dist[:, 1]),
        ]

        CM = plt.cm.seismic

        MAX = np.max(self._optimizer._2pcf)
        N = int(np.sqrt(len(self._optimizer._2pcf)))
        fig = plt.figure(figsize=(14, 4))
        plt.subplots_adjust(wspace=0.5, left=0.07, right=0.95, bottom=0.1, top=0.92)

        plt.subplot(1, 3, 1)
        plt.imshow(
            self._optimizer._2pcf.reshape(N, N),
            extent=EXT,
            interpolation="nearest",
            origin="lower",
            vmin=-MAX,
            vmax=MAX,
            cmap=CM,
        )
        cbar = plt.colorbar()
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        cbar.set_label("$\\xi$", fontsize=16)
        plt.xlabel(r"$\Delta x$", fontsize=16)
        plt.ylabel(r"$\Delta y$", fontsize=16)
        plt.title("Measured 2-PCF", fontsize=16)

        plt.subplot(1, 3, 2)
        plt.imshow(
            self._optimizer._2pcf_fit.reshape(N, N),
            extent=EXT,
            interpolation="nearest",
            origin="lower",
            vmin=-MAX,
            vmax=MAX,
            cmap=CM,
        )
        cbar = plt.colorbar()
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        cbar.set_label("$\\xi'$", fontsize=16)
        plt.xlabel(r"$\Delta x$", fontsize=16)
        plt.title("Fitted 2-PCF", fontsize=16)

        plt.subplot(1, 3, 3)
        residuals = self._optimizer._2pcf.reshape(
            N, N
        ) - self._optimizer._2pcf_fit.reshape(N, N)
        plt.imshow(
            residuals,
            extent=EXT,
            interpolation="nearest",
            origin="lower",
            vmin=-MAX,
            vmax=MAX,
            cmap=CM,
        )
        cbar = plt.colorbar()
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        cbar.set_label("$\\xi - \\xi'$", fontsize=16)
        plt.xlabel(r"$\Delta x$", fontsize=16)
        plt.title("Difference", fontsize=16)

        return fig
