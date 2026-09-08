"""
.. module:: kernel
"""

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy import special
from scipy.interpolate import RegularGridInterpolator
from sklearn.gaussian_process.kernels import (
    StationaryKernelMixin,
    NormalizedKernelMixin,
    Kernel,
)
from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.gaussian_process.kernels import _check_length_scale


def eval_kernel(kernel):
    """
    Some import trickery to get all subclasses
    of sklearn.gaussian_process.kernels.Kernel
    into the local namespace without doing
    "from sklearn.gaussian_process.kernels import *"
    and without importing them all manually.

    Example:
    kernel = eval_kernel("RBF(1)") instead of
    kernel = sklearn.gaussian_process.kernels.RBF(1)
    """

    namespace = {}

    def recurse_subclasses(cls):
        out = []
        for c in cls.__subclasses__():
            out.append(c)
            out.extend(recurse_subclasses(c))
        return out

    clses = recurse_subclasses(Kernel)
    for cls in clses:
        module = __import__(cls.__module__, globals(), namespace, [cls.__name__])
        namespace[cls.__name__] = getattr(module, cls.__name__)

    from numpy import array  # if needed in the namespace

    # Add array to the namespace if necessary
    namespace["array"] = array

    try:
        k = eval(kernel, namespace)
    except Exception as e:
        raise RuntimeError(
            "Failed to evaluate kernel string {0!r}.  "
            "Original exception: {1}".format(kernel, e)
        )

    if isinstance(k.theta, property):
        raise TypeError("String provided was not initialized properly")
    return k


class AnisotropicRBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """A GaussianProcessRegressor Kernel representing a radial basis function (essentially a
    squared exponential or Gaussian) but with arbitrary anisotropic covariance.
    While the parameter for this kernel, an inverse covariance matrix, can be specified directly
    with the `invLam` kwarg, it may be more convenient to instead specify a characteristic
    scale-length for each axis using the `scale_length` kwarg.  Note that a list or array is
    required so that the dimensionality of the kernel can be determined from its length.
    For optimization, it's necessary to reparameterize the inverse covariance matrix in such a way
    as to ensure that it's always positive definite.  To this end, we define `theta` (abbreviated
    `th` below) such that
    invLam = L * L.T
    L = [[exp(th[0])  0              0           ...    0                 0           ]
         [th[n]       exp(th[1])]    0           ...    0                 0           ]
         [th[n+1]     th[n+2]        exp(th[3])  ...    0                 0           ]
         [...         ...            ...         ...    ...               ...         ]
         [th[]        th[]           th[]        ...    exp(th[n-2])      0           ]
         [th[]        th[]           th[]        ...    th[n*(n+1)/2-1]   exp(th[n-1])]]
    I.e., the inverse covariance matrix is Cholesky-decomposed, exp(theta[0:n]) lie on the diagonal
    of the Cholesky matrix, and theta[n:n*(n+1)/2] lie in the lower triangular part of the Cholesky
    matrix.  This parameterization invertably maps all valid n x n covariance matrices to
    R^(n*(n+1)/2).  I.e., the range of each theta[i] is -inf...inf.
    :param  invLam:  Inverse covariance matrix of radial basis function.  Exactly one of invLam and
                     scale_length must be provided.
    :param  scale_length:  Axes-aligned scale lengths of the kernel.  len(scale_length) must be the
                     same as the dimensionality of the kernel, even if the scale length is the same
                     for each axis (i.e., even if the kernel is isotropic).  Exactly one of invLam
                     and scale_length must be provided.
    :param  bounds:  Optional keyword indicating fitting bounds on *theta*.  Can either be a
                     2-element iterable, which will be taken to be the min and max value for every
                     theta element, or an [ntheta, 2] array indicating bounds on each of ntheta
                     elements.
    """

    def __init__(self, invLam=None, scale_length=None, bounds=(-5, 5)):
        if scale_length is not None:
            if invLam is not None:
                raise TypeError(
                    "Cannot set both invLam and scale_length in AnisotropicRBF."
                )
            invLam = np.diag(1.0 / np.array(scale_length) ** 2)

        self.ndim = invLam.shape[0]
        self.ntheta = self.ndim * (self.ndim + 1) // 2
        self._d = np.diag_indices(self.ndim)
        self._t = np.tril_indices(self.ndim, -1)
        self.set_params(invLam)
        bounds = np.array(bounds)
        if bounds.ndim == 1:
            bounds = np.repeat(bounds[None, :], self.ntheta, axis=0)
        assert bounds.shape == (self.ntheta, 2)
        self._bounds = bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)

        if Y is None:
            dists = pdist(X, metric="mahalanobis", VI=self.invLam)
            K = np.exp(-0.5 * dists**2)
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            dists = cdist(X, Y, metric="mahalanobis", VI=self.invLam)
            K = np.exp(-0.5 * dists**2)

        if eval_gradient:
            if self.hyperparameter_cholesky_factor.fixed:
                return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                # dK_pq/dth_k = -0.5 * K_pq *
                #               ((x_p_i-x_q_i) * dInvLam_ij/dth_k * (x_q_j - x_q_j))
                # dInvLam_ij/dth_k = dL_ij/dth_k * L_ij.T  +  L_ij * dL_ij.T/dth_k
                # dL_ij/dth_k is a matrix with all zeros except for one element.  That element is
                # L_ij if k indicates one of the theta parameters landing on the Cholesky diagonal,
                # and is 1.0 if k indicates one of the thetas in the lower triangular region.
                L_grad = np.zeros((self.ntheta, self.ndim, self.ndim), dtype=float)
                L_grad[(np.arange(self.ndim),) + self._d] = self._L[self._d]
                L_grad[(np.arange(self.ndim, self.ntheta),) + self._t] = 1.0

                half_invLam_grad = np.dot(L_grad, self._L.T)
                invLam_grad = half_invLam_grad + np.transpose(
                    half_invLam_grad, (0, 2, 1)
                )

                dX = X[:, np.newaxis, :] - X[np.newaxis, :, :]
                dist_grad = np.einsum("ijk,lkm,ijm->ijl", dX, invLam_grad, dX)
                K_gradient = -0.5 * K[:, :, np.newaxis] * dist_grad
                return K, K_gradient
        else:
            return K

    @property
    def hyperparameter_cholesky_factor(self):
        return Hyperparameter(
            "CholeskyFactor", "numeric", (1e-5, 1e5), int(self.ntheta)
        )

    def get_params(self, deep=True):
        return {"invLam": self.invLam}

    def set_params(self, invLam=None):
        if invLam is not None:
            self.invLam = invLam
            self._L = np.linalg.cholesky(self.invLam)
            self._theta = np.hstack([np.log(self._L[self._d]), self._L[self._t]])

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta
        self._L = np.zeros_like(self.invLam)
        self._L[np.diag_indices(self.ndim)] = np.exp(theta[: self.ndim])
        self._L[np.tril_indices(self.ndim, -1)] = theta[self.ndim :]
        self.invLam = np.dot(self._L, self._L.T)

    def __repr__(self):
        return "{0}(invLam={1!r})".format(self.__class__.__name__, self.invLam)

    @property
    def bounds(self):
        return self._bounds


class VonKarman(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """VonKarman kernel.

    Parameters
    -----------
    length_scale : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.

    length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on length_scale
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                "length_scale",
                "numeric",
                self.length_scale_bounds,
                len(self.length_scale),
            )
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """

        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = pdist(X, metric="euclidean")
            Filter = dists != 0.0
            K = np.zeros_like(dists)
            K[Filter] = (dists[Filter] / length_scale) ** (5.0 / 6.0) * special.kv(
                5.0 / 6.0, 2 * np.pi * dists[Filter] / length_scale
            )
            K = squareform(K)

            lim0 = special.gamma(5.0 / 6.0) / (2 * (np.pi ** (5.0 / 6.0)))
            np.fill_diagonal(K, lim0)
            K /= lim0
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")

            dists = cdist(X, Y, metric="euclidean")
            Filter = dists != 0.0
            K = np.zeros_like(dists)
            K[Filter] = (dists[Filter] / length_scale) ** (5.0 / 6.0) * special.kv(
                5.0 / 6.0, 2 * np.pi * dists[Filter] / length_scale
            )
            lim0 = special.gamma(5.0 / 6.0) / (2 * (np.pi ** (5.0 / 6.0)))
            if np.sum(Filter) != len(K[0]) * len(K[:, 0]):
                K[~Filter] = lim0
            K /= lim0

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                raise ValueError(
                    "Gradient can only be evaluated with isotropic VonKarman kernel for the moment."
                )
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
            )
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0]
            )


class AnisotropicVonKarman(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """A GaussianProcessRegressor Kernel representing a Von-Karman correlation function
    with an arbitrary anisotropic covariance. While the parameter for this kernel,
    an inverse covariance matrix, can be specified directly with the `invLam` kwarg,
    it may be more convenient to instead specify a characteristic scale-length for each axis
    using the `scale_length` kwarg.  Note that a list or array is required so that the dimensionality
    of the kernel can be determined from its length. For optimization, it's necessary to reparameterize
    the inverse covariance matrix in such a way as to ensure that it's always positive definite.
    To this end, we define `theta` (abbreviated `th` below) such that
    invLam = L * L.T
    L = [[exp(th[0])  0              0           ...    0                 0           ]
         [th[n]       exp(th[1])]    0           ...    0                 0           ]
         [th[n+1]     th[n+2]        exp(th[3])  ...    0                 0           ]
         [...         ...            ...         ...    ...               ...         ]
         [th[]        th[]           th[]        ...    exp(th[n-2])      0           ]
         [th[]        th[]           th[]        ...    th[n*(n+1)/2-1]   exp(th[n-1])]]
    I.e., the inverse covariance matrix is Cholesky-decomposed, exp(theta[0:n]) lie on the diagonal
    of the Cholesky matrix, and theta[n:n*(n+1)/2] lie in the lower triangular part of the Cholesky
    matrix.  This parameterization invertably maps all valid n x n covariance matrices to
    R^(n*(n+1)/2).  I.e., the range of each theta[i] is -inf...inf.
    :param  invLam:  Inverse covariance matrix of radial basis function.  Exactly one of invLam and
                     scale_length must be provided.
    :param  scale_length:  Axes-aligned scale lengths of the kernel.  len(scale_length) must be the
                     same as the dimensionality of the kernel, even if the scale length is the same
                     for each axis (i.e., even if the kernel is isotropic).  Exactly one of invLam
                     and scale_length must be provided.
    :param  bounds:  Optional keyword indicating fitting bounds on *theta*.  Can either be a
                     2-element iterable, which will be taken to be the min and max value for every
                     theta element, or an [ntheta, 2] array indicating bounds on each of ntheta
                     elements.
    """

    def __init__(self, invLam=None, scale_length=None, bounds=(-5, 5)):
        if scale_length is not None:
            if invLam is not None:
                raise TypeError(
                    "Cannot set both invLam and scale_length in AnisotropicVonKarman."
                )
            invLam = np.diag(1.0 / np.array(scale_length) ** 2)

        self.ndim = invLam.shape[0]
        self.ntheta = self.ndim * (self.ndim + 1) // 2
        self._d = np.diag_indices(self.ndim)
        self._t = np.tril_indices(self.ndim, -1)
        self.set_params(invLam)
        bounds = np.array(bounds)
        if bounds.ndim == 1:
            bounds = np.repeat(bounds[None, :], self.ntheta, axis=0)
        assert bounds.shape == (self.ntheta, 2)
        self._bounds = bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)

        if Y is None:
            dists = pdist(X, metric="mahalanobis", VI=self.invLam)
            Filter = dists != 0.0
            K = np.zeros_like(dists)
            K[Filter] = dists[Filter] ** (5.0 / 6.0) * special.kv(
                5.0 / 6.0, 2 * np.pi * dists[Filter]
            )
            lim0 = special.gamma(5.0 / 6.0) / (2 * ((np.pi) ** (5.0 / 6.0)))
            K = squareform(K)
            np.fill_diagonal(K, lim0)
            K /= lim0
        else:
            if eval_gradient:
                raise ValueError("Gradient can not be evaluated.")
            dists = cdist(X, Y, metric="mahalanobis", VI=self.invLam)
            Filter = dists != 0.0
            K = np.zeros_like(dists)
            K[Filter] = dists[Filter] ** (5.0 / 6.0) * special.kv(
                5.0 / 6.0, 2 * np.pi * dists[Filter]
            )
            lim0 = special.gamma(5.0 / 6.0) / (2 * ((np.pi) ** (5.0 / 6.0)))
            if np.sum(Filter) != len(K[0]) * len(K[:, 0]):
                K[~Filter] = lim0
            K /= lim0

        if eval_gradient:
            raise ValueError("Gradient can not be evaluated.")
        else:
            return K

    @property
    def hyperparameter_cholesky_factor(self):
        return Hyperparameter(
            "CholeskyFactor", "numeric", (1e-5, 1e5), int(self.ntheta)
        )

    def get_params(self, deep=True):
        return {"invLam": self.invLam}

    def set_params(self, invLam=None):
        if invLam is not None:
            self.invLam = invLam
            self._L = np.linalg.cholesky(self.invLam)
            self._theta = np.hstack([np.log(self._L[self._d]), self._L[self._t]])

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta
        self._L = np.zeros_like(self.invLam)
        self._L[np.diag_indices(self.ndim)] = np.exp(theta[: self.ndim])
        self._L[np.tril_indices(self.ndim, -1)] = theta[self.ndim :]
        self.invLam = np.dot(self._L, self._L.T)

    def __repr__(self):
        return "{0}(invLam={1!r})".format(self.__class__.__name__, self.invLam)

    @property
    def bounds(self):
        return self._bounds


class EmpiricalCorrelationKernel(StationaryKernelMixin, Kernel):
    """A tabulated stationary anisotropic kernel built from a measured
    2d 2-point correlation function, following Gomes et al. (2025)
    (AJ 170:361, doi:10.3847/1538-3881/ae1a7b).

    The kernel has no free hyperparameters (theta is empty): it evaluates
    K(X, Y)[i, j] = xi(X_i - Y_j) by bilinear interpolation of the given
    correlation function grid, and is zero beyond the grid (compact
    support).

    A tabulated correlation function is not guaranteed to be positive
    semi-definite between arbitrary points, so by default the covariance
    matrices built when Y is None are projected onto the closest positive
    semi-definite matrix by clipping their negative eigenvalues to zero.
    This is equivalent to the singular value clipping used by
    Gomes et al. (2025).

    Input is expected to be 2-dimensional, i.e. X.shape = (n_samples, 2).

    :param x_grid:  Lag coordinates of the grid columns, zero lag
                    at index len(x_grid)//2. (nx,) ndarray
    :param y_grid:  Lag coordinates of the grid rows, zero lag
                    at index len(y_grid)//2. (ny,) ndarray
    :param xi_grid: 2d correlation function in treecorr TwoD layout,
                    i.e. indexed [iy, ix]. (ny, nx) ndarray
    :param clip_eigenvalues: Whether to clip the negative eigenvalues of
                    the covariance matrices built when Y is None.
                    [default: True]
    """

    # Maximum number of pair lags interpolated at once by __call__, to
    # bound the size of the temporary arrays (a few 100 MB) instead of
    # scaling with n_samples^2.
    _chunk_size = 4_000_000

    def __init__(self, x_grid, y_grid, xi_grid, clip_eigenvalues=True):
        self.x_grid = np.asarray(x_grid)
        self.y_grid = np.asarray(y_grid)
        self.xi_grid = np.asarray(xi_grid)
        self.clip_eigenvalues = clip_eigenvalues
        if self.xi_grid.shape != (len(self.y_grid), len(self.x_grid)):
            raise ValueError(
                "xi_grid shape %s does not match (len(y_grid), len(x_grid)) = %s"
                % (str(self.xi_grid.shape), str((len(self.y_grid), len(self.x_grid))))
            )
        # xi_grid is indexed [iy, ix], while the interpolation axes are
        # (x lag, y lag), hence the transpose.
        self._z = np.ascontiguousarray(self.xi_grid.T)
        # Uniform grids (the empirical_2pcf solver always produces them)
        # take a direct vectorized bilinear interpolation, much faster
        # than the generic RegularGridInterpolator, which is kept as a
        # fallback for non-uniform grids.
        steps_x = np.diff(self.x_grid)
        steps_y = np.diff(self.y_grid)
        uniform = (
            len(steps_x) > 0
            and len(steps_y) > 0
            and np.all(steps_x > 0)
            and np.all(steps_y > 0)
            and np.allclose(steps_x, steps_x[0])
            and np.allclose(steps_y, steps_y[0])
        )
        if uniform:
            self._interp = None
            self._hx = steps_x[0]
            self._hy = steps_y[0]
        else:
            self._interp = RegularGridInterpolator(
                (self.x_grid, self.y_grid),
                self._z,
                method="linear",
                bounds_error=False,
                fill_value=0.0,
            )

    def _eval_lags(self, dx, dy):
        """Bilinear interpolation of the tabulated correlation function
        at the given (dx, dy) lags, zero outside the grid.

        :param dx, dy: Lags where to evaluate the correlation function.
                       ndarrays of matching shape.
        """
        if self._interp is not None:
            return self._interp(np.stack([dx, dy], axis=-1))
        nx = len(self.x_grid)
        ny = len(self.y_grid)
        gx = (dx - self.x_grid[0]) / self._hx
        gy = (dy - self.y_grid[0]) / self._hy
        inside = (gx >= 0.0) & (gx <= nx - 1) & (gy >= 0.0) & (gy <= ny - 1)
        ix = np.clip(np.floor(gx).astype(np.intp), 0, nx - 2)
        iy = np.clip(np.floor(gy).astype(np.intp), 0, ny - 2)
        tx = gx - ix
        ty = gy - iy
        z = self._z
        out = (
            (1.0 - tx) * (1.0 - ty) * z[ix, iy]
            + tx * (1.0 - ty) * z[ix + 1, iy]
            + (1.0 - tx) * ty * z[ix, iy + 1]
            + tx * ty * z[ix + 1, iy + 1]
        )
        return np.where(inside, out, 0.0)

    def _pair_covariance(self, X, Y):
        """Evaluate K[i, j] = xi(X_i - Y_j) in chunks of rows, so that
        the temporary arrays never scale with len(X) * len(Y)."""
        n1 = len(X)
        n2 = len(Y)
        K = np.empty((n1, n2))
        chunk = max(1, self._chunk_size // max(n2, 1))
        for start in range(0, n1, chunk):
            end = min(start + chunk, n1)
            dx = X[start:end, 0][:, np.newaxis] - Y[np.newaxis, :, 0]
            dy = X[start:end, 1][:, np.newaxis] - Y[np.newaxis, :, 1]
            K[start:end] = self._eval_lags(dx, dy)
        return K

    @property
    def xi0(self):
        """Zero-lag value of the correlation function, i.e. the variance
        of the field."""
        return float(self._eval_lags(np.zeros(1), np.zeros(1))[0])

    def __call__(self, X, Y=None, eval_gradient=False):
        if eval_gradient:
            raise ValueError("Gradient can not be evaluated.")
        X = np.atleast_2d(X)
        if np.shape(X)[1] != 2:
            raise ValueError(
                "EmpiricalCorrelationKernel supports only 2d coordinates. "
                "Current ndim: %i" % (np.shape(X)[1])
            )
        if Y is None:
            K = self._pair_covariance(X, X)
            # The tabulated correlation function is point-symmetric except
            # for its first row/column (the most negative lag has no
            # positive counterpart on the grid), so symmetrize to get an
            # exactly symmetric covariance matrix.
            K = 0.5 * (K + K.T)
            if self.clip_eigenvalues:
                eigenvalues, eigenvectors = np.linalg.eigh(K)
                if np.any(eigenvalues < 0.0):
                    K = (eigenvectors * np.clip(eigenvalues, 0.0, None)).dot(
                        eigenvectors.T
                    )
                    K = 0.5 * (K + K.T)
        else:
            Y = np.atleast_2d(Y)
            K = self._pair_covariance(X, Y)
        return K

    def diag(self, X):
        return np.full(len(X), self.xi0)

    def __repr__(self):
        return "{0}(npix={1!r}, xi0={2:.4g})".format(
            self.__class__.__name__, self.xi_grid.shape, self.xi0
        )
