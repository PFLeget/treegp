"""
.. module:: grid_gp
"""

import warnings
import numpy as np

from scipy import fft
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg


def _upsample_map(xi, upsample):
    """Upsample a 2d map by an integer factor using exact trigonometric
    (Fourier zero-padding) interpolation. The input map is assumed to be
    square with even side length and centered on pixel N//2 (treecorr
    TwoD layout after _shift_and_bin); the output is centered on pixel
    (N * upsample) // 2 and matches the input exactly on the input
    nodes. Exact for band-limited maps, which the cleaned correlation
    function is by construction (its Fourier power spectrum was
    thresholded on the same grid).

    :param xi:       2d map, zero lag at pixel N//2. (N, N) ndarray
    :param upsample: Integer upsampling factor. [required]
    """
    if upsample == 1:
        return xi.copy()
    n = xi.shape[0]
    m = n * upsample
    # Spectrum with the frequency origin at the center, frequencies
    # running from -n/2 to n/2 - 1 in each axis.
    f_shift = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(xi)))
    # Split the (unpaired) Nyquist row and column between -n/2 and
    # +n/2 so that the interpolant of a real map is real.
    f_ext = np.zeros((n + 1, n + 1), dtype=complex)
    f_ext[:n, :n] = f_shift
    f_ext[n, :n] = f_shift[0, :]
    f_ext[:n, n] = f_shift[:, 0]
    f_ext[n, n] = f_shift[0, 0]
    f_ext[0, :] *= 0.5
    f_ext[n, :] *= 0.5
    f_ext[:, 0] *= 0.5
    f_ext[:, n] *= 0.5
    # Embed into the fine-grid spectrum, frequencies -m/2 to m/2 - 1.
    pad = m // 2 - n // 2
    f_big = np.zeros((m, m), dtype=complex)
    f_big[pad : pad + n + 1, pad : pad + n + 1] = f_ext
    out = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(f_big)))
    return out.real * upsample**2


def _symmetrize_kernel_image(xi):
    """Return a point-symmetric copy of a kernel image, extended by one
    row and column. The input is a square even-sized map with zero lag
    at pixel N//2, whose first row and column (the most negative lag)
    have no positive counterpart on the grid; the output has odd side
    length N + 1, zero lag exactly at its center, and satisfies
    out[c + i, c + j] == out[c - i, c - j]. This is the grid-space
    equivalent of the 0.5 * (K + K.T) symmetrization used for the dense
    covariance matrices (the unpaired edge lags end up halved, exactly
    as in the dense case where their mirror image falls outside the
    grid and evaluates to zero), and makes the Fourier transform of the
    image exactly real.

    :param xi: 2d kernel image, zero lag at pixel N//2. (N, N) ndarray
    """
    n = xi.shape[0]
    ext = np.zeros((n + 1, n + 1))
    ext[:n, :n] = xi
    return 0.5 * (ext + ext[::-1, ::-1])


class GridConvolutionGP(object):
    """Fast, low-memory solve and predict engine for the empirical-2pcf
    (Gomes et al. 2025) kernel, exploiting its grid structure instead of
    dense linear algebra.

    The training covariance is represented as K = W G W^T, where W is
    the sparse bilinear (tent) spreading matrix of the points onto a
    uniform fine grid and G is the circular convolution by the kernel
    image on a zero-padded grid, diagonalized by FFT. The negative
    Fourier modes of the padded kernel image are clipped to zero once,
    in Fourier space, which makes the effective kernel
    (tent * xi * tent, with a non-negative spectrum) continuously
    positive semi-definite: the covariance of ANY point set is PSD by
    construction, replacing the O(N^3) per-point-set eigenvalue clipping
    of the dense path. The padded grid is large enough that the circular
    wrap-around never reaches a real pair separation.

    (K + diag(y_err^2)) alpha = y is solved by Jacobi-preconditioned
    conjugate gradient, where each matvec costs one FFT pair on the
    padded grid: O(n_grid log n_grid + n_points) time and memory instead
    of O(N^3) time and O(N^2) memory. The posterior mean is then a
    single field, ŷ(x) = sum_j alpha_j k(x - x_j), computed once as a
    convolution on the grid; each predict call only bilinearly samples
    that precomputed field, so its cost is independent of the number of
    training points (useful when predicting per CCD on a full visit).

    Compared to the dense path, the kernel is smoothed by the tent
    spreading (one fine-grid pixel wide, i.e. pixel_size / upsample) and
    the PSD projection is done on the padded grid instead of on the
    point-set covariance; both effects are second order and shrink with
    upsample.

    :param xi_grid:    Cleaned 2d correlation function in treecorr TwoD
                       layout ([iy, ix]), zero lag at pixel npix//2, as
                       produced by empirical_2pcf. (npix, npix) ndarray
    :param pixel_size: Pixel size of xi_grid, in the same units as the
                       coordinates of the field.
    :param upsample:   Integer upsampling factor of the spreading grid
                       with respect to the xi_grid pixels. The kernel
                       image is upsampled exactly (Fourier zero-padding,
                       xi_grid is band-limited by construction), so
                       upsample only controls the tent smoothing scale
                       and the grid memory. [default: 2]
    :param cg_rtol:    Relative tolerance of the conjugate gradient
                       solve. [default: 1e-7]
    :param cg_maxiter: Maximum number of conjugate gradient iterations.
                       [default: 500]
    """

    def __init__(self, xi_grid, pixel_size, upsample=2, cg_rtol=1e-7, cg_maxiter=500):
        xi_grid = np.asarray(xi_grid, dtype=float)
        if xi_grid.ndim != 2 or xi_grid.shape[0] != xi_grid.shape[1]:
            raise ValueError(
                "xi_grid must be square. Current shape: %s" % (str(xi_grid.shape))
            )
        if xi_grid.shape[0] % 2 != 0:
            raise ValueError(
                "xi_grid side length must be even. Current shape: %s"
                % (str(xi_grid.shape))
            )
        if not isinstance(upsample, (int, np.integer)) or upsample < 1:
            raise ValueError(
                "upsample must be a positive integer. Current value: %s" % (upsample)
            )
        self.pixel_size = float(pixel_size)
        self.upsample = int(upsample)
        self.cg_rtol = float(cg_rtol)
        self.cg_maxiter = int(cg_maxiter)
        # Half width of the kernel support, from the zero lag to the
        # grid edge.
        self.max_sep = (xi_grid.shape[0] // 2) * self.pixel_size
        # Fine grid spacing and exactly-upsampled, symmetrized kernel
        # image (odd side length, zero lag at its center).
        self.h = self.pixel_size / self.upsample
        self._kernel_image = _symmetrize_kernel_image(
            _upsample_map(xi_grid, self.upsample)
        )
        self.reset()

    def reset(self):
        """Forget the current solution (and the training-set geometry),
        so that the next predict triggers a new solve."""
        self._alpha = None
        self._mean_field = None
        self._nx = None
        self._ny = None
        self.n_iterations = None
        self.spectrum_min = None
        self.xi0_eff = None

    def _setup_geometry(self, X):
        """Build the fine spreading grid covering the training points
        plus the kernel support, the padded FFT grid, and the clipped
        non-negative kernel spectrum on it."""
        ks = self._kernel_image.shape[0]
        margin = self.max_sep + self.h
        self._x0 = np.min(X[:, 0]) - margin
        self._y0 = np.min(X[:, 1]) - margin
        # Number of grid nodes covering the domain (bilinear spreading
        # needs one node beyond the last point in each axis).
        self._nx = int(np.ceil((np.max(X[:, 0]) - self._x0) / self.h)) + 2
        self._ny = int(np.ceil((np.max(X[:, 1]) - self._y0) / self.h)) + 2
        # Padded FFT grid: large enough for a linear (wrap-free)
        # convolution of the field with the kernel image.
        self._lx = fft.next_fast_len(self._nx + ks - 1, real=True)
        self._ly = fft.next_fast_len(self._ny + ks - 1, real=True)
        # Kernel image with its zero lag rolled to pixel (0, 0) of the
        # padded grid, and its spectrum, exactly real thanks to the
        # point symmetry of the image. Clipping the negative modes to
        # zero is the one-time PSD projection.
        buf = np.zeros((self._ly, self._lx))
        buf[:ks, :ks] = self._kernel_image
        buf = np.roll(buf, (-(ks // 2), -(ks // 2)), axis=(0, 1))
        spectrum = fft.rfft2(buf).real
        self.spectrum_min = float(np.min(spectrum))
        self._spectrum = np.clip(spectrum, 0.0, None)
        # Effective zero-lag variance of the clipped kernel.
        image_clip = fft.irfft2(self._spectrum, s=(self._ly, self._lx))
        self.xi0_eff = float(image_clip[0, 0])

    def _spread_matrix(self, X, name="X"):
        """Sparse bilinear (tent) spreading matrix of the given points
        onto the fine grid: W[i, iy * nx + ix] holds the weight of point
        i on grid node (iy, ix). Points must be inside the grid."""
        gx = (X[:, 0] - self._x0) / self.h
        gy = (X[:, 1] - self._y0) / self.h
        if np.any(gx < 0) or np.any(gx > self._nx - 2):
            raise ValueError("Some %s coordinates fall outside the grid." % (name))
        if np.any(gy < 0) or np.any(gy > self._ny - 2):
            raise ValueError("Some %s coordinates fall outside the grid." % (name))
        ix = np.floor(gx).astype(np.intp)
        iy = np.floor(gy).astype(np.intp)
        tx = gx - ix
        ty = gy - iy
        n = len(X)
        rows = np.repeat(np.arange(n), 4)
        cols = np.empty((n, 4), dtype=np.intp)
        cols[:, 0] = iy * self._nx + ix
        cols[:, 1] = iy * self._nx + ix + 1
        cols[:, 2] = (iy + 1) * self._nx + ix
        cols[:, 3] = (iy + 1) * self._nx + ix + 1
        vals = np.empty((n, 4))
        vals[:, 0] = (1.0 - tx) * (1.0 - ty)
        vals[:, 1] = tx * (1.0 - ty)
        vals[:, 2] = (1.0 - tx) * ty
        vals[:, 3] = tx * ty
        return sparse.csr_matrix(
            (vals.ravel(), (rows, cols.ravel())),
            shape=(n, self._nx * self._ny),
        )

    def _convolve(self, field):
        """Convolve a field given on the (ny, nx) grid nodes with the
        clipped kernel image, through the zero-padded FFT grid."""
        buf = np.zeros((self._ly, self._lx))
        buf[: self._ny, : self._nx] = field
        out = fft.irfft2(fft.rfft2(buf) * self._spectrum, s=(self._ly, self._lx))
        return out[: self._ny, : self._nx]

    def solve(self, X, y, y_err):
        """Solve (K + diag(y_err^2)) alpha = y by preconditioned
        conjugate gradient, where K is the PSD-by-construction grid
        kernel evaluated between the training points.

        :param X:     Coordinates of the field. (n_samples, 2)
        :param y:     Values of the field, already centered (mean and
                      spatial average subtracted). (n_samples)
        :param y_err: Error of y. (n_samples)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        y_err = np.asarray(y_err, dtype=float)
        self.reset()
        self._setup_geometry(X)
        self._w_train = self._spread_matrix(X, name="training")
        noise = y_err**2
        if np.min(noise) <= 0.0:
            warnings.warn(
                "Some points have zero error: K + diag(y_err^2) is only "
                "positive SEMI-definite and the conjugate gradient solve "
                "may struggle to converge; setting white_noise > 0 can help."
            )

        w = self._w_train

        def matvec(v):
            field = (w.T @ v).reshape(self._ny, self._nx)
            return w @ self._convolve(field).ravel() + noise * v

        n = len(y)
        operator = LinearOperator((n, n), matvec=matvec, dtype=float)
        precond_diag = 1.0 / (self.xi0_eff + noise)
        precond = LinearOperator((n, n), matvec=lambda v: precond_diag * v, dtype=float)
        iterations = [0]

        def callback(xk):
            iterations[0] += 1

        alpha, info = cg(
            operator,
            y,
            rtol=self.cg_rtol,
            atol=0.0,
            maxiter=self.cg_maxiter,
            M=precond,
            callback=callback,
        )
        if info > 0:
            resid = np.linalg.norm(matvec(alpha) - y) / np.linalg.norm(y)
            warnings.warn(
                "Conjugate gradient did not converge to rtol=%.1e in %i "
                "iterations (relative residual: %.1e). The solution is "
                "used anyway; increasing white_noise or cg_maxiter can "
                "help." % (self.cg_rtol, self.cg_maxiter, resid)
            )
        self.n_iterations = iterations[0]
        self._alpha = alpha

    def predict(self, X):
        """Evaluate the posterior mean ŷ(x) = sum_j alpha_j k(x - x_j)
        at the given coordinates by bilinear sampling of the mean field,
        which is computed once (a single convolution of the spread alpha
        with the kernel image) and cached until the next solve. Points
        beyond the kernel support of every training point return 0.

        :param X: The coordinates at which to interpolate. (n_samples, 2)
        """
        if self._alpha is None:
            raise RuntimeError("solve() must be called before predict().")
        if self._mean_field is None:
            field = (self._w_train.T @ self._alpha).reshape(self._ny, self._nx)
            self._mean_field = self._convolve(field)
        X = np.asarray(X, dtype=float)
        gx = (X[:, 0] - self._x0) / self.h
        gy = (X[:, 1] - self._y0) / self.h
        inside = (gx >= 0.0) & (gx <= self._nx - 1) & (gy >= 0.0) & (gy <= self._ny - 1)
        ix = np.clip(np.floor(gx).astype(np.intp), 0, self._nx - 2)
        iy = np.clip(np.floor(gy).astype(np.intp), 0, self._ny - 2)
        tx = gx - ix
        ty = gy - iy
        f = self._mean_field
        out = (
            (1.0 - tx) * (1.0 - ty) * f[iy, ix]
            + tx * (1.0 - ty) * f[iy, ix + 1]
            + (1.0 - tx) * ty * f[iy + 1, ix]
            + tx * ty * f[iy + 1, ix + 1]
        )
        out[~inside] = 0.0
        return out
