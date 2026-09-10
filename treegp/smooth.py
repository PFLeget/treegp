"""
.. module:: smooth
"""

import numpy as np
from scipy.ndimage import correlate


def _poly_basis(order):
    """Exponents (px, py) of the 2d polynomial basis of total degree ``order``.

    The first element is always (0, 0), the constant term.
    """
    exps = []
    for total in range(order + 1):
        for px in range(total, -1, -1):
            exps.append((px, total - px))
    return exps


def savgol2d(z, window=5, order=2, weights=None, min_points=None, fill_empty=False):
    """2d Savitzky-Golay filter with support for missing data and weights.

    For each cell of the grid, a 2d polynomial of total degree ``order`` is
    fitted by weighted least squares to the cells of the ``window`` x ``window``
    neighbourhood, and the smoothed value is the polynomial evaluated at the
    cell centre. Cells that are nan (or have zero weight) are simply left out
    of the fit, and so are the cells that fall outside the grid, so borders
    and holes are handled by fitting the polynomial on the available part of
    the window. With ``weights=None`` and no missing data, this is the
    classical Savitzky-Golay filter (a linear FIR filter); with weights it is
    a local weighted polynomial regression.

    The fit is computed for all cells at once: each entry of the normal
    equations is a correlation of the weights (or of weights * data) with a
    small polynomial kernel, followed by a batched linear solve.

    :param z:          2d array to smooth. nan marks missing cells.
    :param window:     Odd integer, full width of the square window in cells.
                       (default: 5)
    :param order:      Total degree of the polynomial. (default: 2)
    :param weights:    Optional 2d array of non-negative weights, same shape
                       as ``z`` (e.g. number of points per cell, since the
                       variance of a bin average scales as 1/N). None means
                       uniform weights on the finite cells. (default: None)
    :param min_points: Minimum number of valid cells in the window needed to
                       compute a value; cells with fewer valid neighbours are
                       set to nan. (default: number of polynomial terms + 2)
    :param fill_empty: If False, cells that are nan in ``z`` stay nan in the
                       output, so the footprint is preserved. If True, they
                       are filled with the local polynomial estimate when
                       enough neighbours are available. (default: False)
    :returns: 2d array, same shape as ``z``.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("savgol2d only supports 2d arrays.")
    window = int(window)
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be an odd positive integer.")
    order = int(order)
    if order < 0:
        raise ValueError("order must be a non-negative integer.")
    exps = _poly_basis(order)
    nterms = len(exps)
    if window**2 < nterms:
        raise ValueError(
            "window=%i is too small for a polynomial of order %i (%i terms)."
            % (window, order, nterms)
        )
    if min_points is None:
        min_points = nterms + 2
    min_points = max(int(min_points), nterms)

    finite = np.isfinite(z)
    if weights is None:
        w = finite.astype(float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != z.shape:
            raise ValueError("weights must have the same shape as z.")
        w = np.where(finite & np.isfinite(w) & (w > 0), w, 0.0)
    valid = w > 0
    zw = np.where(valid, z, 0.0) * w

    out = np.full(z.shape, np.nan)
    if not np.any(valid):
        return out

    # Polynomial basis evaluated on the window offsets, scaled to [-1, 1]
    # for conditioning. Kernel index k corresponds to offset k - h, which is
    # the convention of scipy.ndimage.correlate.
    h = window // 2
    d = np.arange(-h, h + 1, dtype=float) / max(h, 1)
    dy, dx = np.meshgrid(d, d, indexing="ij")
    basis = [dx**px * dy**py for (px, py) in exps]

    A = np.empty(z.shape + (nterms, nterms))
    b = np.empty(z.shape + (nterms,))
    for i in range(nterms):
        b[..., i] = correlate(zw, basis[i], mode="constant", cval=0.0)
        for j in range(i, nterms):
            A[..., i, j] = correlate(w, basis[i] * basis[j], mode="constant", cval=0.0)
            if j != i:
                A[..., j, i] = A[..., i, j]

    npts = correlate(
        valid.astype(float), np.ones((window, window)), mode="constant", cval=0.0
    )
    ok = npts >= min_points
    if fill_empty:
        ok &= True
    else:
        ok &= finite

    if np.any(ok):
        A_ok = A[ok]
        b_ok = b[ok]
        try:
            sol = np.linalg.solve(A_ok, b_ok[..., None])[..., 0, 0]
        except np.linalg.LinAlgError:
            # Some windows are degenerate (e.g. all valid cells on a line);
            # fall back to the minimum-norm solution for all of them.
            sol = np.einsum("nij,nj->ni", np.linalg.pinv(A_ok), b_ok)[:, 0]
        sol[~np.isfinite(sol)] = np.nan
        out[ok] = sol

    return out
