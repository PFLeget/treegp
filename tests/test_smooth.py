from __future__ import print_function
import warnings
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter

import treegp
from treegp.smooth import savgol2d

from treegp_test_helper import timer


def _quadratic(ny, nx, rng):
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    x = (x - nx / 2.0) / nx
    y = (y - ny / 2.0) / ny
    c = rng.normal(size=6)
    return c[0] + c[1] * x + c[2] * y + c[3] * x**2 + c[4] * x * y + c[5] * y**2


@timer
def test_savgol2d_exact_on_quadratic():
    """Order-2 filter reproduces any quadratic surface, borders included."""
    rng = np.random.default_rng(42)
    truth = _quadratic(40, 55, rng)

    for window in [5, 7]:
        out = savgol2d(truth, window=window, order=2)
        np.testing.assert_allclose(out, truth, atol=1e-10)

    # A 3x3 window has 9 cells for 6 terms: exact in the interior, but the
    # borders have too few neighbours (6 or 4) and are nan by default.
    out = savgol2d(truth, window=3, order=2)
    np.testing.assert_allclose(out[1:-1, 1:-1], truth[1:-1, 1:-1], atol=1e-10)
    assert np.all(np.isnan(out[0, :])) and np.all(np.isnan(out[:, 0]))

    # Same thing with holes.
    z = truth.copy()
    holes = rng.random(z.shape) < 0.15
    z[holes] = np.nan

    out = savgol2d(z, window=5, order=2)
    assert np.all(np.isnan(out[holes])), "footprint must be preserved"
    good = ~holes & np.isfinite(out)
    assert good.sum() > 0.95 * (~holes).sum()
    np.testing.assert_allclose(out[good], truth[good], atol=1e-10)

    out = savgol2d(z, window=5, order=2, fill_empty=True)
    filled = holes & np.isfinite(out)
    assert filled.sum() > 0.9 * holes.sum()
    np.testing.assert_allclose(out[filled], truth[filled], atol=1e-10)


@timer
def test_savgol2d_order0_is_box_average():
    """Order 0 with uniform weights is the (nan-aware) box average."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(30, 40))
    out = savgol2d(z, window=5, order=0, min_points=1)
    # Interior: plain uniform filter. Borders differ because the box average
    # is normalized by the number of available cells only.
    ref = uniform_filter(z, size=5, mode="constant")
    np.testing.assert_allclose(out[2:-2, 2:-2], ref[2:-2, 2:-2], atol=1e-12)
    # Corner: average of the 3x3 available block.
    np.testing.assert_allclose(out[0, 0], z[:3, :3].mean(), atol=1e-12)


@timer
def test_savgol2d_matches_1d_savgol():
    """A field varying along x only gives the 1d Savitzky-Golay result."""
    x = np.linspace(0, 6.0, 80)
    f = np.sin(x) + 0.3 * x**2 + np.random.default_rng(2).normal(0, 0.1, size=80)
    z = np.tile(f, (25, 1))
    for window, order in [(5, 2), (7, 2), (7, 3), (9, 1)]:
        out = savgol2d(z, window=window, order=order)
        ref = savgol_filter(f, window, order, mode="interp")
        h = window // 2
        # Away from the x borders (scipy extrapolates from the last full
        # window there, we fit the truncated window). All rows must agree,
        # y-borders included.
        np.testing.assert_allclose(
            out[:, h:-h], np.tile(ref[h:-h], (25, 1)), atol=1e-10
        )


@timer
def test_savgol2d_noise_reduction():
    """White noise on a smooth field is reduced without biasing the field."""
    rng = np.random.default_rng(3)
    ny, nx = 120, 150
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    truth = np.sin(2 * np.pi * x / 40.0) * np.cos(2 * np.pi * y / 50.0)
    sigma = 0.5
    z = truth + rng.normal(0, sigma, size=truth.shape)

    out = savgol2d(z, window=5, order=2)
    inner = (slice(5, -5), slice(5, -5))
    rms_before = np.std((z - truth)[inner])
    rms_after = np.std((out - truth)[inner])
    assert rms_after < 0.55 * rms_before
    # Bias on the smooth field stays small compared to the noise.
    assert abs(np.mean((out - truth)[inner])) < 0.02 * sigma

    # Unweighted result is the same whether the noise-free field is passed
    # as counts weights all equal or as None.
    out_w = savgol2d(z, window=5, order=2, weights=np.full(z.shape, 3.0))
    np.testing.assert_allclose(out_w, out, atol=1e-10)


@timer
def test_savgol2d_weights_and_min_points():
    """Zero weight is equivalent to nan; min_points controls sparse windows."""
    rng = np.random.default_rng(4)
    z = rng.normal(size=(20, 20))
    mask = rng.random(z.shape) < 0.3

    z_nan = z.copy()
    z_nan[mask] = np.nan
    w = np.where(mask, 0.0, 1.0)
    out_nan = savgol2d(z_nan, window=5, order=2, fill_empty=True)
    out_w = savgol2d(z, window=5, order=2, weights=w, fill_empty=True)
    np.testing.assert_allclose(out_w, out_nan, atol=1e-10, equal_nan=True)

    # Count-like weights change the answer (more weight on high-count cells).
    w2 = rng.integers(1, 50, size=z.shape).astype(float)
    out_w2 = savgol2d(z, window=5, order=2, weights=w2)
    assert not np.allclose(out_w2, savgol2d(z, window=5, order=2))

    # An isolated valid cell has too few neighbours.
    z_iso = np.full((11, 11), np.nan)
    z_iso[5, 5] = 2.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = savgol2d(z_iso, window=5, order=2)
        assert np.all(np.isnan(out))
        out = savgol2d(z_iso, window=5, order=0, min_points=1)
        assert out[5, 5] == 2.0
        assert np.isnan(out[0, 0])
        # Nothing valid at all.
        assert np.all(np.isnan(savgol2d(np.full((5, 5), np.nan))))

    # Argument validation.
    for kwargs in [dict(window=4), dict(window=3, order=3), dict(order=-1)]:
        try:
            savgol2d(z, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("savgol2d should reject %s" % kwargs)
    try:
        savgol2d(z, weights=np.ones((3, 3)))
    except ValueError:
        pass
    else:
        raise AssertionError("savgol2d should reject mismatched weights")

    # Exported at package level.
    assert treegp.savgol2d is savgol2d


if __name__ == "__main__":
    test_savgol2d_exact_on_quadratic()
    test_savgol2d_order0_is_box_average()
    test_savgol2d_matches_1d_savgol()
    test_savgol2d_noise_reduction()
    test_savgol2d_weights_and_min_points()
