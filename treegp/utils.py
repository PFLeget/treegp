import numpy as np

# TO DO: E/B mode computation using numpy should use
# treecorr instead.


def vcorr(x, y, dx, dy, rmin=5.0 / 3600.0, rmax=1.5, dlogr=0.05, maxpts=30000):
    """
    Produce angle-averaged 2-point correlation functions of astrometric error
    for the supplied sample of data, using brute-force pair counting.
    Output are the following functions:
    logr - mean log of radius in each bin
    xi_+ - <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2>
    xi_- - <vr1 vr2 - vt1 vt2>
    xi_x - <vr1 vt2 + vt1 vr2>
    xi_z2 - <vx1 vx2 - vy1 vy2 + 2 i vx1 vy2>

    Parameters
    ----------
    x, y : array_like. positions of objects.
    dx, dy : array_like. astrometric shift.
    rmin : float. minimum separation in degrees. (default: 5.0 / 3600.0)
    rmax : float. maximum separation in degrees. (default: 1.5)
    dlogr : float. bin size in log(r). (default: 0.05)
    maxpts : int. maximum number of points to use. (default: 30000)

    Returns
    -------
    logr, xiplus, ximinus, xicross, xiz2 : array_like.
    """
    if len(x) > maxpts:
        # Subsample array to get desired number of points
        rate = float(maxpts) / len(x)
        print("Subsampling rate {:5.3f}%".format(rate * 100.0))
        use = np.random.random(len(x)) <= rate
        x = x[use]
        y = y[use]
        dx = dx[use]
        dy = dy[use]
    print("Length ", len(x))
    # Get index arrays that make all unique pairs
    i1, i2 = np.triu_indices(len(x))
    # Omit self-pairs
    use = i1 != i2
    i1 = i1[use]
    i2 = i2[use]
    del use

    # Make complex separation vector
    dr = 1j * (y[i2] - y[i1])
    dr += x[i2] - x[i1]

    # log radius vector used to bin data
    logdr = np.log(np.absolute(dr))
    logrmin = np.log(rmin)
    bins = int(np.ceil(np.log(rmax / rmin) / dlogr))
    hrange = (logrmin, logrmin + bins * dlogr)
    counts = np.histogram(logdr, bins=bins, range=hrange)[0]
    logr = np.histogram(logdr, bins=bins, range=hrange, weights=logdr)[0] / counts

    # First accumulate un-rotated stats
    v = dx + 1j * dy
    vv = dx[i1] * dx[i2] + dy[i1] * dy[i2]
    xiplus = np.histogram(logdr, bins=bins, range=hrange, weights=vv)[0] / counts
    vv = v[i1] * v[i2]
    xiz2 = np.histogram(logdr, bins=bins, range=hrange, weights=vv)[0] / counts

    # Now rotate into radial / perp components
    vv *= np.conj(dr)
    vv *= np.conj(dr)
    dr = dr.real * dr.real + dr.imag * dr.imag
    vv /= dr
    del dr
    ximinus = np.histogram(logdr, bins=bins, range=hrange, weights=vv)[0] / counts
    xicross = np.imag(ximinus)
    ximinus = np.real(ximinus)

    return logr, xiplus, ximinus, xicross, xiz2


def xiB(logr, xiplus, ximinus):
    """
    Return estimate of pure B-mode correlation function
    """
    # Integral of d(log r) ximinus(r) from r to infty:
    dlogr = np.zeros_like(logr)
    dlogr[1:-1] = 0.5 * (logr[2:] - logr[:-2])
    tmp = np.array(ximinus) * dlogr
    integral = np.cumsum(tmp[::-1])[::-1]
    return 0.5 * (xiplus - ximinus) + integral


def comp_eb(u, v, du, dv, **kwargs):
    """
    Compute E/B decomposition of astrometric error correlation function

    Parameters
    ----------
    u, v : array_like. positions of objects.
    du, dv : array_like. astrometric shift.

    returns
    -------
    xie, xib, logr : array_like. E-mode, B-mode,
    and log of binned distance separation in 2-point correlation function.
    """

    logr, xiplus, ximinus, xicross, xiz2 = vcorr(u, v, du, dv, **kwargs)
    xib = xiB(logr, xiplus, ximinus)
    xie = xiplus - xib
    return xie, xib, logr
