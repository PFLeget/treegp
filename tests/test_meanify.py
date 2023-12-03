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


if __name__ == "__main__":
    test_meanify()
    test_gpinterp_meanify()
