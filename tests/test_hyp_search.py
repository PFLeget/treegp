from __future__ import print_function
import numpy as np
import treegp

from treegp_test_helper import timer
from treegp_test_helper import get_correlation_length_matrix
from treegp_test_helper import make_1d_grf
from treegp_test_helper import make_2d_grf

@timer
def test_hyperparameter_search_1d():
    optimizer = ['log-likelihood', 'two-pcf']
    npoints = [100, 2000]
    noise = 0.01
    sigma = [1., 2., 1., 2.]
    l = [0.5, 0.8, 8., 10.]
    kernels = ['RBF', 'RBF', 'VonKarman', 'VonKarman']
    max_sep = [1.75, 1.75, 1.25, 1.25]
    
    for n, opt in enumerate(optimizer):
        for i, ker in enumerate(kernels):
            # Generate 1D gaussian random fields.
            kernel = "%f**2 * %s(%f)"%((sigma[i], ker, l[i]))
            kernel_skl = treegp.eval_kernel(kernel)
            x, y, y_err = make_1d_grf(kernel_skl,
                                      noise=noise,
                                      seed=42, npoints=npoints[n])

            # Do gp interpolation without hyperparameters
            # fitting (truth is put initially).
            gp = treegp.GPInterpolation(kernel=kernel, optimizer=opt, anisotropic=False, 
                                        normalize=True, nbins=15, min_sep=0.1, 
                                        max_sep=max_sep[i])
            gp.initialize(x, y, y_err=y_err)
            gp.solve()
            # test if found hyperparameters are close the true hyperparameters.
            np.testing.assert_allclose(kernel_skl.theta, gp.kernel.theta, atol=7e-1)
            
            if opt is "two-pcf":
                xi, xi_weight, distance, coord, mask = gp.return_2pcf()
                np.testing.assert_allclose(xi, gp._optimizer._2pcf, atol=1e-10)
            if opt is "log-likelihood":
                logL = gp.return_log_likelihood()
                np.testing.assert_allclose(logL, gp._optimizer._logL, atol=1e-10)

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
            mean_pull = np.mean(pull)
            std_pull = np.std(pull)

            # Test that mean of the pull is close to zeros and std of the pull bellow 1.
            np.testing.assert_allclose(0., mean_pull, atol=3.*(std_pull)/np.sqrt(npoints[n]))
            if std_pull > 1.:
                raise ValueError("std_pull is > 1. Current value std_pull = %f"%(std_pull))

            # Test that for extrapolation, interpolation is the mean function (0 here)
            # and the diagonal of the covariance matrix is close to the hyperameters is
            # link to the amplitudes of the fluctuation of the gaussian random fields.

            new_x = np.linspace(np.max(x)+6.*l[i], np.max(x)+7.*l[i], npoints[n]).reshape((npoints[n],1))
        
            y_predict, y_cov = gp.predict(new_x, return_cov=True)
            y_std = np.sqrt(np.diag(y_cov))
            
            np.testing.assert_allclose(np.mean(y)*np.ones_like(y_std), y_predict, atol=1e-5)
            sig = np.sqrt(np.exp(gp.kernel.theta[0]))
            np.testing.assert_allclose(sig*np.ones_like(y_std), y_std, atol=1e-5)

@timer
def test_hyperparameter_search_2d():
    optimizer = ['log-likelihood', 'two-pcf']
    npoints = [400, 2000]

    noise = 0.01
    sigma = 2.
    size = 0.5
    g1 = 0.2
    g2 = 0.2
    ker = 'AnisotropicRBF'

    # Generate 2D gaussian random fields.
    L = get_correlation_length_matrix(size, g1, g2)
    invL = np.linalg.inv(L)
    kernel = "%f**2*%s"%((sigma, ker))
    kernel += "(invLam={0!r})".format(invL)
    kernel_skl = treegp.eval_kernel(kernel)

    for n, opt in enumerate(optimizer):
        x, y, y_err = make_2d_grf(kernel_skl,
                                  noise=noise,
                                  seed=42, npoints=npoints[n])

        # Do gp interpolation without hyperparameters
        # fitting (truth is put initially).
        gp = treegp.GPInterpolation(kernel=kernel, optimizer=opt, anisotropic=True,
                                    normalize=True, nbins=21, min_sep=0.,
                                    max_sep=1., p0=[0.3, 0.,0.])
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
        np.testing.assert_allclose(0., mean_pull, atol=3.*(std_pull)/np.sqrt(npoints[n]))
        if std_pull > 1.:
            raise ValueError("std_pull is > 1. Current value std_pull = %f"%(std_pull))

        # Test that for extrapolation, interpolation is the mean function (0 here)
        # and the diagonal of the covariance matrix is close to the hyperameters is
        # link to the amplitudes of the fluctuation of the gaussian random fields.

        np.random.seed(42)
        x1 = np.random.uniform(np.max(x)+6.*size,
                               np.max(x)+6.*size, npoints[n])
        x2 = np.random.uniform(np.max(x)+6.*size,
                               np.max(x)+6.*size, npoints[n])
        new_x = np.array([x1, x2]).T

        y_predict, y_cov = gp.predict(new_x, return_cov=True)
        y_std = np.sqrt(np.diag(y_cov))

        np.testing.assert_allclose(np.mean(y), y_predict, atol=1e-5)
        sig = np.sqrt(np.exp(gp.kernel.theta[0]))
        np.testing.assert_allclose(sig*np.ones_like(y_std), y_std, atol=1e-5)

if __name__ == "__main__":

    test_hyperparameter_search_1d()
    test_hyperparameter_search_2d()
