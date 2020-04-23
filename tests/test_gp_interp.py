from __future__ import print_function
import numpy as np
import treegp

from treegp_test_helper import timer
from treegp_test_helper import get_correlation_length_matrix
from treegp_test_helper import make_1d_grf
from treegp_test_helper import make_2d_grf

@timer
def test_gp_interp_1d():
    npoints = 40
    noise = [None, 0.1]
    # When there is no noise, a "magic"
    # factor is needed in order to be abble
    # to get a numericaly definite positive
    # matrix and to get a gp interpolation (determinant
    # of the kernel matrix is close to 0.). This
    # problem is solved by adding a little bit of
    # white noise when there is no noise.
    white_noise = [1e-5, 0.]
    sigma = [1., 2.]
    l = [2., 2.]
    atols_on_data = [0., 1e-3]
    kernels = ['RBF', 'VonKarman']

    for ker in kernels:
        for i in range(2):
            # Generate 1D gaussian random fields.
            kernel = "%f**2 * %s(%f)"%((sigma[i], ker, l[i]))
            kernel_skl = treegp.eval_kernel(kernel)
            x, y, y_err = make_1d_grf(kernel_skl,
                                      noise=noise[i],
                                      seed=42, npoints=npoints)

            # Do gp interpolation without hyperparameters
            # fitting (truth is put initially).
            gp = treegp.GPInterpolation(kernel=kernel, optimizer="none",
                                        white_noise=white_noise[i])
            gp.initialize(x, y, y_err=y_err)

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
            if noise[i] is not None:
                pull /= np.sqrt(y_err**2 + y_std**2)
            else:
                # Test that prediction is equal to the data at data
                # postion. Also test that diagonal of predict
                # covariance is zeros at data positions when no noise.
                np.testing.assert_allclose(y, y_predict, atol=3.*white_noise[i])
                np.testing.assert_allclose(np.zeros_like(y_std), y_std, atol=3.*white_noise[i])

            mean_pull = np.mean(pull)
            std_pull = np.std(pull)

            # Test that mean of the pull is close to zeros and std of the pull bellow 1.
            np.testing.assert_allclose(0., mean_pull, atol=3.*(std_pull)/np.sqrt(npoints))
            if std_pull > 1.:
                raise ValueError("std_pull is > 1. Current value std_pull = %f"%(std_pull))

            # Test that for extrapolation, interpolation is the mean function (0 here)
            # and the diagonal of the covariance matrix is close to the hyperameters is
            # link to the amplitudes of the fluctuation of the gaussian random fields.

            new_x = np.linspace(np.max(x)+6.*l[i], np.max(x)+7.*l[i], npoints).reshape((npoints,1))

            gp = treegp.GPInterpolation(kernel=kernel, optimizer="none", normalize=False,
                                        white_noise=white_noise[i])
            gp.initialize(x, y, y_err=y_err)
            y_predict, y_cov = gp.predict(new_x, return_cov=True)
            y_std = np.sqrt(np.diag(y_cov))

            np.testing.assert_allclose(np.zeros_like(y_predict), y_predict, atol=1e-5)
            np.testing.assert_allclose(sigma[i]*np.ones_like(y_std), y_std, atol=1e-5)

@timer
def test_gp_interp_2d():
    npoints = 200
    noise = [None, 0.1]
    # When there is no noise, a "magic"
    # factor is needed in order to be abble
    # to get a numericaly definite positive
    # matrix and to get a gp interpolation (determinant
    # of the kernel matrix is close to 0.). This
    # problem is solved by adding a little bit of
    # white noise when there is no noise.
    white_noise = [1e-5, 0.]
    sigma = [1., 1.]
    size = [2., 4.]
    g1 = [0., 0.2]
    g2 = [0., 0.2]
    atols_on_data = [0., 1e-3]
    kernels = ['AnisotropicRBF', 'AnisotropicVonKarman']

    for ker in kernels:
        for i in range(2):
            # Generate 2D gaussian random fields.
            L = get_correlation_length_matrix(size[i], g1[i], g2[i])
            invL = np.linalg.inv(L)
            kernel = "%f**2*%s"%((sigma[i], ker))
            kernel += "(invLam={0!r})".format(invL)
            kernel_skl = treegp.eval_kernel(kernel)

            x, y, y_err = make_2d_grf(kernel_skl,
                                      noise=noise[i],
                                      seed=42, npoints=npoints)

            # Do gp interpolation without hyperparameters
            # fitting (truth is put initially).
            gp = treegp.GPInterpolation(kernel=kernel, optimizer="none",
                                        white_noise=white_noise[i])
            gp.initialize(x, y, y_err=y_err)

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
            if noise[i] is not None:
                pull /= np.sqrt(y_err**2 + y_std**2)
            else:
                # Test that prediction is equal to the data at data
                # postion. Also test that diagonal of predict
                # covariance is zeros at data positions when no noise.
                np.testing.assert_allclose(y, y_predict, atol=3.*white_noise[i])
                np.testing.assert_allclose(np.zeros_like(y_std), y_std, atol=3.*white_noise[i])

            mean_pull = np.mean(pull)
            std_pull = np.std(pull)

            # Test that mean of the pull is close to zeros and std of the pull bellow 1.
            np.testing.assert_allclose(0., mean_pull, atol=3.*(std_pull)/np.sqrt(npoints))
            if std_pull > 1.:
                raise ValueError("std_pull is > 1. Current value std_pull = %f"%(std_pull))

            # Test that for extrapolation, interpolation is the mean function (0 here)
            # and the diagonal of the covariance matrix is close to the hyperameters is
            # link to the amplitudes of the fluctuation of the gaussian random fields.

            np.random.seed(42)
            x1 = np.random.uniform(np.max(x)+6.*size[i],
                                   np.max(x)+6.*size[i], npoints)
            x2 = np.random.uniform(np.max(x)+6.*size[i],
                                   np.max(x)+6.*size[i], npoints)
            new_x = np.array([x1, x2]).T

            gp = treegp.GPInterpolation(kernel=kernel, optimizer="none", normalize=False,
                                        white_noise=white_noise[i])
            gp.initialize(x, y, y_err=y_err)
            y_predict, y_cov = gp.predict(new_x, return_cov=True)
            y_std = np.sqrt(np.diag(y_cov))

            np.testing.assert_allclose(np.zeros_like(y_predict), y_predict, atol=1e-5)
            np.testing.assert_allclose(sigma[i]*np.ones_like(y_std), y_std, atol=1e-5)


if __name__ == "__main__":

    test_gp_interp_1d()
    test_gp_interp_2d()

