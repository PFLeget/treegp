"""
.. module:: gp_interp
"""

import treegp
import numpy as np
import copy

from .kernels import eval_kernel

from sklearn.gaussian_process.kernels import Kernel
from sklearn.neighbors import KNeighborsRegressor
from scipy.linalg import cholesky, cho_solve


class GPInterpolation(object):
    """
    An interpolator that uses two-point correlation function and gaussian process to interpolate a single surface.

    :param kernel:       A string that can be `eval`ed to make a
                         sklearn.gaussian_process.kernels.Kernel object.  The reprs of
                         sklearn.gaussian_process.kernels will work, as well as the repr of a
                         custom piff VonKarman object.  [default: 'RBF(1)']
    :param optimize:     Boolean indicating whether or not to try and optimize the kernel by
                         computing the two-point correlation function.  [default: True]
    :param anisotropic:  2D 2-point correlation function. Used 2D correlation function for the
                         fiting part of the GP instead of a 1D correlation function. [default: False]
    :param normalize:    Whether to normalize the interpolation parameters to have a mean of 0.
                         Normally, the parameters being interpolated are not mean 0, so you would
                         want this to be True, but if your parameters have an a priori mean of 0,
                         then subtracting off the realized mean would be invalid.  [default: True]
    :param white_noise:  A float value that indicate the ammount of white noise that you want to
                         use during the gp interpolation. This is an additional uncorrelated noise
                         added to the error of the PSF parameters. [default: 0.]
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
    :param average_fits: A fits file that have the spatial average functions of PSF parameters
                         build in it. Build using meanify and piff output across different
                         exposures. See meanify documentation. [default: None]
    """
    def __init__(self, kernel='RBF(1)', optimize=True, optimizer='two-pcf', anisotropic=False, normalize=True,
                 white_noise=0., n_neighbors=4, average_fits=None, nbins=20, min_sep=None, max_sep=None):

        self.normalize = normalize
        self.optimize = optimize
        self.optimizer = optimizer
        self.white_noise = white_noise
        self.n_neighbors = n_neighbors
        self.anisotropic = anisotropic
        self.nbins = nbins
        self.min_sep = min_sep
        self.max_sep = max_sep

        self.kwargs = {
            'optimize': optimize,
            'kernel': kernel,
            'normalize':normalize
        }

        if isinstance(kernel,str):
            self.kernel_template = eval_kernel(kernel)
        else:
            raise TypeError("kernel should be a string a list or a numpy.ndarray of string")

        if self.optimizer not in ['two-pcf', 'log-likelihood']:
            raise ValueError("Only two-pcf and log-likelihood are supported for optimizer. Current value: %s"%(self.optimizer))

        if average_fits is not None:
            import fitsio
            average = fitsio.read(average_fits)
            X0 = average['COORDS0'][0]
            y0 = average['PARAMS0'][0]
        else:
            X0 = None
            y0 = None
        self._X0 = X0
        self._y0 = y0

    def _fit(self, kernel, X, y, y_err):
        """Update the Kernel with data.

        :param kernel: sklearn.gaussian_process kernel.
        :param X:  The independent covariates.  (n_samples, 2)
        :param y:  The dependent responses.  (n_samples, n_targets)
        :param y_err: Error of y. (n_samples, n_targets)
        """
        if self.optimize:
            # Hyperparameters estimation using 2-point correlation
            # function information.
            if self.optimizer == 'two-pcf':
                self._optimizer = treegp.two_pcf(X, y, y_err,
                                                 self.min_sep, self.max_sep,
                                                 nbins=self.nbins,
                                                 anisotropic=self.anisotropic)
                kernel = self._optimizer.optimizer(kernel)
            # Hyperparameters estimation using maximum likelihood fit.
            if self.optimizer == 'log-likelihood':
                self._optimizer = None
                kernel = None
                print('TO IMPLEMENT')

        return kernel

    def predict(self, Xstar, return_cov=False):
        """ Predict responses given covariates.
        :param Xstar:  The independent covariates at which to interpolate.  (n_samples, 2).
        :returns:  Regressed parameters  (n_samples, n_targets)
        """
        y_init = copy.deepcopy(self._y)
        y_err = copy.deepcopy(self._y_err)

        ystar, y_cov = self.return_gp_predict(y_init-self._mean-self._spatial_average,
                                              self._X, Xstar, self.kernel, y_err=y_err,
                                              return_cov=return_cov)
        ystar = ystar.T
        spatial_average = self._build_average_meanify(Xstar)
        ystar += self._mean + spatial_average
        if return_cov:
            return ystar, y_cov
        else:
            return ystar

    def return_gp_predict(self, y, X1, X2, kernel, y_err, return_cov=False):
        """Compute interpolation with gaussian process for a given kernel.

        :param y:  The dependent responses.  (n_samples, n_targets)
        :param X1:  The independent covariates.  (n_samples, 2)
        :param X2:  The independent covariates at which to interpolate.  (n_samples, 2)
        :param kernel: sklearn.gaussian_process kernel.
        :param y_err: Error of y. (n_samples, n_targets)
        """
        HT = kernel.__call__(X2, Y=X1)
        K = kernel.__call__(X1) + np.eye(len(y))*y_err**2
        factor = (cholesky(K, overwrite_a=True, lower=False), False)
        alpha = cho_solve(factor, y, overwrite_b=False)
        y_predict = np.dot(HT,alpha.reshape((len(alpha),1))).T[0] 
        if return_cov:
            fact = cholesky(K, lower=True) # I am computing maybe twice the same things...
            v = cho_solve((fact, True), HT.T)
            y_cov = kernel.__call__(X2) - HT.dot(v)
            return y_predict, y_cov
        else:
            return y_predict, None

    def initialize(self, X, y, y_err=None):
        """Initialize both the interpolator to some state prefatory to any solve iterations and
        initialize the stars for use with this interpolator.

        :param stars:   A list of Star instances to interpolate between
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
            y_err = np.sqrt(copy.deepcopy(self._y_err)**2 + self.white_noise**2)
        self._y_err = y_err

        if self.normalize:
            self._mean = np.mean(y - self._spatial_average)
        else:
            self._mean = 0.

    def _build_average_meanify(self, X):
        """Compute spatial average from meanify output for a given coordinate using KN interpolation.
        If no average_fits was given, return array of 0.

        :param X: Coordinates of training stars or coordinates where to interpolate. (n_samples, 2)
        """
        if np.sum(np.equal(self._X0, 0)) != len(self._X0[:,0])*len(self._X0[0]):
            neigh = KNeighborsRegressor(n_neighbors=self.n_neighbors)
            neigh.fit(self._X0, self._y0)
            average = neigh.predict(X)         
            return average
        else:
            return np.zeros((len(X[:,0])))

    def solve(self):
        """Set up this GPInterp object.

        :param stars:    A list of Star instances to interpolate between
        """
        self._init_theta = []
        kernel = copy.deepcopy(self.kernel)
        self._init_theta.append(kernel.theta)
        self.kernel = self._fit(self.kernel, self._X, 
                                self._y-self._mean-self._spatial_average, self._y_err)