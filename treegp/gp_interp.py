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
    An interpolator that uses 2-point correlation function informations
    or Maximum Likelihood informations to do a gaussian process to interpolate
    a single surface.

    :param kernel:       A string that can be `eval`ed to make a
                         sklearn.gaussian_process.kernels.Kernel object.  The reprs of
                         sklearn.gaussian_process.kernels will work, as well as the repr of a
                         custom treegp VonKarman object.  [default: 'RBF(1)']
    :param optimizer:    Indicates which techniques to use for optimizing the kernel. Three options
                         are available. "none" does not optimize hyperparameters and used the one
                         given in the kernel. "two-pcf" optimize the kernel on the 1d 2-point
                         correlation function estimate by treecorr. "anisotropic" optimize the kernel
                         on the 2d 2-point correlation function estimate by treecorr.
                         "log-likelihood" used the classical maximum likelihood method.
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
    """
    def __init__(self, kernel='RBF(1)', optimizer='two-pcf', 
                 normalize=True, p0=[3000., 0.,0.],
                 white_noise=0., n_neighbors=4, average_fits=None, indice_meanify=None,
                 nbins=20, min_sep=None, max_sep=None):

        self.normalize = normalize
        self.optimizer = optimizer
        self.white_noise = white_noise
        self.n_neighbors = n_neighbors
        self.nbins = nbins
        self.min_sep = min_sep
        self.max_sep = max_sep

        if self.optimizer == 'anisotropic':
            self.robust_fit = True
        else:
            self.robust_fit = False

        self.p0_robust_fit = p0
        self.indice_meanify = indice_meanify

        if isinstance(kernel,str):
            self.kernel_template = eval_kernel(kernel)
        else:
            raise TypeError("kernel should be a string a list or a numpy.ndarray of string")

        if self.optimizer not in ['anisotropic', 'two-pcf', 'log-likelihood', 'none']:
            raise ValueError("Only anisotropic, two-pcf, log-likelihood and none are supported for optimizer. Current value: %s"%(self.optimizer))

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
        :param X:  Coordinates of the field.  (n_samples, 1 or 2)
        :param y:  Values of the field.  (n_samples)
        :param y_err: Error of y. (n_samples)
        """
        if self.optimizer is not "none":
            # Hyperparameters estimation using 2-point correlation
            # function information.
            if self.optimizer in ['two-pcf', 'anisotropic']:
                anisotropic = self.optimizer == 'anisotropic'
                self._optimizer = treegp.two_pcf(X, y, y_err,
                                                 self.min_sep, self.max_sep,
                                                 nbins=self.nbins,
                                                 anisotropic=anisotropic,
                                                 robust_fit=self.robust_fit,
                                                 p0=self.p0_robust_fit)
                kernel = self._optimizer.optimizer(kernel)
            # Hyperparameters estimation using maximum likelihood fit.
            if self.optimizer == 'log-likelihood':
                self._optimizer = treegp.log_likelihood(X, y, y_err)
                kernel = self._optimizer.optimizer(kernel)
        return kernel

    def predict(self, X, return_cov=False):
        """ Predict responses to given coordinates.

        :param X:  The coordinates at which to interpolate.  (n_samples, 1 or 2).
        :returns:  Regressed parameters  (n_samples)
        """
        y_init = copy.deepcopy(self._y)
        y_err = copy.deepcopy(self._y_err)

        y_interp, y_cov = self.return_gp_predict(y_init-self._mean-self._spatial_average,
                                                 self._X, X, self.kernel, y_err=y_err,
                                                 return_cov=return_cov)
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
            y_err = np.sqrt(copy.deepcopy(self._y_err)**2 + self.white_noise**2)
        self._y_err = y_err

        if self.normalize:
            self._mean = np.mean(y - self._spatial_average)
        else:
            self._mean = 0.

    def _build_average_meanify(self, X):
        """Compute spatial average from meanify output for a given coordinate using KN interpolation.
        If no average_fits was given, return array of 0.

        :param X: Coordinates of training coordinates where to interpolate. (n_samples, 1 or 2)
        """
        if np.sum(np.equal(self._X0, 0)) != len(self._X0[:,0])*len(self._X0[0]):
            neigh = KNeighborsRegressor(n_neighbors=self.n_neighbors)
            neigh.fit(self._X0, self._y0)
            average = neigh.predict(X)         
            if self.indice_meanify is not None:
                average = average[:,self.indice_meanify]
            return average
        else:
            return np.zeros((len(X[:,0])))

    def solve(self):
        """Set up this GPInterp object.
        Solve for hyperparameters if requested using 2-point correlation
        function method or maximum likelihood.
        """
        self._init_theta = []
        kernel = copy.deepcopy(self.kernel)
        self._init_theta.append(kernel.theta)
        self.kernel = self._fit(self.kernel, self._X, 
                                self._y-self._mean-self._spatial_average, self._y_err)

    def return_2pcf(self):
        """
        Return 2-point correlation function and its variance using Bootstrap.
        """
        anisotropic = self.optimizer == "anisotropic"
        pcf = treegp.two_pcf(self._X, self._y-self._mean-self._spatial_average, self._y_err,
                             self.min_sep, self.max_sep,
                             nbins=self.nbins,
                             anisotropic=anisotropic)
        xi, xi_weight, distance, coord, mask = pcf.return_2pcf()
        return xi, xi_weight, distance, coord, mask

    def return_log_likelihood(self, theta=None):
        """
        Return of log likehood of gaussian process
        for given hyperparameters.

        :param theta: Array of hyperparamters. (default: None)
        """
        kernel = copy.deepcopy(self.kernel)
        if theta is not None:
            kernel = kernel.clone_with_theta(theta)
        logl = treegp.log_likelihood(self._X, self._y-self._mean-self._spatial_average, self._y_err)
        log_likelihood = logl.log_likelihood(kernel)
        return log_likelihood
