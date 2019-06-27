"""
.. module:: gp_interp_2pcf
"""

import numpy as np
import treecorr
import copy

from sklearn.gaussian_process.kernels import Kernel
from sklearn.neighbors import KNeighborsRegressor
from scipy import optimize
from scipy.linalg import cholesky, cho_solve
from scipy.stats import binned_statistic_2d


class bootstrap_2pcf(object):

    def __init__(self, X, y, y_err,
                 min_sep, max_sep, nbins=20, anisotropic=False):
        """Fit statistical uncertaintie on two-point correlation function using bootstraping.

        :param X:           The independent covariates.  (n_samples, 2)
        :param y:           The dependent responses.  (n_samples, n_targets)
        :param y_err:       Error of y. (n_samples, n_targets)
        :param min_sep:     Minimum bin for treecorr. (float)
        :param max_sep:     Maximum bin for treecorr. (float)
        :param nbins:       Number of bins (if 1D correlation function) of the square root of the number
                            of bins (if 2D correlation function) used in TreeCorr to compute the
                            2-point correlation function. [default: 20]
        :param anisotropic:  2D 2-point correlation function.
                            Used 2D correlation function for the
                            fiting part of the GP. (Boolean)
        """
        self.X = X
        self.y = y
        self.y_err = y_err
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.nbins = nbins
        self.anisotropic = anisotropic

    def resample_bootstrap(self):
        """
        Make a single bootstarp resampling on stars.
        """
        npsfs = len(self.y)

        ind_star = np.random.randint(0,npsfs-1, size=npsfs)
        u_ressample = self.X[:,0][ind_star]
        v_ressample = self.X[:,1][ind_star]
        y_ressample = self.y[ind_star]
        y_err_ressample = self.y_err[ind_star]

        return u_ressample, v_ressample, y_ressample, y_err_ressample

    def comp_2pcf(self, X, y, y_err):
        """
        Estimate 2-point correlation function using TreeCorr.

        :param X:  The independent covariates.  (n_samples, 2)
        :param y:  The dependent responses.  (n_samples, n_targets)
        :param y_err: Error of y. (n_samples, n_targets)
        """
        if np.sum(y_err) == 0:
            w = None
        else:
            w = 1./y_err**2

        if self.anisotropic:
            cat = treecorr.Catalog(x=X[:,0], y=X[:,1], k=(y-np.mean(y)), w=w)
            kk = treecorr.KKCorrelation(min_sep=self.min_sep, max_sep=self.max_sep, nbins=self.nbins,
                                        bin_type='TwoD', bin_slop=0)
            kk.process(cat)
            # Need a mask in the case of the 2D correlation function, to compute
            # the covariance matrix using the bootstrap. The 2D correlation
            # function is symmetric, so just half of the correlation function
            # is useful to compute the covariance matrix. If it is not done,
            # the covariance matrix is consequently not positive definite.
            npixels = len(kk.xi)**2
            mask = np.ones_like(kk.xi, dtype=bool)
            mask = mask.reshape((int(np.sqrt(npixels)), int(np.sqrt(npixels))))

            boolean_mask_odd = (len(mask)%2 == 0)
            even_or_odd = len(mask)%2
            nmask = int((len(mask)/2) + even_or_odd)
            mask[nmask:,:] = False
            mask[nmask-1][nmask:] = boolean_mask_odd

            mask = mask.reshape(npixels)

            distance = np.array([kk.dx.reshape(npixels), kk.dy.reshape(npixels)]).T
            Coord = distance
            xi = kk.xi.reshape(npixels)
        else:
            cat = treecorr.Catalog(x=X[:,0], y=X[:,1], k=(y-np.mean(y)), w=w)
            kk = treecorr.KKCorrelation(min_sep=self.min_sep, max_sep=self.max_sep, nbins=self.nbins)
            kk.process(cat)
            distance = kk.meanr
            mask = np.ones_like(kk.xi, dtype=bool)
            Coord = np.array([distance,np.zeros_like(distance)]).T
            xi = kk.xi

        return xi, distance, Coord, mask

    def comp_xi_covariance(self, n_bootstrap=1000, mask=None, seed=610639139):
        """
        Estimate 2-point correlation function covariance matrix using Bootstrap.

        :param seed: seed of the random generator.
        """
        np.random.seed(seed)
        xi_bootstrap = []
        for i in range(n_bootstrap):
            u, v, y, y_err = self.resample_bootstrap()
            coord = np.array([u, v]).T
            xi, d, c, m = self.comp_2pcf(coord, y, y_err)
            if mask is None:
                mask = np.array([True]*len(xi))
            xi_bootstrap.append(xi[mask])
        xi_bootstrap = np.array(xi_bootstrap)

        dxi = xi_bootstrap - np.mean(xi_bootstrap, axis=0)
        xi_cov = 1./(len(dxi)-1.) * np.dot(dxi.T, dxi)
        return xi_cov

    def return_2pcf(self, seed=610639139):
        """
        Return 2-point correlation function and its variance using Bootstrap.

        :param seed: seed of the random generator.
        """
        xi, distance, coord, mask = self.comp_2pcf(self.X, self.y, self.y_err)
        if self.anisotropic:
            # Choice done from Andy Taylor et al. 2012
            def f_bias(x, npixel=len(xi[mask])):
                top = x - 1.
                bottom = x - npixel - 2.
                return (top/bottom) - 2.
            results = optimize.fsolve(f_bias, len(xi[mask]) + 10)
            xi_cov = self.comp_xi_covariance(n_bootstrap=int(results[0]), mask=mask, seed=seed)
            bias_factor = (int(results[0]) - 1.) / (int(results[0]) - len(xi[mask]) - 2.)
            xi_weight = np.linalg.inv(xi_cov) * bias_factor
        else:
            # let like developed initialy for the moment
            xi_weight = np.eye(len(xi)) * 1./np.var(self.y)
        return xi, xi_weight, distance, coord, mask

class GPInterp2pcf(object):
    """
    An interpolator that uses two-point correlation function and gaussian process to interpolate a single surface.

    :param kernel:       A string that can be `eval`ed to make a
                         sklearn.gaussian_process.kernels.Kernel object.  The reprs of
                         sklearn.gaussian_process.kernels will work, as well as the repr of a
                         custom piff VonKarman object.  [default: 'RBF(1)']
    :param optimize:     Boolean indicating whether or not to try and optimize the kernel by
                         computing the two-point correlation function.  [default: True]
    :param anisotropic:   2D 2-point correlation function. Used 2D correlation function for the
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
                         function. In the same units as the keys. Compute automaticaly if it
                         is not given. [default: None]
    :param max_sep:      Maximum separation between pairs when computing 2-point correlation
                         function. In the same units as the keys. Compute automaticaly if it
                         is not given. [default: None]
    :param average_fits: A fits file that have the spatial average functions of PSF parameters
                         build in it. Build using meanify and piff output across different
                         exposures. See meanify documentation. [default: None]
    :param logger:       A logger object for logging debug info. [default: None]
    """
    def __init__(self, kernel='RBF(1)', optimize=True, anisotropic=False, normalize=True,
                 white_noise=0., n_neighbors=4, average_fits=None, nbins=20, min_sep=None, 
                 max_sep=None, logger=None):

        self.normalize = normalize
        self.optimize = optimize
        self.white_noise = white_noise
        self.n_neighbors = n_neighbors
        self.anisotropic = anisotropic
        self.nbins = nbins
        self.min_sep = min_sep
        self.max_sep = max_sep

        self.kwargs = {
            'keys': keys,
            'optimize': optimize,
            'kernel': kernel,
            'normalize':normalize
        }

        if len(keys)!=2:
            raise ValueError('the total size of keys can not be something else than 2 using two-point correlation function. Here len(keys) = %i'%(len(keys)))

        if type(kernel) is str:
            self.kernel_template = [self._eval_kernel(kernel)]
        else:
            if type(kernel) is not list and type(kernel) is not np.ndarray:
                raise TypeError("kernel should be a string a list or a numpy.ndarray of string")
            else:
                self.kernel_template = [self._eval_kernel(ker) for ker in kernel]

        self._2pcf = []
        self._2pcf_weight = []
        self._2pcf_dist = []
        self._2pcf_fit = []
        self._2pcf_mask = []

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

    @staticmethod
    def _eval_kernel(kernel):
        # Some import trickery to get all subclasses of sklearn.gaussian_process.kernels.Kernel
        # into the local namespace without doing "from sklearn.gaussian_process.kernels import *"
        # and without importing them all manually.
        def recurse_subclasses(cls):
            out = []
            for c in cls.__subclasses__():
                out.append(c)
                out.extend(recurse_subclasses(c))
            return out
        clses = recurse_subclasses(Kernel)
        for cls in clses:
            module = __import__(cls.__module__, globals(), locals(), cls)
            execstr = "{0} = module.{0}".format(cls.__name__)
            exec(execstr, globals(), locals())

        from numpy import array

        try:
            k = eval(kernel)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Failed to evaluate kernel string {0!r}.  "
                               "Original exception: {1}".format(kernel, e))

        if type(k.theta) is property:
            raise TypeError("String provided was not initialized properly")
        return k

    def _fit(self, kernel, X, y, y_err, logger=None):
        """Update the Kernel with data.

        :param kernel: sklearn.gaussian_process kernel.
        :param X:  The independent covariates.  (n_samples, 2)
        :param y:  The dependent responses.  (n_samples, n_targets)
        :param y_err: Error of y. (n_samples, n_targets)
        :param logger:  A logger object for logging debug info. [default: None]
        """
        if logger:
            logger.debug('Start kernel: %s', kernel.set_params())
            logger.debug('gp.fit with mean y = %s',np.mean(y))
        # Save these for potential read/write.
        if self.optimize:
            kernel = self._optimizer_2pcf(kernel,X,y,y_err)
            if logger:
                logger.debug('After fit: kernel = %s',kernel.set_params())
        return kernel

    def _optimizer_2pcf(self, kernel, X, y, y_err):
        """Fit hyperparameter using two-point correlation function.

        :param kernel: sklearn.gaussian_process kernel.
        :param X:  The independent covariates.  (n_samples, 2)
        :param y:  The dependent responses.  (n_samples, n_targets)
        :param y_err: Error of y. (n_samples, n_targets)
        """
        size_x = np.max(X[:,0]) - np.min(X[:,0])
        size_y = np.max(X[:,1]) - np.min(X[:,1])
        rho = float(len(X[:,0])) / (size_x * size_y)
        # if min_sep is None and isotropic GP, set min_sep to the average separation
        # between stars.
        if self.min_sep is not None:
            min_sep = self.min_sep
        else:
            if self.anisotropic:
                min_sep = 0.
            else:
                min_sep = np.sqrt(1./rho)
        # if max_sep is None, set max_sep to half of the size of the 
        # given field.
        if self.max_sep is not None:
            max_sep = self.max_sep
        else:
            max_sep = np.sqrt(size_x**2 + size_y**2)/2.

        bp = bootstrap_2pcf(X, y, y_err,
                            min_sep=min_sep, max_sep=max_sep, nbins=self.nbins,
                            anisotropic=self.anisotropic)
        xi, xi_weight, distance, coord, mask = bp.return_2pcf()

        def PCF(param, k=kernel):
            kernel =  k.clone_with_theta(param)
            pcf = kernel.__call__(coord,Y=np.zeros_like(coord))[:,0]
            return pcf

        xi_mask = xi[mask]
        def chi2(param):
            residual = xi_mask - PCF(param)[mask]
            return residual.dot(xi_weight.dot(residual))

        p0 = kernel.theta

        results_fmin = optimize.fmin(chi2,p0,disp=False)
        results_bfgs = optimize.minimize(chi2,p0,method="L-BFGS-B")
        results = [results_fmin, results_bfgs['x']]
        chi2_min = [chi2(results[0]), chi2(results[1])]
        ind_min = chi2_min.index(min(chi2_min))
        results = results[ind_min]

        self._2pcf.append(xi)
        self._2pcf_weight.append(xi_weight)
        self._2pcf_dist.append(distance)
        kernel = kernel.clone_with_theta(results)
        self._2pcf_fit.append(PCF(kernel.theta))
        self._2pcf_mask.append(mask)
        return kernel

    def return_gp_predict(self, y, X1, X2, kernel, y_err):
        """Compute interpolation with gaussian process for a given kernel.

        :param y:  The dependent responses.  (n_samples, n_targets)
        :param X1:  The independent covariates.  (n_samples, 2)
        :param X2:  The independent covariates at which to interpolate.  (n_samples, 2)
        :param kernel: sklearn.gaussian_process kernel.
        :param y_err: Error of y. (n_samples, n_targets)
        """
        HT = kernel.__call__(X2,Y=X1)
        K = kernel.__call__(X1) + np.eye(len(y))*y_err**2
        factor = (cholesky(K, overwrite_a=True, lower=False), False)
        alpha = cho_solve(factor, y, overwrite_b=False)
        return np.dot(HT,alpha.reshape((len(alpha),1))).T[0]


    def initialize(self, logger=None):
        """Initialize both the interpolator to some state prefatory to any solve iterations and
        initialize the stars for use with this interpolator.

        :param logger:  A logger object for logging debug info. [default: None]
        """
        self.nparams = len(stars[0].fit.params)
        if len(self.kernel_template)==1:
            self.kernels = [copy.deepcopy(self.kernel_template[0]) for i in range(self.nparams)]
        else:
            if len(self.kernel_template) != self.nparams:
                raise ValueError("numbers of kernel provided should be 1 (same for all parameters) or " \
                                 "equal to the number of params (%i), number kernel provided: %i" \
                                 %((self.nparams,len(self.kernel_template))))
            else:
                self.kernels = [copy.deepcopy(ker) for ker in self.kernel_template]

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
            return np.zeros((len(X[:,0]), self.nparams))

    def solve(self, X, y, y_err, logger=None):
        """Set up this GPInterp object.

        :param y:  The dependent responses.  (n_samples, n_targets)
        :param X1:  The independent covariates.  (n_samples, 2)
        :param X2:  The independent covariates at which to interpolate.  (n_samples, 2)
        :param kernel: sklearn.gaussian_process kernel.
        :param y_err: Error of y. (n_samples, n_targets)
        :param logger:   A logger object for logging debug info. [default: None]
        """
        self._X = X
        self._y = y

        if self._X0 is None:
            self._X0 = np.zeros_like(self._X)
            self._y0 = np.zeros_like(self._y)
        self._spatial_average = self._build_average_meanify(X)
        if np.shape(self._spatial_average)[1] > np.shape(self._y)[1]:
            logger.warning('Found {0} dimensions in the spatial average, but only {1} in the fit params. Cutting to {1}'.format(np.shape(self._spatial_average)[1], np.shape(self._y)[1]))
            ncut = np.shape(self._y)[1]
            self._spatial_average = self._spatial_average[:, :ncut]

        if self.white_noise > 0:
            y_err = np.sqrt(y_err**2 + self.white_noise**2)
        self._y_err = y_err

        if self.normalize:
            self._mean = np.mean(y - self._spatial_average, axis=0)
        else:
            self._mean = np.zeros(self.nparams)

        self._init_theta = []
        for i in range(self.nparams):
            kernel = self.kernels[i]
            self._init_theta.append(kernel.theta)
            self.kernels[i] = self._fit(self.kernels[i],
                                        X, y[:,i]-self._mean[i]-self._spatial_average[:,i],
                                        y_err[:,i], logger=logger)
            if logger:
                logger.info('param %d: %s',i,kernel.set_params())
                logger.info('param %d: %s',i,self.kernels[i])

    def predict(self, Xstar):
        """ Predict responses given covariates.
        :param Xstar:  The independent covariates at which to interpolate.  (n_samples, 2).
        :returns:  Regressed parameters  (n_samples, n_targets)
        """
        y_init = self._y
        y_err = self._y_err

        ystar = np.array([self.return_gp_predict(y_init[:,i]-self._mean[i]-self._spatial_average[:,i],
                                                 self._X, Xstar, ker, y_err=y_err[:,i])
                          for i, ker in enumerate(self.kernels)]).T

        spatial_average = self._build_average_meanify(Xstar)

        for i in range(self.nparams):
            ystar[:,i] += self._mean[i] + spatial_average[:,i]

        return ystar
