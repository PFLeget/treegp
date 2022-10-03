from __future__ import print_function
import numpy as np
import treecorr
import treegp
from .kernels import eval_kernel
from scipy import optimize
import iminuit
import sklearn
import copy
import warnings

def get_correlation_length_matrix(size, e1, e2):
    """
    Produce correlation matrix to introduce anisotropy in kernel.
    Used same parametrization as galaxy shape measurement
    in gravitational weak lensing because this is
    mathematicaly equivalent (anistropic kernel will have
    an elliptical shape).

    :param size:   Correlation lenght of the kernel.
    :param e1, e2: Shear applied to isotropic kernel.
    """
    if abs(e1)>1 or abs(e2)>1:
        raise ValueError('abs value of e1 and e2 must be lower than one')
    e = np.sqrt(e1**2 + e2**2)
    q = (1-e) / (1+e)
    phi = 0.5 * np.arctan2(e2,e1)
    rot = np.array([[np.cos(phi), np.sin(phi)],
                    [-np.sin(phi), np.cos(phi)]])
    ell = np.array([[size**2, 0],
                    [0, (size * q)**2]])
    L = np.dot(rot.T, ell.dot(rot))
    return L

def get_kernel_class(A):
    """
    Check that the given kernel is an AnisotropicVonKarman or an
    AnisotropicRBF kernel.

    :param A: sklearn.gaussian_process.kernels
    """
    if A.__class__ in [treegp.kernels.AnisotropicVonKarman,
                       treegp.kernels.AnisotropicRBF,
                       sklearn.gaussian_process.kernels.Product]:

        if A.__class__ in [sklearn.gaussian_process.kernels.Product]:
            ok = False
            for key in A.__dict__:
                if A.__dict__[key].__class__ in [treegp.kernels.AnisotropicVonKarman,
                                                 treegp.kernels.AnisotropicRBF]:
                    kernel_class = A.__dict__[key].__class__
                    ok = True
            if not ok:
                raise ValueError('Work only with treegp.kernels.AnisotropicVonKarman and treegp.kernels.AnisotropicRBF')
        else:
            kernel_class = A.__class__
        return kernel_class
    else:
        raise ValueError('Work only with treegp.kernels.AnisotropicVonKarman and treegp.kernels.AnisotropicRBF')

class robust_2dfit(object):
    """
    Fit hyperparameters on 2D two-point correlation when the analytical profil can
    be discribed as a Radial Basis Function such as a Gaussian kernel or a
    von Karman kernel.

    :param kernel:    sklearn.gaussian_process.kernels
    :param x:         x coordinates of the 2D two point correlation function.
    :param y:         y coordinates of the 2D two point correlation function.
    :param flat_data: flatten 2D two point correlation function.
    :param W:         Inverse of the covariance matrix got from Bootstrap.
    :param mask:      Mask symetric area for Radial basi Function.
    """
    def __init__(self, kernel, flat_data, x, y, W, mask=None):

        if mask is None:
            self.mask = np.array([True]*len(x))
        else:
            self.mask = mask

        self.kernel_class = get_kernel_class(kernel)
        self.flat_data = flat_data
        self.x = x
        self.y = y
        self.coord = np.array([x, y]).T
        self.W = W
        self.N = int(np.sqrt(len(self.x)))

    def _model_skl(self, sigma, corr_length, g1, g2):
        """
        Build analytical two point correlation function
        from sklearn kernel.

        :param sigma:         Standard deviation of the gaussian random field.
        :param corr_length:   Correlation lenght of the kernel.
        :param g1, g2:        Shear applied to isotropic kernel.
        """
        if abs(g1)>1 or abs(g2)>1:
            return None
        else:
            L = get_correlation_length_matrix(corr_length, g1, g2)
            invLam = np.linalg.inv(L)
            kernel_used = sigma**2 * self.kernel_class(invLam=invLam)
            pcf = kernel_used.__call__(self.coord,Y=np.zeros_like(self.coord))[:,0]
            self.kernel_fit = kernel_used
            return pcf

    def chi2(self, param):
        """
        Chi2 function that iminuit will minimize.

        Note: only non linear parameters are searched during
        the minimization. As linear parameters have a unique
        solution for a given set of non-linear parameters,
        am solving the equation to get the analytical solution
        for the linear parameter (alpha).

        :param param: list of non-linear parameters to search.
                      (correlation lenght, e1, and e2).
        """
        model = self._model_skl(1., param[0],
                                param[1], param[2])
        if model is None:
            self.chi2_value = np.inf
        else:
            model = model[self.mask]
            F = np.array([model, np.ones_like(model)]).T
            FWF = np.dot(F.T, self.W).dot(F)
            Y = self.flat_data[self.mask].reshape((len(model), 1))
            self.alpha = np.linalg.inv(FWF).dot(np.dot(F.T, self.W).dot(Y))
            self.alpha[0] = abs(self.alpha[0])
            self.residuals = self.flat_data[self.mask] - ((self.alpha[0] * model) + self.alpha[1])
            self.chi2_value = self.residuals.dot(self.W).dot(self.residuals.reshape((len(model), 1)))

        return self.chi2_value

    def _minimize_minuit(self, p0 = [3000., 0.2, 0.2]):
        """
        Launch a single minimization by minuit given a set of starting point.

        :param p0: List of starting points.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if int(iminuit.__version__[0])>=2:
                self.m = iminuit.Minuit(self.chi2, p0)
                self.m.migrad()
                results = [self.m.params[key].value for key in self.m.parameters]
                self._fit_ok = self.m.accurate
            else:
                self.m = iminuit.Minuit.from_array_func(self.chi2, p0, print_level=0)
                self.m.migrad()
                results = [self.m.values[key] for key in self.m.values.keys()]
                self._fit_ok = self.m.migrad_ok()

        self._minuit_result = results
        self.result = [np.sqrt(self.alpha[0][0]), results[0],
                       results[1], results[2],
                       self.alpha[1][0]]


    def minimize_minuit(self, p0 = [3000., 0.2, 0.2]):
        """
        Launch the minimization by minuit given a set of starting point.
        Launch again on different starting point if minuit failed to find
        a solution.

        :param p0: List of starting points.
        """
        self._minimize_minuit(p0=p0)

        if not self._fit_ok:
            N_restart = 3
            g1 = np.linspace(-0.3, 0.3, N_restart)
            size = np.linspace(p0[0] - p0[0]/10., 2*p0[0], N_restart)
            g1, g2, size = np.meshgrid(g1, g1, size)
            N_restart = int(N_restart**3)
            g1 = g1.reshape(N_restart)
            g2 = g2.reshape(N_restart)
            size = size.reshape(N_restart)
            for i in range(N_restart):
                print("restart fit because failure")
                new_p0 = [size[i], g1[i], g2[i]]
                print(new_p0)
                self._minimize_minuit(p0=new_p0)
                if self._fit_ok:
                    break
        pcf = self._model_skl(self.result[0], self.result[1], 
                              self.result[2], self.result[3])

class two_pcf(object):
    """
    Fit statistical uncertaintie on two-point correlation function using bootstraping.

    :param X:           Coordinates of the field.  (n_samples, 1 or 2)
    :param y:           Values of the field. (n_samples)
    :param y_err:       Error of y. (n_samples)
    :param min_sep:     Minimum bin for treecorr. (float)
    :param max_sep:     Maximum bin for treecorr. (float)
    :param nbins:       Number of bins (if 1D correlation function) of the square root of the number
                        of bins (if 2D correlation function) used in TreeCorr to compute the
                        2-point correlation function. [default: 20]
    :param anisotropic: 2D 2-point correlation function.
                        Used 2D correlation function for the
                        fiting part of the GP. (Boolean)
    :param robust_fit:  Used minuit to fit hyperparameter. Works only
                        anisotropic is True. (Boolean)
    """
    def __init__(self, X, y, y_err,
                 min_sep, max_sep, nbins=20, 
                 anisotropic=False, robust_fit=False, 
                 p0=[3000., 0., 0.]):
        self.ndim = np.shape(X)[1]
        if self.ndim not in [1, 2]:
            raise ValueError('two-pcf support only 1d and 2d modeling for the moment. curent ndim: %i'%(self.ndim))
        if self.ndim == 2:
            self.X = X
        if self.ndim == 1:
            self.X = np.array([X.T, np.zeros_like(X.T)]).T[:,0]
        self.y = y
        self.y_err = y_err
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.nbins = nbins
        self.anisotropic = anisotropic
        self.robust_fit = robust_fit
        self.p0_robust_fit = p0

    def resample_bootstrap(self):
        """
        Make a single bootstrap resampling on data.
        """
        npsfs = len(self.y)

        ind_object = np.random.randint(0,npsfs-1, size=npsfs)
        u_ressample = self.X[:,0][ind_object]
        v_ressample = self.X[:,1][ind_object]
        y_ressample = self.y[ind_object]
        y_err_ressample = self.y_err[ind_object]

        return u_ressample, v_ressample, y_ressample, y_err_ressample

    def comp_2pcf(self, X, y, y_err):
        """
        Estimate 2-point correlation function using TreeCorr.

        :param X:  Coordinates of the field. (n_samples, 1 or 2)
        :param y:  Values of the field. (n_samples)
        :param y_err: Error of y. (n_samples)
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

            dy = (kk.bottom_edges + kk.top_edges) / 2.
            dx = dy.T

            distance = np.array([dx.reshape(npixels), dy.reshape(npixels)]).T
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
        print('start return_2pcf')
        xi, distance, coord, mask = self.comp_2pcf(self.X, self.y, self.y_err)
        print('xi = ',xi)
        if self.anisotropic:
            # Choice done from Andy Taylor et al. 2012
            # see https://doi.org/10.1093/mnras/stt270
            # equation 35
            def f_bias(x, npixel=len(xi[mask])):
                top = x - 1.
                bottom = x - npixel - 2.
                return (top/bottom) - 2.
            results = optimize.fsolve(f_bias, len(xi[mask]) + 10)
            print('results = ',results)
            xi_cov = self.comp_xi_covariance(n_bootstrap=int(results[0]), mask=mask, seed=seed)
            print('xi_cov = ',xi_cov)
            bias_factor = (int(results[0]) - 1.) / (int(results[0]) - len(xi[mask]) - 2.)
            xi_weight = np.linalg.inv(xi_cov) * bias_factor
            print('xi_wt = ',xi_weight)
        else:
            # let like developed initialy for the moment
            xi_weight = np.eye(len(xi)) * 1./np.var(self.y)
        return xi, xi_weight, distance, coord, mask

    def optimizer(self, kernel):
        """
        Fit hyperparameter using two-point correlation function.
        Used chi2 minimization with L-BFGS-B method from scipy.

        :param kernel: sklearn.gaussian_process kernel.
        """
        print('start optimizer')
        size_x = np.max(self.X[:,0]) - np.min(self.X[:,0])
        if self.ndim == 2:
            size_y = np.max(self.X[:,1]) - np.min(self.X[:,1])
            rho = float(len(self.X[:,0])) / (size_x * size_y)
        if self.ndim == 1:
            size_y = 0.
            rho = float(len(self.X[:,0])) / size_x
        print('rho = ',rho)
        # if min_sep is None and isotropic GP, set min_sep to the average separation
        # between data.
        if self.min_sep is not None:
            min_sep = self.min_sep
        else:
            if self.anisotropic:
                min_sep = 0.
            else:
                min_sep = np.sqrt(1./rho)
        print('min_sep = ',min_sep)
        # if max_sep is None, set max_sep to half of the size of the 
        # given field.
        if self.max_sep is not None:
            max_sep = self.max_sep
        else:
            max_sep = np.sqrt(size_x**2 + size_y**2)/2.
        print('max_sep = ',min_sep)

        self.min_sep = min_sep
        self.max_sep = max_sep

        xi, xi_weight, distance, coord, mask = self.return_2pcf()
        print('xi = ',xi)

        def PCF(param, k=kernel):
            kernel = k.clone_with_theta(param)
            pcf = kernel.__call__(coord,Y=np.zeros_like(coord))[:,0]
            return pcf

        xi_mask = xi[mask]
        print('xi_mask = ',xi_mask)
        def chi2(param):
            residual = xi_mask - PCF(param)[mask]
            return residual.dot(xi_weight.dot(residual))

        print('robust_fit? ',self.robust_fit)
        if self.robust_fit:
            robust = robust_2dfit(kernel, xi,
                                  coord[:,0], coord[:,1], 
                                  xi_weight, mask=mask)
            robust.minimize_minuit(p0=self.p0_robust_fit)
            kernel = copy.deepcopy(robust.kernel_fit)
            cst = robust.result[-1]
            self._results_robust = robust.result
            print('results_robust  ',robust.result)
        else:
            p0 = kernel.theta
            results_fmin = optimize.fmin(chi2,p0,disp=False)
            results_bfgs = optimize.minimize(chi2,p0,method="L-BFGS-B")
            results = [results_fmin, results_bfgs['x']]
            chi2_min = [chi2(results[0]), chi2(results[1])]
            ind_min = chi2_min.index(min(chi2_min))
            results = results[ind_min]
            kernel = kernel.clone_with_theta(results)
            cst = 0
            print('results  ',results)

        self._2pcf = xi
        self._2pcf_weight = xi_weight
        self._2pcf_dist = distance
        self._2pcf_fit = PCF(kernel.theta) + cst
        self._2pcf_mask = mask
        self._kernel = copy.deepcopy(kernel)
        print('done making kernel')
        return kernel
