import numpy as np
import copy
from scipy import optimize
from scipy.linalg import cholesky, cho_solve

class log_likelihood(object):

    def __init__(self, X, y, y_err):
        """Fit statistical uncertaintie on two-point correlation function using bootstraping.

        :param X:      The independent covariates.  (n_samples, 2)
        :param y:      The dependent responses.  (n_samples, n_targets)
        :param y_err:  Error of y. (n_samples, n_targets)
        """
        self.X = X
        self.ndata = len(self.X[:,0])
        self.y = y
        self.y_err = y_err

    def log_likelihood(self, kernel):
        """
        Return of log likehood of gaussian process
        for given hyperparameters.

        :param kernel: Sklearn kernel object.
        """
        try:
            K = kernel.__call__(self.X) + np.eye(len(self.y))*self.y_err**2
            L = cholesky(K, overwrite_a=True, lower=False)
            alpha = cho_solve((L, False), self.y, overwrite_b=False)
            chi2 = np.dot(self.y, alpha)
            log_det = np.sum(2.*np.log(np.diag(L)))

            log_likelihood = -0.5 * chi2
            log_likelihood -= (self.ndata / 2.) * np.log((2. * np.pi))
            log_likelihood -= 0.5 * log_det
        except:
            log_likelihood = -np.inf

        return log_likelihood

    def optimizer(self, kernel):
        """
        Fit hyperparameter using maximum likelihood fit.
        Used minimization with L-BFGS-B method from scipy.

        :param kernel: sklearn.gaussian_process kernel.
        """
        def _minus_logl(param, k=kernel):
            kernel = k.clone_with_theta(param)
            log_l = self.log_likelihood(kernel)
            return -log_l

        p0 = kernel.theta
        results_bfgs = optimize.minimize(_minus_logl, p0, method="L-BFGS-B")
        results = results_bfgs['x']
        kernel = kernel.clone_with_theta(results)
        self._kernel = copy.deepcopy(kernel)
        self._logL = self.log_likelihood(self._kernel)
        return kernel
