from __future__ import print_function
import numpy as np
import treegp

from treegp_test_helper import timer
from treegp_test_helper import get_correlation_length_matrix

@timer
def test_anisotropic_rbf_kernel():
    from scipy.spatial.distance import pdist, squareform

    corr_length = [1., 30. ,30. ,30., 30.]
    g1 = [0, 0.4, 0.4, -0.4, -0.4]
    g2 = [0, 0.4, -0.4, 0.4, -0.4]
    kernel_amp = [1e-4, 1e-3, 1e-2, 1., 1.]
    dist = np.linspace(0,10,100)
    coord = np.array([dist,dist]).T

    dist = np.linspace(-10,10,21)

    X, Y = np.meshgrid(dist,dist)
    x = X.reshape(len(dist)**2)
    y = Y.reshape(len(dist)**2)
    coord_corr = np.array([x, y]).T

    def _anisotropic_rbf_kernel(x, sigma, corr_length, g1, g2):
        L = get_correlation_length_matrix(corr_length, g1, g2)
        invL = np.linalg.inv(L)
        dists = pdist(x, metric='mahalanobis', VI=invL)
        K = np.exp(-0.5 * dists**2)
        lim0 = 1.
        K = squareform(K)
        np.fill_diagonal(K, lim0)
        K *= sigma**2
        return K
        
    def _anisotropic_rbf_corr_function(x, y, sigma, 
                                       corr_length, g1, g2):
        L = get_correlation_length_matrix(corr_length, g1, g2)
        l = np.linalg.inv(L)
        dist_a = (l[0,0]*x*x) + (2*l[0,1]*x*y) + (l[1,1]*y*y)
        z = np.exp(-0.5 * dist_a)
        return z*sigma**2
    
    for i in range(5):
        L = get_correlation_length_matrix(corr_length[i], g1[i], g2[i])
        inv_L = np.linalg.inv(L)
        ker = kernel_amp[i]**2 * treegp.AnisotropicRBF(invLam=inv_L)
        ker_treegp = ker.__call__(coord)
        corr_treegp = ker.__call__(coord_corr,Y=np.zeros_like(coord_corr))[:,0]
        ker_test = _anisotropic_rbf_kernel(coord, kernel_amp[i], corr_length[i], g1[i], g2[i])
        corr_test = _anisotropic_rbf_corr_function(x, y, kernel_amp[i],
                                                   corr_length[i], g1[i], g2[i])
        np.testing.assert_allclose(ker_treegp, ker_test, atol=1e-12)
        np.testing.assert_allclose(corr_treegp, corr_test, atol=1e-12)
        
        hyperparameter = ker.theta
        theta = hyperparameter[1:]
        L1 = np.zeros_like(inv_L)
        L1[np.diag_indices(2)] = np.exp(theta[:2])
        L1[np.tril_indices(2, -1)] = theta[2:]
        invLam = np.dot(L1, L1.T)
        np.testing.assert_allclose(inv_L, invLam, atol=1e-12)

@timer
def test_vonkarman_kernel():
    from scipy import special

    corr_lenght = [1., 10., 100., 1000.]
    kernel_amp = [1e-4, 1e-3, 1e-2, 1.]
    dist = np.linspace(0, 10, 100)
    coord = np.array([dist, dist]).T

    dist = np.linspace(0.01, 10, 100)
    coord_corr = np.array([dist, np.zeros_like(dist)]).T

    def _vonkarman_kernel(param,x):
        A = (x[:,0]-x[:,0][:,None])
        B = (x[:,1]-x[:,1][:,None])
        distance = np.sqrt(A*A + B*B)
        Filter = distance != 0.
        K = np.zeros_like(distance)
        K[Filter] = param[0]**2 * ((distance[Filter]/param[1])**(5./6.) *
                                   special.kv(-5./6.,2*np.pi*distance[Filter]/param[1]))
        dist = np.linspace(1e-4,1.,100)
        div = 5./6.
        lim0 = special.gamma(div) /(2 * (np.pi**div) )
        K[~Filter] = param[0]**2 * lim0
        K /= lim0
        return K

    def _vonkarman_corr_function(param, distance):
        div = 5./6.
        lim0 = (2 * (np.pi**div) ) / special.gamma(div)
        return param[0]**2 * lim0 * ((distance/param[1])**(5./6.)) * special.kv(-5./6.,2*np.pi*distance/param[1])

    for corr in corr_lenght:
        for amp in kernel_amp:
            kernel = "%.10f * VonKarman(length_scale=%f)"%((amp**2,corr))
            interp = treegp.GPInterpolation(kernel=kernel,
                                            normalize=False,
                                            white_noise=0.)
            ker = interp.kernel_template

            ker_piff = ker.__call__(coord)
            corr_piff = ker.__call__(coord_corr,Y=np.zeros_like(coord_corr))[:,0]

            ker_test = _vonkarman_kernel([amp,corr],coord)
            corr_test = _vonkarman_corr_function([amp,corr], dist)

            np.testing.assert_allclose(ker_piff, ker_test, atol=1e-12)
            np.testing.assert_allclose(corr_piff, corr_test, atol=1e-12)

@timer
def test_anisotropic_vonkarman_kernel():
    from scipy import special
    from scipy.spatial.distance import pdist, squareform

    corr_length = [1., 30. ,30. ,30., 30.]
    g1 = [0, 0.4, 0.4, -0.4, -0.4]
    g2 = [0, 0.4, -0.4, 0.4, -0.4]
    kernel_amp = [1e-4, 1e-3, 1e-2, 1., 1.]
    dist = np.linspace(0,10,100)
    coord = np.array([dist,dist]).T

    dist = np.linspace(-10,10,21)

    X, Y = np.meshgrid(dist,dist)
    x = X.reshape(len(dist)**2)
    y = Y.reshape(len(dist)**2)
    coord_corr = np.array([x, y]).T

    def _anisotropic_vonkarman_kernel(x, sigma, corr_length, g1, g2):
        L = get_correlation_length_matrix(corr_length, g1, g2)
        invL = np.linalg.inv(L)
        dists = pdist(x, metric='mahalanobis', VI=invL)
        K = dists **(5./6.) *  special.kv(5./6., 2*np.pi * dists)
        lim0 = special.gamma(5./6.) /(2 * ((np.pi)**(5./6.)) )
        K = squareform(K)
        np.fill_diagonal(K, lim0)
        K /= lim0
        K *= sigma**2
        return K
        
    def _anisotropic_vonkarman_corr_function( x, y, sigma, 
                                              corr_length, g1, g2):
        L = get_correlation_length_matrix(corr_length, g1, g2)
        l = np.linalg.inv(L)
        dist_a = (l[0,0]*x*x) + (2*l[0,1]*x*y) + (l[1,1]*y*y)
        z = np.zeros_like(dist_a)
        Filter = dist_a != 0.
        z[Filter] = dist_a[Filter]**(5./12.) *  special.kv(5./6., 2*np.pi * np.sqrt(dist_a[Filter]))
        lim0 = special.gamma(5./6.) /(2 * ((np.pi)**(5./6.)) )
        if np.sum(Filter) != len(z):
            z[~Filter] = lim0
        z /= lim0
        return z*sigma**2
    
    for i in range(5):
        L = get_correlation_length_matrix(corr_length[i], g1[i], g2[i])
        inv_L = np.linalg.inv(L)
        ker = kernel_amp[i]**2 * treegp.AnisotropicVonKarman(invLam=inv_L)
        ker_treegp = ker.__call__(coord)
        corr_treegp = ker.__call__(coord_corr,Y=np.zeros_like(coord_corr))[:,0]
        ker_test = _anisotropic_vonkarman_kernel(coord, kernel_amp[i], corr_length[i], g1[i], g2[i])
        corr_test = _anisotropic_vonkarman_corr_function(x, y, kernel_amp[i],
                                                         corr_length[i], g1[i], g2[i])
        np.testing.assert_allclose(ker_treegp, ker_test, atol=1e-12)
        np.testing.assert_allclose(corr_treegp, corr_test, atol=1e-12)
        
        hyperparameter = ker.theta
        theta = hyperparameter[1:]
        L1 = np.zeros_like(inv_L)
        L1[np.diag_indices(2)] = np.exp(theta[:2])
        L1[np.tril_indices(2, -1)] = theta[2:]
        invLam = np.dot(L1, L1.T)
        np.testing.assert_allclose(inv_L, invLam, atol=1e-12)

@timer
def test_anisotropic_limit():
    """Test that AnisotropicRBF with isotropic covariance equals RBF"""

    np.random.seed(42)

    #test isotropic vs anisotropic RBF
    kernel1 = "RBF(0.45)"
    kernel2 = "AnisotropicRBF(scale_length=[0.45, 0.45])"

    gp1 = treegp.GPInterpolation(kernel=kernel1)
    gp2 = treegp.GPInterpolation(kernel=kernel2)

    X = np.random.rand(1000, 2)
    np.testing.assert_allclose(gp1.kernel_template.__call__(X), gp2.kernel_template.__call__(X))

    #test isotropic vs anisotropic VonKarman
    kernel3 = "VonKarman(0.45)"
    kernel4 = "AnisotropicVonKarman(scale_length=[0.45, 0.45])"

    gp3 = treegp.GPInterpolation(kernel=kernel1)
    gp4 = treegp.GPInterpolation(kernel=kernel2)

    X = np.random.rand(1000, 2)
    np.testing.assert_allclose(gp3.kernel_template.__call__(X), gp4.kernel_template.__call__(X))



if __name__ == "__main__":

    test_anisotropic_rbf_kernel()
    test_vonkarman_kernel()
    test_anisotropic_vonkarman_kernel()
    test_anisotropic_limit()
