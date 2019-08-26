from __future__ import print_function
import numpy as np
import treegp

from treegp_test_helper import timer

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

    #test_anisotropic_rbf_kernel()
    #test_vonkarman_kernel()
    #test_anisotropic_vonkarman_kernel()
    test_anisotropic_limit()
