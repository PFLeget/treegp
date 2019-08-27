# Some helper functions that mutliple test files might want to use

import numpy as np

def timer(f):
    import functools

    @functools.wraps(f)
    def f2(*args, **kwargs):
        import time
        import inspect
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        fname = repr(f).split()[1]
        print('time for %s = %.2f' % (fname, t1-t0))
        return result
    return f2

def get_correlation_length_matrix(size, e1, e2):
    """
    Produce correlation matrix to introduce anisotropy in kernel. 
    Used same parametrization as shape measurement in weak-lensing 
    because this is mathematicaly equivalent (anistropic kernel 
    will have an elliptical shape).

    :param correlation_length: Correlation lenght of the kernel.
    :param g1, g2:             Shear applied to isotropic kernel.
    """
    if abs(e1)>1:
        e1 = 0
    if abs(e2)>1:
        e2 = 0
    e = np.sqrt(e1**2 + e2**2)
    q = (1-e) / (1+e)
    phi = 0.5 * np.arctan2(e2,e1)
    rot = np.array([[np.cos(phi), np.sin(phi)],
                    [-np.sin(phi), np.cos(phi)]])
    ell = np.array([[size**2, 0],
                    [0, (size * q)**2]])
    L = np.dot(rot.T, ell.dot(rot))
    return L

def make_1d_grf(kernel, noise=None, seed=42, npoints=40):
    """
    Function to generate a 1D gaussian random field for a 
    given scikit-learn kernel.
    
    :param kernel:  given sklearn kernel.
    :param noise:   float. Level of noise to add to the 
                    gaussian randomn field. (default: None)
    :param seed:    int. seed of the random process. (default: 42) 
    :param npoints: int. number of points to generate for the 
                    simulations.
    """
    # fixing the seed
    np.random.seed(seed)
    # generate random 1D coordinate
    x = np.random.uniform(-10,10, npoints).reshape((npoints,1))
    # creating the correlation matrix / kernel 
    K = kernel.__call__(x)
    # generating gaussian random field
    y = np.random.multivariate_normal(np.zeros(npoints), K)
    if noise is not None:
        # adding noise
        y += np.random.normal(scale=noise, size=npoints)
        y_err = np.ones_like(y) * noise
        return x, y, y_err
    else:
        return x, y, None

def make_2d_grf(kernel, noise=None, seed=42, N_points=40):
    """
    Function to generate a 1D gaussian random field for a 
    given scikit-learn kernel.
    
    :param kernel:  given sklearn kernel.
    :param noise:   float. Level of noise to add to the 
                    gaussian randomn field. (default: None)
    :param seed:    int. seed of the random process. (default: 42) 
    :param npoints: int. number of points to generate for the 
                    simulations.
    """
    # fixing the seed
    np.random.seed(seed)
    # generate random 2D coordinate
    x1 = np.random.uniform(-10,10, N_points)
    x2 = np.random.uniform(-10,10, N_points)
    x = np.array([x1, x2]).T
    # creating the correlation matrix / kernel 
    K = kernel.__call__(x)
    # generating gaussian random field
    y = np.random.multivariate_normal(np.zeros(N_points), K)
    if noise is not None:
        # adding noise
        y += np.random.normal(scale=noise, size=N_points)
        y_err = np.ones_like(y) * noise
        return x, y, y_err
    else:
        return x, y, None
