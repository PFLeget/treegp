.. image:: https://github.com/PFLeget/treegp/actions/workflows/test_treegp.yaml/badge.svg
   :target: https://github.com/PFLeget/treegp/actions
.. image:: https://codecov.io/gh/PFLeget/treegp/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/PFLeget/treegp
.. image:: https://readthedocs.org/projects/treegp/badge/?version=latest
  :target: https://treegp.readthedocs.io/en/latest/?badge=latest
.. image:: https://img.shields.io/badge/astro--ph.IM-2103.09881-red
    :target: https://arxiv.org/abs/2103.09881
.. image:: https://img.shields.io/badge/DOI-10.1051%2F0004--6361%2F202140463-blue.svg
   :target: https://doi.org/10.1051/0004-6361/202140463
.. image:: https://img.shields.io/badge/RTN--129-TURBO--GP-blue
   :target: https://rtn-129.lsst.io
		

.. inclusion-marker-do-not-remove

Overview
--------

``treegp`` is a python gaussian process code that perform 1D and 2D interpolation.

``treegp`` has some special features compared to other available Gaussian Processes codes:

*   Hyperparameters estimation will scale in O(N log(N)) with the the 2-points correlation function estimation compared to O(N^3) with the classical maximum likelihood.

*   Gaussian process interpolation can be performed around a mean function

*   A tool is provided to compute the mean function (``meanify``)

*   TURBO-GP (TUrbulence Removal By fourier-Optimized Gaussian Process, `Léget et al., RTN-129 <https://rtn-129.lsst.io>`_): the measured anisotropic 2D 2-points correlation function is used directly as the kernel, without fitting any hyperparameters (``optimizer="empirical-2pcf"``, following `Gomes et al. (2025) <https://doi.org/10.3847/1538-3881/ae1a7b>`_), and the Gaussian process is solved on a fine grid with the FFT and a preconditioned conjugate gradient (``solve_method="spectral"``, the default for this optimizer): no O(N^3) factorization, no dense covariance matrix, a positive semi-definite kernel by construction. The other optimizers (``"two-pcf"``, ``"anisotropic"``) implement `Léget et al. (2021) <https://arxiv.org/abs/2103.09881>`_.

*   An interactive, step-by-step explanation of TURBO-GP (with live demos on a real LSSTCam visit) is at https://pfleget.github.io/treegp/ (source in ``slides/``), and the `documentation <https://treegp.readthedocs.io/en/latest/treegp_gp_interp.html#worked-example-l21-and-turbo-gp-on-a-real-lsstcam-visit>`_ has a worked example running both Léget et al. (2021) and TURBO-GP on that visit.

``treegp`` was originally developed for Point Spread Function interpolation within `Piff <https://github.com/rmjarvis/Piff>`_. There is a specific article that describes the math used in ``treegp`` in the context of modelling astrometric shifts of the Subaru Telescope due to atmospheric turbulences. This article can be found
`here <https://arxiv.org/abs/2103.09881>`_.


Installation
------------

The easiest way to install is usually::

  pip install treegp

which will install the latest released version.

If you would instead like to install the development version, you can do so via::

  git clone https://github.com/PFLeget/treegp.git
  cd treegp/
  python setup.py install


Dependencies
------------

``treegp`` has for now the following dependencies (see the quick
installs below):

- libraries listed in the `requirements <requirements.txt>`_ file


Python
``````

``treegp`` is regularly tested on Python 3.9, 3.10, 3.11, 3.12 and 3.13.  It may work in other
versions of Python (e.g. pypy), but these are not currently supported.
