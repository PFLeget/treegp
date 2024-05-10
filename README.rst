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
		

.. inclusion-marker-do-not-remove

Overview
--------

``treegp`` is a python gaussian process code that perform 1D and 2D interpolation.

``treegp`` has some special features compared to other available Gaussian Processes codes:

*   Hyperparameters estimation will scale in O(N log(N)) with the the 2-points correlation function estimation compared to O(N^3) with the classical maximum likelihood.
    
*   Gaussian process interpolation can be performed around a mean function
    
*   A tool is provided to compute the mean function (``meanify``)

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

``treegp`` is regularly tested on Python 3.8, 3.9, 3.10, and 3.11.  It may work in other
versions of Python (e.g. pypy), but these are not currently supported.
