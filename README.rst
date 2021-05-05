.. image:: https://travis-ci.org/PFLeget/treegp.svg?branch=master
    :target: https://travis-ci.org/PFLeget/treegp
.. image:: https://codecov.io/gh/PFLeget/treegp/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/PFLeget/treegp
.. image:: https://img.shields.io/badge/astro--ph.IM-2103.09881-red
    :target: https://arxiv.org/abs/2103.09881

.. inclusion-marker-do-not-remove

Overview
--------

``treegp`` is a python gaussian process code that perform 1D and 2D interpolation.

``treegp`` have some specific features compared to other gaussian processes available code:

*   Hyperparameters estimation will scale in O(N) with the the 2-points correlation function estimation compared to O(N^3) with the classical maximum likelihood.
    
*   Gaussian process interpolation can be performed around a mean function
    
*   Tools are given to compute the mean function (``meanify``)

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

``treegp`` is regularly tested on Python 2.7, 3.6, 3.7, and 3.8.  It may work in other
versions of Python (e.g. pypy), but these are not currently supported.
