.. image:: https://travis-ci.org/PFLeget/treegp.svg?branch=master
    :target: https://travis-ci.org/PFLeget/treegp

.. inclusion-marker-do-not-remove

Overview
--------

treegp is a python gaussian process code that perform 1D and 2D interpolation.

treegp have some specific features compared to other gaussian processes available code:

*   It is possible to use informations from the 2-points correlation function to estimate hyperparameters
*   It is possible to perform the gaussian process interpolation around a mean function
*   Tools are given to compute the mean function

Waiting the paper, the description of the training can be found in french at least for the mean function
`here <https://tel.archives-ouvertes.fr/tel-01467899>`_ .


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

``treegp`` is regularly tested on Python 2.7, 3.5, 3.6, and 3.7.  It may work in other
versions of Python (e.g. pypy or 3.8), but these are not currently supported.
