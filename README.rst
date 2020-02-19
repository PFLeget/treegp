.. image:: https://travis-ci.org/PFLeget/treegp.svg?branch=master
    :target: https://travis-ci.org/PFLeget/treegp

____

**WARNING**: Package under development, more information will come soon

____

.. inclusion-marker-do-not-remove                                                                                            

treegp
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

To install::

  git clone https://github.com/PFLeget/treegp.git

To install treegp, use::

  cd treegp/
  python setup.py install

Dependencies
------------

``treegp`` has for now the following dependencies (see the quick
installs below):

- libraries listed in the `requirements <requirements.txt>`_ file
   

Python
``````

``treegp`` can be run in python 2 and python 3 (need to be done)
