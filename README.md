[![Build Status](https://github.com/PFLeget/treegp/actions/workflows/test_treegp.yaml/badge.svg)](https://github.com/PFLeget/treegp/actions)
[![Codecov](https://codecov.io/gh/PFLeget/treegp/branch/master/graph/badge.svg)](https://codecov.io/gh/PFLeget/treegp)
[![Read the Docs](https://readthedocs.org/projects/treegp/badge/?version=latest)](https://treegp.readthedocs.io/en/latest/?badge=latest)
[![astro-ph.IM](https://img.shields.io/badge/astro--ph.IM-2103.09881-red)](https://arxiv.org/abs/2103.09881)
[![DOI](https://img.shields.io/badge/DOI-10.1051%2F0004--6361%2F202140463-blue.svg)](https://doi.org/10.1051/0004-6361/202140463)

## Overview

`treegp` is a python gaussian process code that perform 1D and 2D interpolation.

`treegp` has some special features compared to other available Gaussian Processes codes:

- Hyperparameters estimation will scale in $\cal{O}(n \ log(n))$ with the 2-points correlation function estimation compared to $\cal{O}(n^3)$ with the classical maximum likelihood.
- Gaussian process interpolation can be performed around a mean function
- A tool is provided to compute the mean function (`meanify`)

`treegp` was originally developed for Point Spread Function interpolation within [Piff](https://github.com/rmjarvis/Piff). There is a specific article that describes the math used in `treegp` in the context of modeling astrometric shifts of the Subaru Telescope due to atmospheric turbulences. This article can be found [here](	https://doi.org/10.1051/0004-6361/202140463).

## Installation

The easiest way to install is usually:

```bash
pip install treegp
```

which will install the latest released version.

If you would instead like to install the development version, you can do so via:

```bash
git clone https://github.com/PFLeget/treegp.git
cd treegp/
python setup.py install
```

## Dependencies

treegp has for now the following dependencies (see the quick installs below):

libraries listed in the [requirements](requirements.txt) file

## Python

treegp is regularly tested on Python 3.7, 3.8, 3.9 and 3.10 It may work in other versions of Python (e.g. pypy), but these are not currently supported.
