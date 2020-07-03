#!/usr/bin/env python

"""Setup script."""

from __future__ import print_function
import glob
import sys
import os
import re
import setuptools
from setuptools import setup, find_packages

# Print some useful information in case there are problems, this info will help troubleshoot.
print("Using setuptools version",setuptools.__version__)
print("Python version = ",sys.version)

with open('README.rst') as f:
    long_description = f.read()

# Read in the version from treegp/_version.py
# cf. http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
version_file = os.path.join('treegp', '_version.py')
with open(version_file, "rt") as f:
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    txt = f.read()
    mo = re.search(VSRE, txt, re.M)
    if mo:
        treegp_version = mo.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s." % (version_file,))
print("treegp version is %s" % (treegp_version))

# Package name
name = 'treegp'

# Packages
packages = find_packages()

# Scripts (none yet, but if we make any, they will live in scripts/)
scripts = glob.glob("scripts/*.py")

# Dependencies
dependencies = ['numpy', 'scipy', 'treecorr>=4.0', 'fitsio>=0.9.12', 'scikit-learn>=0.18']
if sys.version >= '3.0':
    dependencies += ['iminuit']
else:
    # iminuit 1.4 fails on python 2.7
    dependencies += ['iminuit==1.3.8']

package_data = {}

setup(name=name,
      description="treegp",
      long_description=long_description,
      license = "BSD License",
      classifiers=["Topic :: Scientific/Engineering :: Astronomy",
                   "Intended Audience :: Science/Research"],
      author="PFLeget",
      url="https://github.com/PFLeget/treegp",
      download_url="https://github.com/PFLeget/treegp/releases/tag/v%s.zip"%treegp_version,
      install_requires=dependencies,
      version=treegp_version,
      packages=packages,
      scripts=scripts)
