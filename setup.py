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

package_data = {}

setup(name=name,
      description="treegp",
      long_description=long_description,
      license = "BSD License",
      classifiers=["Topic :: Scientific/Engineering :: Astronomy",
                   "Intended Audience :: Science/Research"],
      author="PFLeget",
      author_email="plget [at] lpnhe.in2p3.fr",
      url="https://github.com/PFLeget/treegp",
      download_url="https://github.com/PFLeget/treegp/releases/tag/v%s.zip"%treegp_version,
      version=treegp_version,
      packages=packages,
      scripts=scripts)
