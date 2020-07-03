#!/usr/bin/env python

"""Setup script."""

from __future__ import print_function
import glob
import os
import re
from setuptools import setup, find_packages

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

# Packages (subdirectories in clusters/)
packages = find_packages()

# Scripts (in scripts/)
scripts = glob.glob("scripts/*.py")

package_data = {}

setup(name=name,
      description=("treegp"),
      classifiers=["Topic :: Scientific/Engineering :: Astronomy",
                   "Intended Audience :: Science/Research"],
      author="PFLeget",
      version=treegp_version,
      packages=packages,
      scripts=scripts)
