"""
treegp.
"""

# The version is stored in _version.py as recommended here:
# http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
from ._version import __version__, __version_info__

from .gp_interp import GPInterpolation

from .two_pcf import two_pcf
from .log_likelihood import log_likelihood

from .kernels import AnisotropicRBF
from .kernels import VonKarman
from .kernels import AnisotropicVonKarman
from .kernels import eval_kernel

from .meanify import meanify

from .utils import comp_eb

__all__ = [
    "__version__",
    "__version_info__",
    "GPInterpolation",
    "two_pcf",
    "log_likelihood",
    "AnisotropicRBF",
    "VonKarman",
    "AnisotropicVonKarman",
    "eval_kernel",
    "meanify",
    "comp_eb",
]
