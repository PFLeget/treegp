"""
treegp.
"""

from .gp_interp import GPInterpolation

from .kernels import AnisotropicRBF
from .kernels import VonKarman
from .kernels import AnisotropicVonKarman
from .kernels import eval_kernel

from .meanify import meanify

