"""
SimpleDisk: Code to set up a multidust disk + envelope RADMC-3D models

Copyright (C) 2018 Laszlo Szucs <laszlo.szucs@mpe.mpg.de>

Licensed under GLPv2, for more information see the LICENSE file repository.
"""

from . import bayes

from .main import radmc3dModel, getParams
from .runner import radmc3dRunner

__version__ = "1.0"
__author__ = "Laszlo Szucs (laszlo.szucs@mpe.mpg.de)"

__all__ =  ["radmc3dModel","getParams","radmc3dRunner","bayes"]
