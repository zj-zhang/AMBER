"""Main entry for AMBER"""

from .wrapper import Amber, AmberSpecifications
from .getter import DataToParse
from . import backend
import os

from .wrapper import Amber
from . import architect, modeler, utils, plots

__version__ = "0.1.4"

__all__ = [
    'Amber',
    'AmberSpecifications',
    'DataToParse',
    'backend',
    'architect',
    'modeler',
    'utils',
    'plots'
]
