"""Main entry for AMBER"""

from .wrapper import Amber, AmberSpecifications
from .getter import DataToParse
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print(e)
    tf.logging.set_verbosity(tf.logging.ERROR)

from .wrapper import Amber
from . import architect, modeler, utils, plots

__version__ = "0.1.3"

__all__ = [
    'Amber',
    'AmberSpecifications',
    'DataToParse',
    'architect',
    'modeler',
    'utils',
    'plots'
]
