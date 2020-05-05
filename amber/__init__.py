"""Docstring"""
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except Exception as e:
    print(e)
    tf.logging.set_verbosity(tf.logging.ERROR)

from ._wrapper import Amber

__all__ = [
    'Amber',
    'architect',
    'bootstrap',
    'interpret',
    'modeler',
    'objective',
    'plots',
    'utils'
]
