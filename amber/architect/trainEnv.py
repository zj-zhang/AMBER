# -*- coding: UTF-8 -*-
"""
Training environment provides interactions between several components within architect and outside.

"""

from .optim.controller.controllerTrainEnv import ControllerTrainEnvironment, EnasTrainEnv, MultiManagerEnvironment, ParallelMultiManagerEnvironment

__all__ = [
    'ControllerTrainEnvironment',
    'EnasTrainEnv',
    'MultiManagerEnvironment',
    'ParallelMultiManagerEnvironment'    
]