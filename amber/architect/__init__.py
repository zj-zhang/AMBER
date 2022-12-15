"""
provides search algorithms and helpers for neural network architecture
"""

from .optim import pmbga
from .optim import controller
from .optim.controller import BaseController, GeneralController, MultiInputController, MultiIOController, OperationController, ZeroShotController
from .modelSpace import State, ModelSpace
from .manager import GeneralManager, EnasManager, DistributedGeneralManager
from .trainEnv import ControllerTrainEnvironment, EnasTrainEnv, MultiManagerEnvironment, ParallelMultiManagerEnvironment
from . import buffer, store, reward, trainEnv, modelSpace, optim, base

# alias
Operation = State

__all__ = [
    # funcs
    'BaseController',
    'GeneralController',
    'MultiIOController',
    'MultiInputController',
    'Operation',
    'State',
    'ModelSpace',
    'GeneralManager',
    'EnasManager',
    'ControllerTrainEnvironment',
    'MultiManagerEnvironment',
    'EnasTrainEnv',
    # modules
    'buffer',
    'store',
    'reward',
    'trainEnv',
    'modelSpace',
    'controller',
    # For legacy use
    'OperationController',
]
