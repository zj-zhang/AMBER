"""
provides search algorithms and helpers for neural network architecture
"""


from .controller import BaseController, GeneralController, MultiInputController, MultiIOController, OperationController, \
    ZeroShotController
from .modelSpace import State, ModelSpace
from .manager import GeneralManager, EnasManager, DistributedGeneralManager
from .trainEnv import ControllerTrainEnvironment, EnasTrainEnv, MultiManagerEnvironment, ParallelMultiManagerEnvironment
from . import buffer, store, reward, trainEnv, modelSpace, controller, pmbga

# alias
Operation = State

# TODO: Do not include MultiIO until its tested in multiio branch
__all__ = [
    # funcs
    'BaseController',
    'GeneralController',
    'Operation',
    'State',
    'ModelSpace',
    'GeneralManager',
    'EnasManager',
    'ControllerTrainEnvironment',
    'EnasTrainEnv',
    # modules
    'buffer',
    'store',
    'reward',
    'trainEnv',
    'modelSpace',
    'controller',
    'pmbga',
    # For legacy use
    'OperationController',
]
