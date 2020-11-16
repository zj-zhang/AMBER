"""
The :mod:`architect` module provides neural architecture search implementations and its related helpers

"""


from .controller import GeneralController, MultiInputController, MultiIOController, OperationController
from .modelSpace import State, ModelSpace
from .manager import GeneralManager, NetworkManager
from .trainEnv import ControllerTrainEnvironment, EnasTrainEnv

# alias
Operation = State

# TODO: Do not include MultiIO until its tested in multiio branch
__all__ = [
    'GeneralController',
    'Operation',
    'State',
    'ModelSpace',
    'GeneralManager',
    'ControllerTrainEnvironment',
    'EnasTrainEnv',
    # For legacy use
    'OperationController',
    'NetworkManager'
]