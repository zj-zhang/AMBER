# -*- coding: UTF-8 -*-

"""
Implementations of NAS controller for searching architectures

Changelog
----------
    - Aug. 7, 2018: initial
    - Feb. 6. 2019: finished initial OperationController
    - Jun. 17. 2019: separated to OperationController and GeneralController
    - Aug. 15, 2020: updated documentations

"""


from ._general_controller import BaseController, GeneralController
from ._multiio_controller import MultiInputController, MultiIOController
from ._operation_controller import OperationController


__all__ = [
    'GeneralController',
    'OperationController'
]
