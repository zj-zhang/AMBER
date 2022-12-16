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

import sys
from .base import BaseController
from ....backend import mod_name, _gen_missing_api

lib_types = {
    'tensorflow_1': 'static',
    'pytorch': 'dynamic',
    'tensorflow_2': 'dynamic'
}

controller_cls = ['GeneralController', 'MultiInputController', 'MultiIOController', 'OperationController', 'ZeroShotController']

if lib_types[mod_name] == 'static':
    from . import static as mod
elif lib_types[mod_name] == 'dynamic':
    from . import dynamic as mod
else:
    raise Exception(f"Unsupported {mod_name} for supernet; must be in {list(lib_types.keys())}")

thismod = sys.modules[__name__]
for api in controller_cls:
    if api in mod.__dict__:
        setattr(thismod, api, mod.__dict__[api])
    else:
        setattr(thismod, api, _gen_missing_api(api, mod_name))


__all__ = [
    'BaseController',
] + controller_cls

