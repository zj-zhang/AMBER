"""The base class for modelers with typing hints
"""
from typing import Tuple, List, Union
from collections import OrderedDict
import os
from argparse import Namespace
from typing import Optional, Union, List, Dict, Any, Sequence

import pandas as pd
import numpy as np
from .. import backend as F
import copy

FlexibleOperations = Union[Sequence[F.Operation], F.Operation]

class MockModel:
    def __init__(self, *args, **kwargs):
        pass
    def fit(self, x, y=None, *args, **kwargs):
        return 0
    def evaluate(self, x, y=None, *args, **kwargs):
        return 0
    def predict(self, x, *args, **kwargs):
        x_len = len(x[0]) if type(x) in (tuple, list) else len(x)
        return np.zeros((x_len, 1))

class BaseModelBuilder:
    def __init__(self, inputs_op: FlexibleOperations, output_op: FlexibleOperations, model_compile_dict=None, *args, **kwargs):
        self.model_compile_dict = model_compile_dict or {}
        pass

    def __call__(self, model_states: Sequence[Union[int,F.Operation]]) -> F.Model:
        return MockModel()

class BaseArchitectDecoder:
    pass
