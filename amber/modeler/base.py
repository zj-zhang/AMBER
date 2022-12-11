"""The base class for modelers with typing hints
"""
from typing import Tuple, List, Union
from collections import OrderedDict
import os
from argparse import Namespace
from typing import Optional, Union, List, Dict, Any, Sequence

import pandas as pd
from .. import backend as F
import copy

FlexibleOperations = Union[Sequence[F.Operation], F.Operation]

class BaseModelBuilder:
    def __init__(self, inputs_op: FlexibleOperations, output_op: FlexibleOperations, model_compile_dict=None, *args, **kwargs):
        self.model_compile_dict = model_compile_dict or {}
        pass

    def __call__(self, model_states: Sequence[Union[int,F.Operation]]) -> F.Model:
        pass

class BaseArchitectDecoder:
    pass
