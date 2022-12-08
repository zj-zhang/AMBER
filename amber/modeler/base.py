"""The base class for modelers live here
"""
from typing import Tuple, List, Union
from collections import OrderedDict
import os
from argparse import Namespace
from typing import Optional, Union, List, Dict, Any

import pandas as pd
from .. import backend as F
import copy

class ModelBuilder:
    """Scaffold of Model Builder
    """

    def __init__(self, inputs, outputs, *args, **kwargs):
        raise NotImplementedError("Abstract method.")

    def __call__(self, model_states):
        raise NotImplementedError("Abstract method.")



class GeneralChild(F.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

