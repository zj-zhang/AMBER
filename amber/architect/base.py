"""Typing definitionsa and hints for amber.architect
"""
from typing import List, Tuple, Dict, Set, Callable, Optional, Union, Sequence
from amber import backend as F
import numpy as np

class BaseSearcher:
    pass

class BaseSearchEnvironment:
    pass

class BaseNetworkManager:
    def __init__(self, model_fn=Callable, working_dir=Optional[str], model_compile_dict=None, *args, **kwargs):
        self.model_fn = model_fn
        self.working_dir = working_dir
        self.model_compile_dict = model_compile_dict or {}

    def get_rewards(self, trial: int, model_arc: Sequence[Union[int,F.Operation]], *args, **kwargs) -> float:
        pass

class BaseModelSpace:
    pass

class BaseModelBuffer:
    pass


class BaseReward:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, model: F.Model, data: Union[np.ndarray, Tuple, List], *args, **kwargs) -> Tuple[float, list, list]:
        return 0, [0,], []