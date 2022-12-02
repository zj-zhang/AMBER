"""This file defines the unified tensor and autograd framework interface required by AMBER.
The principles of this interface:
* There should be as few interfaces as possible.
* The interface is used by AMBER system so it is more important to have
  clean definition rather than convenient usage.
* Default arguments should be avoided.
* Keyword or positional arguments should be avoided.
* Argument type should be easier to understand.
It is recommended the frameworks implement all the interfaces. However, it is
also OK to skip some. The generated backend module has an ``is_enabled`` function
that returns whether the interface is supported by the framework or not.
"""

import numpy as np
from typing import List, Tuple, Callable, Dict, Union, Generator

# model
def Model(inputs, outputs):
    pass

def get_loss(loss, y_true, y_pred) -> float:  # type: ignore
    pass

def get_metric(m: Union[str, Callable]) -> Callable: # type: ignore
    pass

# layers
def create_parameter(name, shape, dtype='float32', initializer=None, trainable=True, seed=None) -> np.ndarray: # type: ignore
    pass

def get_train_op(loss, variables, optimizer, **kwargs) -> Tuple: # type: ignore
    pass

def get_layer(x, op) -> Callable: # type: ignore
    pass

# tensor
def data_type_dict() -> Dict: # type: ignore
    pass

def reshape(x, shape) -> np.ndarray: # type: ignore
    pass

def squeeze(x, axis) -> np.ndarray: # type: ignore
    pass

def concat(x, axis=-1) -> np.ndarray: # type: ignore
    pass

def split(x, num_splits, axis) -> List[np.ndarray]: # type: ignore
    pass

# math
def abs(x) -> np.ndarray: # type: ignore
    pass

def minimum(a,b) -> np.ndarray: # type: ignore
    pass

def maximum(a,b) -> np.ndarray: # type: ignore
    pass

def clip_by_value(x, clip_value_min, clip_value_max) -> np.ndarray: # type: ignore
    pass

def matmul(a,b) -> np.ndarray: # type: ignore
    pass

def reduce_sum(x, axis=-1) -> Union[float, np.ndarray]: # type: ignore
    pass

def reduce_mean(x, axis=-1) -> Union[float, np.ndarray]: # type: ignore
    pass

def log(x) -> Union[float, np.ndarray]: # type: ignore
    pass

def exp(x) -> Union[float, np.ndarray]: # type: ignore
    pass

def sigmoid(x) -> Union[float, np.ndarray]: # type: ignore
    pass

def tanh(x) -> Union[float, np.ndarray]: # type: ignore
    pass

# session
def session_cache() -> Dict: # type:ignore
    pass

def init_all_params(sess, var_scope=None) -> None: # type:ignore
    pass

class variable_scope:
    def __init__(self, name, *args, **kwargs):
        pass
    def __enter__(self):
        pass
    def __exit__(self):
        pass

