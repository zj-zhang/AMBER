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
import contextlib
from typing import List, Tuple, Callable, Dict, Union, Generator

# model
def Model(inputs, outputs):
    pass

def Sequential(layers) -> Callable: # type:ignore
    pass

def trainable_variables(scope: str) -> Union[List, Tuple]: # type:ignore
    pass

def get_optimizer(opt, parameters, opt_config=None):
    pass

def get_loss(loss, y_true, y_pred) -> float:  # type: ignore
    pass

def get_metric(m: Union[str, Callable]) -> Callable: # type: ignore
    pass

def get_callback(m: Union[str, Callable]) -> Callable: # type:ignore
    pass

def get_train_op(loss, variables, optimizer, **kwargs) -> Tuple: # type: ignore
    pass

def get_layer(x, op) -> Callable: # type: ignore
    pass

def plot_model(model, to_file): # type:ignore
    pass

# tensor
def data_type_dict() -> Dict: # type: ignore
    pass

def TensorType(): # type:ignore
    pass

def Variable(initial_value, dtype, shape, trainable=True, name=None): # type:ignore
    pass

def to_numpy(tensor) -> np.ndarray: #type:ignore
    pass

def placeholder(shape, dtype, name=None) -> Union[float, np.ndarray]: # type:ignore
    pass

def assign(x, y) -> Callable: # type:ignore
    pass

def ones(shape, dtype=None) -> np.ndarray: # type:ignore
    pass

def zeros(shape, dtype=None) -> np.ndarray: # type:ignore
    pass

def zeros_like(input) -> np.ndarray: #type:ignore
    pass

def one_hot(tensor, num_classes=-1) -> np.ndarray: # type:ignore
    pass

def create_parameter(name, shape, dtype='float32', initializer=None, trainable=True, seed=None) -> np.ndarray: # type: ignore
    pass

def get_param_name(x) -> str: # type:ignore
    pass

def get_shape(x) -> np.ndarray: # type:ignore
    pass

shape = get_shape

def reshape(x, shape) -> np.ndarray: # type: ignore
    pass

def expand_dims(x, axis) -> np.ndarray: # type:ignore
    pass

def squeeze(x, axis) -> np.ndarray: # type: ignore
    pass

def concat(x, axis=-1) -> np.ndarray: # type: ignore
    pass

def split(x, num_splits, axis) -> List[np.ndarray]: # type: ignore
    pass

def where(condition, x=None, y=None, name=None) -> np.array: #type:ignore
    pass

def cast(x, dtype) -> np.array: # type:ignore
    pass

def fill(dims, value) -> np.array: # type:ignore
    pass

def map_fn(f: Callable, x) -> np.array: # type:ignore
    pass

def cond(pred, true_fn=None, false_fn=None, name=None) -> np.ndarray: # type:ignore
    pass

def stop_gradient(input) -> np.ndarray: # type:ignore
    pass

def embedding_lookup(params, ids) -> np.array: # type:ignore
    pass

def transpose(x, perm=None) -> np.ndarray: # type:ignore
    pass

def stack(x) -> np.ndarray: # type:ignore
    pass

# math
def get_math_func(n: Union[str,Callable]) -> Callable: # type:ignore
    pass

def abs(x) -> np.ndarray: # type: ignore
    pass

def minimum(x,y) -> np.ndarray: # type: ignore
    pass

def maximum(x,y) -> np.ndarray: # type: ignore
    pass

def less(x,y) -> Union[float, np.ndarray]: # type:ignore
    pass

def greater(x,y) -> Union[float, np.ndarray]: # type:ignore
    pass

def equal(x,y) -> np.bool: # type:ignore
    pass

def pow(x, y) -> Union[float, np.ndarray]: # type:ignore
    pass

def case(pred_fn_pairs, default=None, exclusive=False, strict=False, name='case'): # type:ignore
    pass

def clip_by_value(x, clip_value_min, clip_value_max) -> np.ndarray: # type: ignore
    pass

def matmul(x,y) -> np.ndarray: # type: ignore
    pass

def reduce_sum(x, axis=None) -> Union[float, np.ndarray]: # type: ignore
    pass

def reduce_mean(x, axis=None) -> Union[float, np.ndarray]: # type: ignore
    pass

def log(x) -> Union[float, np.ndarray]: # type: ignore
    pass

def exp(x) -> Union[float, np.ndarray]: # type: ignore
    pass

def sigmoid(x) -> Union[float, np.ndarray]: # type: ignore
    pass

def tanh(x) -> Union[float, np.ndarray]: # type: ignore
    pass

def relu(x) -> Union[float, np.ndarray]: # type: ignore
    pass

def softmax(x, axis=None) -> Union[float, np.ndarray]: # type: ignore
    pass

def boolean_mask(tensor, mask, axis=None, name='boolean_mask') -> np.array: # type:ignore
    pass

def logical_and(x, y) -> np.array: # type:ignore
    pass

def range(start, limit, delta=1, dtype=None, name='range') -> np.ndarray: # type:ignore
    pass

def tensordot(a, b, axes, name=None) -> np.ndarray: # type:ignore
    pass

def multinomial(logits, num_samples) -> np.ndarray: # type:ignore
    pass

# session
def SessionType(): # type:ignore
    pass

def Session(): # type:ignore
    pass

def TestCase(): # type:ignore
    pass

def session_cache() -> Dict: # type:ignore
    pass

def init_all_params(sess, var_scope=None) -> None: # type:ignore
    pass

def set_session(): # type:ignore
    pass

def get_session(): # type:ignore
    pass

def clear_session(): # type:ignore
    pass

@contextlib.contextmanager
def variable_scope(name, *args, **kwargs):
    pass

class device_scope:
    def __init__(self, name, *args, **kwargs):
        pass
    def __enter__(self):
        pass
    def __exit__(self):
        pass

class session_scope:
    def __init__(self, name, *args, **kwargs):
        pass
    def __enter__(self):
        pass
    def __exit__(self):
        pass

