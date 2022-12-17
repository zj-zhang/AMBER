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
def Model(inputs, outputs): pass


def Sequential(layers) -> Callable: pass # type:ignore


def trainable_variables(scope: str) -> Union[List, Tuple]: pass # type:ignore
    

def get_optimizer(opt, parameters, opt_config=None): pass
    

def get_loss(loss, y_true, y_pred, reduction:str ='mean') -> float: pass  # type: ignore
    

def get_metric(m: Union[str, Callable]) -> Callable: pass # type: ignore
    

def get_callback(m: Union[str, Callable]) -> Callable: pass # type:ignore
    

def get_train_op(loss, variables, optimizer, **kwargs) -> Tuple: pass # type: ignore
    

def get_layer(x, op) -> Callable: pass # type: ignore
    

def plot_model(model, to_file): pass # type:ignore
    

# tensor
def data_type_dict() -> Dict: pass # type: ignore
    

def TensorType(): pass # type:ignore
    

def Variable(initial_value, dtype, shape, trainable=True, name=None): pass # type:ignore
    

def to_numpy(tensor) -> np.ndarray: pass #type:ignore
    

def placeholder(shape, dtype, name=None) -> Union[float, np.ndarray]: pass # type:ignore
    

def assign(x, y) -> Callable: pass # type:ignore
    

def ones(shape, dtype=None) -> np.ndarray: pass # type:ignore
    

def zeros(shape, dtype=None) -> np.ndarray: pass # type:ignore
    

def zeros_like(input) -> np.ndarray: pass #type:ignore
    

def one_hot(tensor, num_classes=-1) -> np.ndarray: pass # type:ignore
    

def create_parameter(name, shape, dtype='float32', initializer=None, trainable=True, seed=None) -> np.ndarray: pass # type: ignore
    

def get_param_name(x) -> str: pass # type:ignore
    

def get_shape(x) -> np.ndarray: pass # type:ignore
    

shape = get_shape

def reshape(x, shape) -> np.ndarray: pass # type: ignore
    

def expand_dims(x, axis) -> np.ndarray: pass # type:ignore
    

def squeeze(x, axis) -> np.ndarray: pass # type: ignore
    

def concat(x, axis=-1) -> np.ndarray: pass # type: ignore
    

def split(x, num_splits, axis) -> List[np.ndarray]: pass # type: ignore
    

def where(condition, x=None, y=None, name=None) -> np.array: pass  #type:ignore
    

def cast(x, dtype) -> np.array: pass # type:ignore
    

def fill(dims, value) -> np.array: pass # type:ignore
    

def map_fn(f: Callable, x) -> np.array: pass # type:ignore
    

def cond(pred, true_fn=None, false_fn=None, name=None) -> np.ndarray: pass # type:ignore
    

def stop_gradient(input) -> np.ndarray: pass # type:ignore
    

def embedding_lookup(params, ids) -> np.array: pass # type:ignore
    

def transpose(x, perm=None) -> np.ndarray: pass # type:ignore
    

def stack(x) -> np.ndarray: pass # type:ignore
    

# math
def get_math_func(n: Union[str,Callable]) -> Callable: pass # type:ignore
    

def abs(x) -> np.ndarray: pass # type: ignore
    

def minimum(x,y) -> np.ndarray: pass # type: ignore
    

def maximum(x,y) -> np.ndarray: pass # type: ignore
    

def less(x,y) -> Union[float, np.ndarray]: pass # type:ignore
    

def greater(x,y) -> Union[float, np.ndarray]: pass # type:ignore
    

def equal(x,y) -> np.bool: pass # type:ignore
    

def pow(x, y) -> Union[float, np.ndarray]: pass # type:ignore
    

def case(pred_fn_pairs, default=None, exclusive=False, strict=False, name='case'): pass # type:ignore
    

def clip_by_value(x, clip_value_min, clip_value_max) -> np.ndarray: pass # type: ignore
    

def matmul(x,y) -> np.ndarray: pass # type: ignore
    

def reduce_sum(x, axis=None) -> Union[float, np.ndarray]: pass # type: ignore
    

def reduce_mean(x, axis=None) -> Union[float, np.ndarray]: pass # type: ignore
    

def log(x) -> Union[float, np.ndarray]: pass # type: ignore
    

def exp(x) -> Union[float, np.ndarray]: pass # type: ignore
    

def sigmoid(x) -> Union[float, np.ndarray]: pass # type: ignore
    

def tanh(x) -> Union[float, np.ndarray]: pass # type: ignore
    

def relu(x) -> Union[float, np.ndarray]: pass # type: ignore
    

def softmax(x, axis=None) -> Union[float, np.ndarray]: pass # type: ignore
    

def boolean_mask(tensor, mask, axis=None, name='boolean_mask') -> np.array: pass # type:ignore
    

def logical_and(x, y) -> np.array: pass # type:ignore
    

def range(start, limit, delta=1, dtype=None, name='range') -> np.ndarray: pass # type:ignore
    

def tensordot(a, b, axes, name=None) -> np.ndarray: pass # type:ignore
    

def multinomial(logits, num_samples) -> np.ndarray: pass # type:ignore
    

# session
def SessionType(): pass # type:ignore
    

def Session(): pass # type:ignore
    

def TestCase(): pass # type:ignore
    

def session_cache() -> Dict: pass # type:ignore
    

def init_all_params(sess, var_scope=None) -> None: pass # type:ignore
    

def set_session(): pass # type:ignore
    

def get_session(): pass # type:ignore
    

def clear_session(): pass # type:ignore
    

@contextlib.contextmanager
def variable_scope(name, *args, **kwargs): pass


@contextlib.contextmanager
def device_scope(name, *args, **kwargs): pass

@contextlib.contextmanager
def session_scope(name, *args, **kwargs): pass

