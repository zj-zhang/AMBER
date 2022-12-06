import torch
import numpy as np
from .cache import *

def data_type_dict():
    return { 'float16' : torch.float16, 'float32' : torch.float32, 'int8': torch.int8, 'int16': torch.int16, 'int32': torch.int32, 'bool': torch.bool}

TensorType = torch.Tensor

def Variable(initial_value, shape=None, dtype=None, trainable=True, name=None):
    dtype = dtype or torch.float32
    if np.isscalar(initial_value) and shape is not None:
        tensor = torch.tensor(initial_value * np.ones(shape), requires_grad=trainable, dtype=dtype)
    elif type(initial_value) is TensorType:
        tensor = initial_value
    else:
        tensor = torch.tensor(initial_value, requires_grad=trainable, dtype=dtype)
    return tensor

def to_numpy(tensor):
    if type(tensor) is TensorType:
        return tensor.detach().cpu().numpy()
    elif hasattr(tensor, '__iter__'):
        return [to_numpy(t) for t in tensor]
    else:
        raise TypeError(f"unknown type for {tensor}")

def create_parameter(name, shape, dtype=None, initializer=None, trainable=True, seed=None):
    dtype = dtype or torch.float32
    param = torch.nn.Parameter(
        data=torch.zeros(shape, dtype=dtype), 
        requires_grad=trainable,
    )
    if initializer is None:
         torch.nn.init.uniform_(param, -0.1, 0.1)
    elif type(initializer) is str:
        if initializer == 'zeros':
            torch.nn.init.constant_(param, 0)
        elif initializer == 'he_normal':
            torch.nn.init.xavier_normal_(param)
        elif initializer == 'uniform':
            torch.nn.init.uniform_(param, -0.1, 0.1)
        else:
            raise Exception('str initializer not understood')
    device=GLOBAL_DEFAULT_GRAPH.get_device()
    if device != 'cpu':
        param = param.to(device)
    GLOBAL_DEFAULT_GRAPH.add_param(name=name, param=param)
    return param

def get_params_name(x):
    return x.name

def get_shape(x):
    return np.array([x for x in x.size()])

shape = get_shape

def one_hot(tensor, num_classes=-1):
    if type(tensor) is not TensorType:
        tensor = torch.tensor(tensor, dtype=torch.long)
    return torch.nn.functional.one_hot(tensor, num_classes)

def assign(x, y):
    assert type(x) is TensorType
    if type(y) is TensorType:
        x.data = y.data
    else:
        x.data = y

def concat(x, axis=-1):
    return torch.concat(x, dim=axis)

def reshape(x, shape):
    return torch.reshape(x, shape)

def squeeze(x, axis=None):
    return torch.squeeze(x, dim=axis)

def expand_dims(x, axis=-1):
    return torch.unsqueeze(x, dim=axis)

def split(x, num_splits, axis):
    split_size = x.size(dim=axis) // num_splits
    return torch.split(x, split_size_or_sections=split_size, dim=axis)

def where(condition, x=None, y=None, name=None):
    return torch.where(condition=condition, x=x, y=y)

def cast(x, dtype):
    if type(x) is TensorType:
        return x.type(dtype)
    else:
        return torch.tensor(x, dtype=dtype)

def fill(dims, value):
    return torch.fill(dims, value)

def ones(shape, dtype=None):
    dtype = dtype or torch.float32
    return torch.ones(shape, dtype=dtype)

def zeros(shape, dtype=None):
    dtype = dtype or torch.float32
    if np.isscalar(shape):
        shape = (shape)
    return torch.zeros(shape, dtype=dtype)

def zeros_like(input):
    return torch.zeros_like(input)

def stop_gradient(input):
    return input.detach()

def transpose(x, perm=None):
    if perm is None:
        perm = np.arange(len(get_shape(x))).tolist()[::-1]
    return torch.permute(x, dims=perm)

def stack(x):
    return torch.stack(x)

def embedding_lookup(params, ids):
    return params[cast(ids, torch.long)]
