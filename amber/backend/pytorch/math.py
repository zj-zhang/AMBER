import torch

def abs(x):
    return torch.abs(x)

def matmul(x, y):
    return torch.matmul(x, y)

def reduce_sum(x, axis=None, keepdim=False):
    if type(x) in (list, tuple):
        x = torch.concat(x)
    if axis is None:
        return torch.sum(x)
    else:
        return torch.sum(x, dim=axis, keepdim=keepdim)

def reduce_mean(x, axis=None):
    if type(x) in (list, tuple):
        x = torch.concat(x)
    if axis is None:
        return torch.mean(x)
    else:
        return torch.mean(x, dim=axis)

def log(x):
    return torch.log(x)

def exp(x):
    return torch.exp(x)

def sigmoid(x):
    return torch.sigmoid(x)

def tanh(x):
    return torch.tanh(x)

def softmax(x, axis=None):
    axis = axis or -1
    return torch.softmax(x, dim=axis)

def minimum(x,y):
    return torch.minimum(x, y)

def maximum(x,y):
    return torch.maximum(x, y)

def clip_by_value(x, clip_value_min, clip_value_max):
    return torch.clip(x, clip_value_min, clip_value_max)

def relu(x):
    return torch.relu(x)

def boolean_mask(tensor, mask, axis=None, name='boolean_mask'):
    assert axis is None, "torch boolean_mask/masked_select does not support axis"
    return torch.masked_select(input=tensor, mask=mask)

def logical_and(x, y):
    return torch.logical_and(x, y)

def range(start, limit, delta=1, dtype=None, name='range'):
    return torch.range(start, end=limit, step=delta, dtype=dtype)

def less(x,y):
    return torch.less(x, y)

def greater(x,y):
    return torch.greater(x,y)

def equal(x, y):
    return torch.equal(x, y)

def pow(x, y):
    return torch.pow(x, y)

def tensordot(a, b, axes, name=None):
    return torch.tensordot(a, b, dims=axes)

def multinomial(logits, num_samples):
    return torch.multinomial(input=softmax(logits), num_samples=num_samples)
