import tensorflow as tf

def get_math_func(n):
    if callable(n):
        return n
    elif type(n) is str:
        if n.lower() == 'relu':
            return tf.nn.relu
        elif n.lower() == 'sigmoid':
            return tf.sigmoid
        elif n.lower() == 'tanh':
            return tf.tanh
        elif n.lower() == 'softmax':
            return tf.nn.softmax
        elif n.lower() == 'linear':
            return lambda x:x
        elif n.lower() == 'dropout':
            return tf.nn.dropout
        else:
            raise ValueError("unknown string math func: %s" % n)
    else:
        raise TypeError(f"unknown type math func: {n}")

def abs(x):
    return tf.abs(x)

def matmul(x, y):
    return tf.matmul(x, y)

def reduce_sum(x, axis=None, keepdim=False):
    return tf.reduce_sum(x, axis=axis, keepdims=keepdim)

def reduce_mean(x, axis=None):
    return tf.reduce_mean(x, axis=axis)

def log(x):
    return tf.log(x)

def exp(x):
    return tf.exp(x)

def sigmoid(x):
    return tf.sigmoid(x)

def tanh(x):
    return tf.tanh(x)

def softmax(x, axis=None):
    return tf.nn.softmax(x, axis=axis)

def minimum(x,y):
    return tf.minimum(x, y)

def maximum(x,y):
    return tf.maximum(x, y)

def clip_by_value(x, clip_value_min, clip_value_max):
    return tf.clip_by_value(x, clip_value_min, clip_value_max)

def relu(x):
    return tf.nn.relu(x)

def boolean_mask(tensor, mask, axis=None, name='boolean_mask'):
    return tf.boolean_mask(tensor=tensor, mask=mask, axis=None, name='boolean_mask')

def logical_and(x, y):
    return tf.logical_and(x, y)

def range(start, limit, delta=1, dtype=None, name='range'):
    return tf.range(start, limit=limit, delta=delta, dtype=dtype, name=name)

def less(x,y):
    return tf.less(x, y)

def greater(x,y):
    return tf.greater(x,y)

def equal(x, y):
    return tf.equal(x, y)

def pow(x, y):
    return tf.math.pow(x, y)

def case(pred_fn_pairs, default=None, exclusive=False, strict=False, name='case'):
    return tf.case(pred_fn_pairs=pred_fn_pairs, default=default, exclusive=exclusive, strict=strict, name=name)

def tensordot(a, b, axes, name=None):
    return tf.tensordot(a, b, axes=axes, name=name)

def multinomial(logits, num_samples):
    return tf.multinomial(logits=logits, num_samples=num_samples)
