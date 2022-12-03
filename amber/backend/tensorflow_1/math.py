import tensorflow as tf

def abs(x):
    return tf.abs(x)

def matmul(x, y):
    return tf.matmul(x, y)

def reduce_sum(x, axis=None):
    return tf.reduce_sum(x, axis=axis)

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

def equal(x, y):
    return tf.equal(x, y)

def pow(x, y):
    return tf.math.pow(x, y)

def case(pred_fn_pairs, default=None, exclusive=False, strict=False, name='case'):
    return tf.case(pred_fn_pairs=pred_fn_pairs, default=default, exclusive=exclusive, strict=strict, name=name)



