import tensorflow as tf

def abs(x):
    return tf.abs(x)

def matmul(a,b):
    return tf.matmul(a,b)

def reduce_sum(x, axis=-1):
    if x.dtype == tf.bool:
        x = tf.cast(x, tf.int32)
    return tf.reduce_sum(x, axis=axis)

def reduce_mean(x, axis=-1):
    if x.dtype == tf.bool:
        x = tf.cast(x, tf.int32)
    return tf.reduce_mean(x, axis=axis)

def log(x):
    return tf.log(x)

def exp(x):
    return tf.exp(x)

def sigmoid(x):
    return tf.sigmoid(x)

def tanh(x):
    return tf.tanh(x)

def minimum(a,b):
    return tf.minimum(a,b)

def maximum(a,b):
    return tf.maximum(a,b)

def clip_by_value(x, clip_value_min, clip_value_max):
    return tf.clip_by_value(x, clip_value_min, clip_value_max)