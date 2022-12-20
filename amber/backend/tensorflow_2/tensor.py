import tensorflow as tf
import numpy as np
from .cache import CURRENT_GRAPH, GLOBAL_DEFAULT_GRAPH

def data_type_dict():
    return { 'float16' : tf.float16, 'float32' : tf.float32, 'int8': tf.int8, 'int16': tf.int16, 'int32': tf.int32, 'bool': tf.bool}

TensorType = tf.Tensor

def Variable(initial_value, shape=None, dtype=None, trainable=True, name=None):
    dtype = dtype or tf.float32
    if np.isscalar(initial_value) and shape is not None:
        tensor = tf.Variable(initial_value * np.ones(shape), trainable=trainable, dtype=dtype)
    elif type(initial_value) is TensorType:
        tensor = initial_value
    else:
        return tf.Variable(initial_value=initial_value, dtype=dtype, shape=shape, trainable=trainable, name=name)

def placeholder(shape, dtype=None, name=None):
    dtype = dtype or tf.float32
    return tf.compat.v1.placeholder(shape=shape, dtype=dtype, name=name)

def create_parameter(name, shape, dtype=tf.float32, initializer=None, trainable=True, seed=None):
    if initializer is None:
        initializer = tf.keras.initializers.HeNormal(seed=seed)
    elif type(initializer) is str:
        initializer = tf.keras.initializers.Constant(0.0) if initializer == 'zeros' else \
                      tf.keras.initializers.HeNormal(seed=seed) if initializer == 'he_normal' else \
                      tf.keras.initializers.RandomUniform(-0.1, 0.1, seed=seed)
    param = tf.Variable(initial_value=initializer(shape=shape, dtype=dtype), name=name, trainable=trainable, dtype=dtype)
    GLOBAL_DEFAULT_GRAPH.add_param(name=name, param=param)
    return param

def get_params_name(x):
    return x.name

def get_shape(x):
    return np.array([x for x in x.get_shape()])

shape = get_shape

def concat(x, axis=-1):
    return tf.concat(x, axis=axis)

def reshape(x, shape):
    return tf.reshape(x, shape)

def squeeze(x, axis=None):
    # XXX: change this to check if shape[axis]==1
    try:
        return tf.squeeze(x, axis=axis)
    except:
        return x

def expand_dims(x, axis=-1):
    return tf.expand_dims(x, axis)

def split(x, num_splits, axis):
    return tf.split(x, num_or_size_splits=num_splits, axis=axis)

def where(condition, x=None, y=None, name=None):
    return tf.where(condition=condition, x=x, y=y, name=name)

def cast(x, dtype):
    return tf.cast(x, dtype)

def fill(dims, value):
    return tf.fill(dims, value)

def map_fn(f, x):
    return tf.map_fn(f, x)

def ones(shape, dtype=None):
    dtype = dtype or tf.float32
    return tf.ones(shape, dtype=dtype)

def zeros(shape, dtype=None):
    dtype = dtype or tf.float32
    return tf.zeros(shape, dtype=dtype)

def zeros_like(input):
    return tf.zeros_like(input)

def embedding_lookup(params, ids):
    return tf.nn.embedding_lookup(params=params, ids=ids)

def cond(pred, true_fn=None, false_fn=None, name=None):
    return tf.cond(pred=pred, true_fn=true_fn, false_fn=false_fn, name=name)

def stop_gradient(input):
    return tf.stop_gradient(input)

def transpose(x, perm=None):
    return tf.transpose(x, perm=perm)

def stack(x):
    return tf.stack(x)

def to_numpy(tensor):
    if issubclass(type(tensor), TensorType):
        if tf.executing_eagerly():
            return tensor.numpy()
        else:
            sess = tf.compat.v1.keras.backend.get_session()
            return sess.run(tensor)
    elif hasattr(tensor, '__iter__'):
        return [to_numpy(t) for t in tensor]
    else:
        raise TypeError(f"unknown type for {tensor}")

def one_hot(tensor, num_classes=-1):
    if type(tensor) is not TensorType:
        tensor = tf.Variable(tensor, dtype=tf.int32, trainable=False)
    return tf.one_hot(indices=tensor, depth=num_classes)