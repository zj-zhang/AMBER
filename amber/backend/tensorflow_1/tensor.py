import tensorflow as tf
import numpy as np


def data_type_dict():
    return { 'float16' : tf.float16, 'float32' : tf.float32, 'int8': tf.int8, 'int16': tf.int16, 'int32': tf.int32, 'bool': tf.bool}

TensorType = tf.Tensor

def Variable(initial_value, shape, dtype=None, trainable=True, name=None):
    dtype = dtype or tf.float32
    return tf.Variable(initial_value=initial_value, dtype=dtype, shape=shape, trainable=trainable, name=name)

def placeholder(shape, dtype=None, name=None):
    dtype = dtype or tf.float32
    return tf.placeholder(shape=shape, dtype=dtype, name=name)

def get_param_name(var):
    return var.name

def one_hot(tensor, num_classes=-1):
    return tf.one_hot(indices=tensor, depth=num_classes)


def create_parameter(name, shape, dtype=tf.float32, initializer=None, trainable=True, seed=None):
    if initializer is None:
        try:
            initializer = tf.keras.initializers.he_normal(seed=seed)
        except AttributeError:
            initializer = tf.initializers.he_normal(seed=seed)
    elif type(initializer) is str:
        initializer = tf.constant_initializer(0.0) if initializer == 'zeros' else \
                      tf.initializers.he_normal(seed=seed) if initializer == 'he_normal' else \
                      tf.random_uniform_initializer(-0.1, 0.1) if initializer == 'uniform' else \
                        Exception('str initializer not understood')
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, dtype=dtype)

def get_params_name(x):
    return x.name

def get_shape(x):
    return np.array([x.value for x in x.get_shape()])

def shape(x):
    return tf.shape(x)

def assign(x, y):
    return tf.assign(x, y)

def concat(x, axis=-1):
    return tf.concat(x, axis=axis)

def reshape(x, shape):
    return tf.reshape(x, shape)

def squeeze(x, axis=None):
    return tf.squeeze(x, axis=axis)

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

