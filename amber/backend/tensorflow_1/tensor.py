import tensorflow as tf


def data_type_dict():
    return { 'float16' : tf.float16, 'float32' : tf.float32, 'int8': tf.int8, 'int16': tf.int16, 'int32': tf.int32}


def create_parameter(name, shape, dtype=tf.float32, initializer=None, trainable=True, seed=None):
    if initializer is None:
        try:
            initializer = tf.keras.initializers.he_normal(seed=seed)
        except AttributeError:
            initializer = tf.initializers.he_normal(seed=seed)
    elif type(initializer) is str:
        initializer = tf.constant_initializer(0.0) if initializer == 'zeros' else \
                      tf.initializers.he_normal(seed=seed) if initializer == 'he_normal' else \
                      tf.initializers.uniform(-0.1, 0.1)
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, dtype=dtype)


def concat(x, axis=-1):
    return tf.concat(x, axis=axis)

def reshape(x, shape):
    return tf.reshape(x, shape)

def squeeze(x, axis=None):
    return tf.squeeze(x, axis=axis)

def split(x, num_splits, axis):
    return tf.split(x, num_or_size_splits=num_splits, axis=axis)