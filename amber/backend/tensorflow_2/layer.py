import tensorflow as tf
from ._operators import Layer_deNovo, SeparableFC, sparsek_vec


def get_layer(x, op, custom_objects=None, with_bn=False):
    """Getter method for a Keras layer, including native Keras implementation and custom layers that are not included in
    Keras.

    Parameters
    ----------
    x : tf.keras.layers or None
        The input Keras layer
    op : amber.architect.Operation, or callable
        The target layer to be built
    custom_objects : dict, or None
        Allow stringify custom objects by parsing a str->class dict
    with_bn : bool, optional
        If true, add batch normalization layers before activation

    Returns
    -------
    x : tf.keras.layers
        The built target layer connected to input x
    """
    custom_objects = custom_objects or {}
    if callable(op):
        return op()(x)
    elif op.Layer_type == 'activation':
        return tf.keras.layers.Activation(**op.Layer_attributes)(x)
    elif op.Layer_type == 'dense':
        if with_bn is True:
            actv_fn = op.Layer_attributes.pop('activation', 'linear')
            x = tf.keras.layers.Dense(**op.Layer_attributes)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(actv_fn)(x)
            return x
        else:
            return tf.keras.layers.Dense(**op.Layer_attributes)(x)

    elif op.Layer_type == 'sfc':
        return SeparableFC(**op.Layer_attributes)(x)

    elif op.Layer_type == 'input':
        return tf.keras.layers.Input(**op.Layer_attributes)

    elif op.Layer_type == 'conv1d':
        if with_bn is True:
            actv_fn = op.Layer_attributes.pop('activation', 'linear')
            x = tf.keras.layers.Conv1D(**op.Layer_attributes)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(actv_fn)(x)
            return x
        else:
            return tf.keras.layers.Conv1D(**op.Layer_attributes)(x)

    elif op.Layer_type == 'conv2d':
        if with_bn is True:
            actv_fn = op.Layer_attributes.pop('activation', 'linear')
            x = tf.keras.layers.Conv2D(**op.Layer_attributes)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(actv_fn)(x)
            return x
        else:
            return tf.keras.layers.Conv2D(**op.Layer_attributes)(x)

    elif op.Layer_type == 'denovo':
        x = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, axis=-1))(x)
        x = tf.keras.layers.Permute(dims=(2, 1, 3))(x)
        x = Layer_deNovo(**op.Layer_attributes)(x)
        x = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=1))(x)
        return x

    elif op.Layer_type == 'sparsek_vec':
        x = tf.keras.layers.Lambda(sparsek_vec, **op.Layer_attributes)(x)
        return x

    elif op.Layer_type == 'maxpool1d':
        return tf.keras.layers.MaxPooling1D(**op.Layer_attributes)(x)

    elif op.Layer_type == 'maxpool2d':
        return tf.keras.layers.MaxPooling2D(**op.Layer_attributes)(x)

    elif op.Layer_type == 'avgpool1d':
        return tf.keras.layers.AveragePooling1D(**op.Layer_attributes)(x)

    elif op.Layer_type == 'avgpool2d':
        return tf.keras.layers.AveragePooling2D(**op.Layer_attributes)(x)

    elif op.Layer_type == 'lstm':
        return tf.keras.layers.LSTM(**op.Layer_attributes)(x)

    elif op.Layer_type == 'flatten':
        return tf.keras.layers.Flatten()(x)

    elif op.Layer_type == 'globalavgpool1d':
        return tf.keras.layers.GlobalAveragePooling1D()(x)

    elif op.Layer_type == 'globalavgpool2d':
        return tf.keras.layers.GlobalAveragePooling2D()(x)

    elif op.Layer_type == 'globalmaxpool1d':
        return tf.keras.layers.GlobalMaxPooling1D()(x)

    elif op.Layer_type == 'globalmaxpool2d':
        return tf.keras.layers.GlobalMaxPooling2D()(x)

    elif op.Layer_type == 'dropout':
        return tf.keras.layers.Dropout(**op.Layer_attributes)(x)

    elif op.Layer_type == 'identity':
        return tf.keras.layers.Lambda(lambda t: t, **op.Layer_attributes)(x)

    elif op.Layer_type == 'gaussian_noise':
        return tf.keras.layers.GaussianNoise(**op.Layer_attributes)(x)

    elif op.Layer_type == 'concatenate':
        return tf.keras.layers.Concatenate(**op.Layer_attributes)(x)
    
    elif op.Layer_type in custom_objects:
        return custom_objects[op.Layer_type](**op.Layer_attributes)(x)

    else:
        raise ValueError('Layer_type "%s" is not understood' % op.Layer_type)

