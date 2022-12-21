import tensorflow as tf
from ._operators import Layer_deNovo, SeparableFC, sparsek_vec
from . import cache

def get_layer(x=None, op=None, custom_objects=None, with_bn=False):
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
        layer = op()
    elif op.Layer_type == 'activation':
        layer = tf.keras.layers.Activation(**op.Layer_attributes)
    elif op.Layer_type == 'dense':
        if with_bn is True:
            assert x is not None
            actv_fn = op.Layer_attributes.pop('activation', 'linear')
            x = tf.keras.layers.Dense(**op.Layer_attributes)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(actv_fn)(x)
            return x
        else:
            layer = tf.keras.layers.Dense(**op.Layer_attributes)

    elif op.Layer_type == 'sfc':
        layer = SeparableFC(**op.Layer_attributes)

    elif op.Layer_type == 'input':
        assert x is None
        return tf.keras.layers.Input(**op.Layer_attributes)

    elif op.Layer_type == 'conv1d':
        if with_bn is True:
            assert x is not None
            actv_fn = op.Layer_attributes.pop('activation', 'linear')
            x = tf.keras.layers.Conv1D(**op.Layer_attributes)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(actv_fn)(x)
            return x
        else:
            layer = tf.keras.layers.Conv1D(**op.Layer_attributes)

    elif op.Layer_type == 'conv2d':
        if with_bn is True:
            assert x is not None
            actv_fn = op.Layer_attributes.pop('activation', 'linear')
            x = tf.keras.layers.Conv2D(**op.Layer_attributes)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation(actv_fn)(x)
            return x
        else:
            layer = tf.keras.layers.Conv2D(**op.Layer_attributes)

    elif op.Layer_type == 'denovo':
        x = tf.keras.layers.Lambda(lambda t: tf.expand_dims(t, axis=-1))(x)
        x = tf.keras.layers.Permute(dims=(2, 1, 3))(x)
        x = Layer_deNovo(**op.Layer_attributes)(x)
        x = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=1))(x)
        return x
    
    elif op.Layer_type == 'batchnorm' or op.Layer_type == 'BatchNormalization'.lower():
        return tf.keras.layers.BatchNormalization()(x, training=op.Layer_attributes.get("training", True))

    elif op.Layer_type == 'sparsek_vec':
        layer = tf.keras.layers.Lambda(sparsek_vec, **op.Layer_attributes)

    elif op.Layer_type in ('maxpool1d','maxpooling1d'):
        layer = tf.keras.layers.MaxPooling1D(pool_size=op.Layer_attributes.get('pool_size',2), strides=op.Layer_attributes.get('strides', None), padding=op.Layer_attributes.get('padding', "valid"), data_format=op.Layer_attributes.get('data_format', 'channels_last'))

    elif op.Layer_type in ('maxpool2d','maxpooling2d'):
        layer = tf.keras.layers.MaxPooling2D(**op.Layer_attributes)

    elif op.Layer_type in ('avgpool1d', 'avgpooling1d', 'averagepooling1d'):
        layer = tf.keras.layers.AveragePooling1D(pool_size=op.Layer_attributes.get('pool_size',2), strides=op.Layer_attributes.get('strides', None), padding=op.Layer_attributes.get('padding', "valid"), data_format=op.Layer_attributes.get('data_format', 'channels_last'))

    elif op.Layer_type in ('avgpool2d', 'avgpooling2d', 'averagepooling2d'):
        layer = tf.keras.layers.AveragePooling2D(**op.Layer_attributes)

    elif op.Layer_type == 'lstm':
        layer = tf.keras.layers.LSTM(**op.Layer_attributes)

    elif op.Layer_type == 'flatten':
        layer = tf.keras.layers.Flatten()

    elif op.Layer_type in ('globalavgpool1d', 'GlobalAveragePooling1D'.lower()):
        layer = tf.keras.layers.GlobalAveragePooling1D(**op.Layer_attributes)

    elif op.Layer_type in ('globalavgpool2d', 'GlobalAveragePooling2D'.lower()):
        layer = tf.keras.layers.GlobalAveragePooling2D(**op.Layer_attributes)

    elif op.Layer_type in ('globalmaxpool1d', 'GlobalMaxPooling1D'.lower()):
        layer = tf.keras.layers.GlobalMaxPooling1D(**op.Layer_attributes)

    elif op.Layer_type in ('globalmaxpool2d', 'GlobalMaxPooling2D'.lower()):
        layer =  tf.keras.layers.GlobalMaxPooling2D(**op.Layer_attributes)

    elif op.Layer_type == 'dropout':
        return tf.keras.layers.Dropout(
            rate=op.Layer_attributes.get('rate'), 
            noise_shape=op.Layer_attributes.get('noise_shape', None), 
            seed=op.Layer_attributes.get('seed', None))(x, training=op.Layer_attributes.get("training", True))

    elif op.Layer_type == 'identity':
        layer = tf.keras.layers.Lambda(lambda t: t, name=op.Layer_attributes.get('name', None))

    elif op.Layer_type == 'gaussian_noise':
        layer = tf.keras.layers.GaussianNoise(**op.Layer_attributes)

    elif op.Layer_type == 'concatenate':
        layer = tf.keras.layers.Concatenate(**op.Layer_attributes)
     
    elif op.Layer_type in custom_objects:
        return custom_objects[op.Layer_type](**op.Layer_attributes)(x)

    else:
        raise ValueError('Layer_type "%s" is not understood' % op.Layer_type)

    return layer(x) if x is not None else layer


def trainable_variables(scope: str):
    #return tf.compat.v1.trainable_variables(scope=scope)
    scope = scope or ''
    if isinstance(scope, tf.Module):
        return scope.trainable_variables()
    elif type(scope) is str:
        return [param for name, param in cache.CURRENT_GRAPH.param_cache.items() if scope in name]


def get_loss(loss, y_true, y_pred, reduction='mean'):
    assert reduction in ('mean', 'none')
    if type(loss) is str:
        loss = loss.lower()
        if loss == 'mse' or loss == 'mean_squared_error':
            loss_ = tf.square(y_true - y_pred)
        elif loss == 'categorical_crossentropy':
            loss_ = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        elif loss == 'binary_crossentropy':
            loss_ = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        elif loss == 'nllloss_with_logits':
            loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        else:
            raise Exception("cannot understand string loss: %s" % loss)
    elif callable(loss):
        loss_ = loss(y_true, y_pred)
    else:
        raise TypeError("Expect loss argument to be str or callable, got %s" % type(loss))
    if reduction == 'mean':
        loss_ = tf.reduce_mean(loss_)
    return loss_


def get_metric(m):
    if callable(m):
        return m
    elif m.lower() == 'mae':
        return tf.keras.metrics.MAE
    elif m.lower() == 'mse':
        return tf.keras.metrics.MSE
    elif m.lower() == 'acc':
        def acc(y_true, y_pred):
            return tf.reduce_mean(y_true)
        # return tf.keras.metrics.Accuracy
        return acc
    elif m.lower() == 'auc':
        return tf.keras.metrics.AUC
    elif m.lower() == 'kl_div':
        return tf.keras.metrics.kullback_leibler_divergence
    else:
        raise Exception("cannot understand metric type: %s" % m)

def get_optimizer(opt, parameters, opt_config=None):
    opt_config = opt_config or {'learning_rate':0.01}
    if callable(opt):
        opt_ = opt
    elif type(opt) is str:
        if opt.lower() == 'adam':
            opt_ = tf.keras.optimizers.Adam
        elif opt.lower() == 'sgd':
            opt_ = tf.keras.optimizers.SGD
        else:
            raise Exception(f"unknown opt {opt}")
    else:
        raise Exception(f"unknown opt {opt}")
    return opt_(**opt_config)


def get_train_op(loss, variables, optimizer, tape=None):
    tape = tf.GradientTape() if tape is None else tape
    optimizer.minimize(loss, variables, tape=tape)


def get_callback(m):
    if callable(m):
        return m
    elif m == 'EarlyStopping':
        return tf.keras.callbacks.EarlyStopping
    elif m == 'ModelCheckpoint':
        return tf.keras.callbacks.ModelCheckpoint
    elif m == 'CallbackList':
        from keras.callbacks import CallbackList
        return CallbackList
    elif m == 'BaseLogger':
        return tf.keras.callbacks.BaseLogger
    elif m == 'History':
        return tf.keras.callbacks.History
    else:
        raise Exception("cannot understand callback: %s" % m)

# alias
Model = tf.keras.models.Model
