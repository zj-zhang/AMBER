import tensorflow as tf


def create_weight(name, shape, initializer=None, trainable=True, seed=None):
    if initializer is None:
        try:
            initializer = tf.keras.initializers.he_normal(seed=seed)
        except AttributeError:
            initializer = tf.initializers.he_normal(seed=seed)
    return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)


def create_bias(name, shape, initializer=None):
    if initializer is None:
        initializer = tf.constant_initializer(0.0)
    return tf.get_variable(name, shape, initializer=initializer)


# get_tf_loss
def get_loss(loss, y_true, y_pred):
    if type(loss) is str:
        loss = loss.lower()
        if loss == 'mse' or loss == 'mean_squared_error':
            loss_ = tf.reduce_mean(tf.square(y_true - y_pred))
        elif loss == 'categorical_crossentropy':
            loss_ = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
        elif loss == 'binary_crossentropy':
            loss_ = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        else:
            raise Exception("cannot understand string loss: %s" % loss)
    elif type(loss) is callable:
        loss_ = loss(y_true, y_pred)
    else:
        raise TypeError("Expect loss argument to be str or callable, got %s" % type(loss))
    return loss_

# get_tf_metrics
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
    else:
        raise Exception("cannot understand metric type: %s" % m)
