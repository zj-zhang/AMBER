import tensorflow as tf

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


def get_train_op(loss, variables, optimizer, **kwargs):
    assert tf.keras.backend.backend() == 'tensorflow'
    # TODO: change to TF.keras
    from keras.optimizers import get as get_opt
    grads = tf.gradients(loss, variables)
    grad_var = []
    no_grad_var = []
    for g, v in zip(grads, variables):
        if g is None:
            # get sub-scope name; if is optimizer-related, ignore
            if 'compile' in v.name.split('/'):
                continue
            no_grad_var.append(v)
        else:
            grad_var.append(v)
    # if no_grad_var:
    #    warnings.warn(
    #        "\n" + "=" * 80 + "\nWarning: the following tf.variables have no gradients"
    #                   " and have been discarded: \n %s" % no_grad_var, stacklevel=2)
    opt = get_opt(optimizer)
    train_op = opt.get_updates(loss, grad_var) # type: ignore
    try:
        config = opt.get_config() # type: ignore
    except NotImplementedError:  # if cannot get learning-rate when eager-execution is disableed
        config = {'lr':None}
    try:
        learning_rate = config['lr']
    except:  # for newer version of keras
        learning_rate = config['learning_rate']
    return train_op, learning_rate, opt


# alias
Model = tf.keras.models.Model
