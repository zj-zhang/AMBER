import tensorflow as tf


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
    train_op = opt.get_updates(loss, grad_var)
    try:
        config = opt.get_config()
    except NotImplementedError:  # if cannot get learning-rate when eager-execution is disableed
        config = {'lr':None}
    try:
        learning_rate = config['lr']
    except:  # for newer version of keras
        learning_rate = config['learning_rate']
    return train_op, learning_rate, opt