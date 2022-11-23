import tensorflow as tf

# get_tf_layer
def get_layer(fn_str):
    fn_str = fn_str.lower()
    if fn_str == "relu":
        return tf.nn.relu
    elif fn_str == "linear":
        return lambda x: x
    elif fn_str == "softmax":
        return tf.nn.softmax
    elif fn_str == "sigmoid":
        return tf.nn.sigmoid
    elif fn_str == 'leaky_relu':
        return tf.nn.leaky_relu
    elif fn_str == 'elu':
        return tf.nn.elu
    elif fn_str == 'tanh':
        return tf.nn.tanh
    else:
        raise Exception("cannot get tensorflow layer for: %s" % fn_str)
