"""GPU session and device management
"""

import tensorflow as tf

session_cache = {}

def init_all_params(sess, var_scope=None):
    var_scope = var_scope or ''
    vars = [v for v in tf.all_variables() if v.name.startswith(var_scope)]
    sess.run(tf.initialize_variables(vars))

def variable_scope(name, *args, **kwargs):
    return tf.variable_scope(name_or_scope=name, *args, **kwargs)

