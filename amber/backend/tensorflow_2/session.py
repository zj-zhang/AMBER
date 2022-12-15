"""GPU session and device management
"""

import tensorflow as tf

session_cache = {}

def Session():
    sess = tf.compat.v1.Session()
    session_cache[str(sess)] = sess
    return sess

def init_all_params(sess, var_scope=None):
    var_scope = var_scope or ''
    vars = [v for v in tf.compat.v1.all_variables() if v.name.startswith(var_scope)]
    sess.run(tf.compat.v1.initialize_variables(vars))

def variable_scope(name, *args, **kwargs):
    reuse = kwargs.pop("reuse", tf.compat.v1.AUTO_REUSE)
    return tf.compat.v1.variable_scope(name_or_scope=name, reuse=reuse, *args, **kwargs)

