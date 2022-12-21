"""GPU session and device management
"""

import tensorflow as tf
from tensorflow.test import TestCase
import gc
from . import cache

session_cache = {}
SessionType = tf.compat.v1.Session

def Session():
    sess = tf.compat.v1.Session()
    session_cache[str(sess)] = sess
    return sess

def set_session(sess):
    tf.compat.v1.keras.backend.set_session(sess)

def clear_session():
    tf.keras.backend.clear_session()
    gc.collect()

def get_session():
    return tf.keras.backend.get_session()

def init_all_params(sess, var_scope=None):
    var_scope = var_scope or ''
    vars = [v for v in tf.all_variables() if v.name.startswith(var_scope)]
    sess.run(tf.initialize_variables(vars))

# def variable_scope(name, *args, **kwargs):
#     reuse = kwargs.pop("reuse", tf.compat.v1.AUTO_REUSE)
#     return tf.compat.v1.variable_scope(name_or_scope=name, reuse=reuse, *args, **kwargs)

class variable_scope:
    def __init__(self, name, *args, **kwargs):
        self.name = name
    def __enter__(self):
        cache.CURRENT_GRAPH.append_var_scope(self.name)
    def __exit__(self, *args):
        cache.CURRENT_GRAPH.strip_var_scope()
 

def device_scope(name, *args, **kwargs):
    return tf.device(name, *args, **kwargs)

def session_scope(session, *args, **kwargs):
    return session.as_default()
