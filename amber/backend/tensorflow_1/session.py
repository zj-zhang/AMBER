"""GPU session and device management
"""

import tensorflow as tf
from tensorflow.test import TestCase
import gc

session_cache = {}
SessionType = tf.Session

def Session():
    sess = tf.Session()
    session_cache[str(sess)] = sess
    return sess

def set_session(sess):
    tf.keras.backend.set_session(sess)

def clear_session():
    tf.keras.backend.clear_session()
    gc.collect()

def get_session():
    return tf.keras.backend.get_session()

def init_all_params(sess, var_scope=None):
    var_scope = var_scope or ''
    vars = [v for v in tf.all_variables() if v.name.startswith(var_scope)]
    sess.run(tf.initialize_variables(vars))

def variable_scope(name, *args, **kwargs):
    reuse = kwargs.pop("reuse", tf.AUTO_REUSE)
    return tf.variable_scope(name_or_scope=name, reuse=reuse, *args, **kwargs)


def device_scope(name, *args, **kwargs):
    return tf.device(name, *args, **kwargs)

def session_scope(session, *args, **kwargs):
    return session.as_default()
