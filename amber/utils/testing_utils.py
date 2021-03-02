"""
Testing utilities for amber
"""

import tensorflow as tf
try:
    import keras
except ImportError:
    from tensorflow import keras as keras
from .. import architect


class TestCase(tf.test.TestCase):
    def tearDown(self):
        tf.keras.backend.clear_session()
        keras.backend.clear_session()
        super(TestCase, self).tearDown()


def get_example_conv1d_space(out_filters=8, num_layers=2):
    model_space = architect.ModelSpace()
    num_pool = 1
    expand_layers = [num_layers//k-1 for k in range(1, num_pool)]
    layer_sharing = {}
    for i in range(num_layers):
        model_space.add_layer(i, [
            architect.Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
            architect.Operation('maxpool1d', filters=out_filters, pool_size=4, strides=1),
            architect.Operation('identity', filters=out_filters)
      ])
        if i in expand_layers:
            out_filters *= 2
        if i > 0:
            layer_sharing[i] = 0
    return model_space, layer_sharing

