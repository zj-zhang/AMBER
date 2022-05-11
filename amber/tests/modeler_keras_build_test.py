"""Test Keras modeler"""

import tensorflow as tf
import numpy as np
from parameterized import parameterized
from amber.utils import testing_utils
from amber import modeler
from amber import architect
import logging, sys
logging.disable(sys.maxsize)


class TestKerasBuilder(testing_utils.TestCase):
    def setUp(self):
        self.model_space, _ = testing_utils.get_example_conv1d_space(num_layers=2)
        self.target_arc = [0, 0, 1]
        self.input_op = architect.Operation('input', shape=(10, 4), name="input")
        self.output_op = architect.Operation('dense', units=1, activation='sigmoid', name="output")
        self.model_compile_dict = {'loss': 'binary_crossentropy', 'optimizer': 'sgd'}
        self.x = np.random.choice(2, 40).reshape((1, 10, 4))
        self.y = np.random.sample(1).reshape((1, 1))
        self.modeler = modeler.KerasResidualCnnBuilder(
            inputs_op=self.input_op,
            output_op=self.output_op,
            model_space=self.model_space,
            fc_units=5,
            flatten_mode='flatten',
            model_compile_dict=self.model_compile_dict
        )

    def test_get_model(self):
        model = self.modeler(self.target_arc)
        old_loss = model.evaluate(self.x, self.y)
        model.fit(self.x, self.y, batch_size=1, epochs=100, verbose=0)
        new_loss = model.evaluate(self.x, self.y)
        self.assertLess(new_loss, old_loss)


class TestKerasGetLayer(testing_utils.TestCase):

    @parameterized.expand([
        # fc
        ((100,), 'dense', {'units': 4}, tf.Tensor),
        ((100,), 'dense', {'units': 4}, tf.Tensor, None, {'with_bn': True}),
        ((100,), 'identity', {}, tf.Tensor),
        # conv
        ((100, 4), 'conv1d', {'filters': 5, 'kernel_size': 8}, tf.Tensor),
        ((100, 4), 'conv1d', {'filters': 5, 'kernel_size': 8}, tf.Tensor, None, {'with_bn': True}),
        ((100, 4), 'denovo', {'filters': 5, 'kernel_size': 8}, tf.Tensor),
        ((100, 4), 'maxpool1d', {'pool_size': 4, 'strides': 4}, tf.Tensor),
        ((100, 4), 'avgpool1d', {'pool_size': 4, 'strides': 4}, tf.Tensor),
        ((100, 4, 4), 'conv2d', {'filters': 5, 'kernel_size': (2, 2)}, tf.Tensor),
        ((100, 4, 4), 'conv2d', {'filters': 5, 'kernel_size': (2, 2)}, tf.Tensor, None, {'with_bn': True}),
        ((100, 4, 4), 'maxpool2d', {'pool_size': (2, 2)}, tf.Tensor),
        ((100, 4, 4), 'avgpool2d', {'pool_size': (2, 2)}, tf.Tensor),
        ((100, 4), 'lstm', {'units': 2}, tf.Tensor),
        # reshape
        ((100, 4), 'flatten', {}, tf.Tensor),
        ((100, 4), 'globalavgpool1d', {}, tf.Tensor),
        ((100, 4), 'globalmaxpool1d', {}, tf.Tensor),
        ((100, 4, 4), 'globalavgpool2d', {}, tf.Tensor),
        ((100, 4, 4), 'globalmaxpool2d', {}, tf.Tensor),
        ((100, 4), 'sfc', {'output_dim': 100, 'symmetric': False}, tf.Tensor),
        # regularizer
        ((100,), 'dropout', {'rate': 0.3}, tf.Tensor),
        ((100, 4), 'dropout', {'rate': 0.3}, tf.Tensor),
        ((100,), 'sparsek_vec', {}, tf.Tensor),
        ((100,), 'gaussian_noise', {'stddev': 1}, tf.Tensor),

    ])
    def test_get_layers(self, input_shape, layer_name, layer_attr, exp_type=None, exp_error=None, kwargs=None):
        kwargs = kwargs or {}
        x = modeler.dag.get_layer(x=None, state=architect.Operation('Input', shape=input_shape))
        operation = architect.Operation(layer_name, **layer_attr)
        if exp_error is None:
            layer = modeler.dag.get_layer(x=x, state=operation, **kwargs)
            self.assertIsInstance(layer, exp_type)
        else:
            self.assertRaises(exp_error, modeler.dag.get_layer, x=x, state=operation, **kwargs)

    @parameterized.expand([
        # undef should throw error, or return None
        ((100,), architect.Operation('conv1d', filters=5, kernel_size=8), ValueError),
        ((100,), architect.Operation('conv4d', filters=5, kernel_size=8), ValueError),
        ((100,), architect.Operation('concatenate'), ValueError),
    ])
    def test_get_layers_catch_exception(self, input_shape, operation, exp_error):
        x = modeler.dag.get_layer(x=None, state=architect.Operation('Input', shape=input_shape))
        self.assertRaises(exp_error, modeler.dag.get_layer, x=x, state=operation)


if __name__ == '__main__':
    tf.test.main()
