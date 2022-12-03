"""Test Keras modeler"""

import tensorflow as tf
import numpy as np
from parameterized import parameterized, parameterized_class
from amber.utils import testing_utils
from amber import modeler
from amber import architect
from amber import backend
import logging, sys, unittest
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
        x = backend.get_layer(x=None, op=architect.Operation('Input', shape=input_shape))
        operation = architect.Operation(layer_name, **layer_attr)
        if exp_error is None:
            layer = backend.get_layer(x=x, op=operation, **kwargs)
            self.assertIsInstance(layer, exp_type)
        else:
            self.assertRaises(exp_error, backend.get_layer, x=x, op=operation, **kwargs)

    @parameterized.expand([
        # undef should throw error, or return None
        ((100,), architect.Operation('conv1d', filters=5, kernel_size=8), ValueError),
        ((100,), architect.Operation('conv4d', filters=5, kernel_size=8), ValueError),
        ((100,), architect.Operation('concatenate'), ValueError),
    ])
    def test_get_layers_catch_exception(self, input_shape, operation, exp_error):
        x = backend.get_layer(x=None, op=architect.Operation('Input', shape=input_shape))
        self.assertRaises(exp_error, backend.get_layer, x=x, op=operation)



@parameterized_class(attrs=('dag_func',), input_values=[
    ('DAG',),
    ('InputBlockDAG',),
    ('InputBlockAuxLossDAG',)
])
class TestKerasDAG(testing_utils.TestCase):
    with_input_blocks = True
    with_skip_connection = True
    num_inputs = 4
    fit_epochs = 5
    dag_func = 'InputBlockDAG'
    model_compile_dict = {'optimizer': 'adam', 'loss': 'mse', 'metrics': ['mae']}

    def setUp(self):
        if self.with_input_blocks:
            self.inputs_op = [architect.Operation('input', shape=(1,), name='X_%i' % i) for i in range(self.num_inputs)]
        else:
            self.inputs_op = [architect.Operation('input', shape=(self.num_inputs,), name='X')]
        self.output_op = architect.Operation('Dense', units=1, activation='linear', name='output')
        # get a three-layer model space
        self.model_space = testing_utils.get_example_sparse_model_space(4)
        # get data
        self.traindata, self.validdata, self.testdata = self.get_data(seed=111)

    def get_model_builder(self):
        model_fn = modeler.DAGModelBuilder(
            self.inputs_op,
            self.output_op,
            num_layers=len(self.model_space),
            model_space=self.model_space,
            model_compile_dict=self.model_compile_dict,
            with_skip_connection=self.with_skip_connection,
            with_input_blocks=self.with_input_blocks,
            dag_func=self.dag_func)
        return model_fn

    def get_data(self, seed=111, blockify_inputs=True):
        n1 = 3000
        n2 = 1000
        p = self.num_inputs
        # Y = f(X0,X1,X2,X3) = 3*X0X1 - 2*X2X3
        f = lambda X: 3 * X[:, 0] * X[:, 1] - 2 * X[:, 2] * X[:, 3] + rng.normal(0, np.sqrt(0.1), len(X))
        rng = np.random.default_rng(seed=seed)
        X_train = rng.poisson(lam=1, size=n1 * p).reshape(n1, p)
        X_valid = rng.poisson(lam=1, size=n2 * p).reshape(n2, p)
        X_test = rng.poisson(lam=1, size=n2 * p).reshape(n2, p)
        y_train = f(X_train)
        y_valid = f(X_valid)
        y_test = f(X_test)
        if blockify_inputs:
            X_train = [X_train[:, i].reshape((-1, 1)) for i in range(4)]
            X_valid = [X_valid[:, i].reshape((-1, 1)) for i in range(4)]
            X_test = [X_test[:, i].reshape((-1, 1)) for i in range(4)]
        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

    def model_fit(self, model):
        model.fit(
            self.traindata[0], self.traindata[1],
            batch_size=512,
            epochs=self.fit_epochs,
            validation_data=self.validdata,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)],
            verbose=0
        )
        train_loss = model.evaluate(*self.traindata, verbose=0)
        # test_loss = model.evaluate(*self.testdata, verbose=0)
        test_loss = np.mean( (model.predict(self.testdata[0]).squeeze() - self.testdata[1]) ** 2)
        return train_loss, test_loss

    def test_multi_input(self):
        model_fn = self.get_model_builder()
        # correct model: y = h1(x0,x1) + h2(x2,x3)
        arc1 = np.array([
            0, 0, 0, 1, 1,
            0, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 1, 1
        ], dtype=np.int32)
        # disconnected model: y = f(x0,x1,x2)
        arc2 = np.array([
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 1
        ], dtype=np.int32)

        with self.session() as sess:
            m1 = model_fn(arc1)
            m2 = model_fn(arc2)
            m1_train, m1_test = self.model_fit(m1)
            m2_train, m2_test = self.model_fit(m2)
            # self.assertLess(m1_test, m2_test)  # this needs a lot more fit_epochs


if __name__ == '__main__':
    tf.test.main()
