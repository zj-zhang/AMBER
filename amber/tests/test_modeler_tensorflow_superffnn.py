"""Test DAG modeler

These modelers will convert a DAG representation to a neural network
"""

import logging
import os
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf
from parameterized import parameterized_class

from amber import architect, modeler
from amber import backend as F
from amber.utils import testing_utils

logging.disable(sys.maxsize)


# See also: test Ann with a upstream feature model, e.g. CNN, see `test_supernet_featmodel.py`
@parameterized_class([
        {'with_output_blocks': False},
        {'with_input_blocks': False},
        {'reg': 0},
        {'reg': 1e-8},
        {'loss': 'mse'},
        {'loss': tf.keras.losses.MeanSquaredError()},
        {'loss': ['mse']},
        {'metrics': []},
        {'metrics': ['mae']}
])
class TestEnasAnnDAG(testing_utils.TestCase):
    with_output_blocks = True
    with_input_blocks = True
    reg = 0
    metrics = ['mae']
    loss = 'mse'

    fit_epochs = 10
    with_skip_connection = True
    num_inputs = 4

    def setUp(self):
        self.model_compile_dict = {'optimizer': 'sgd', 'loss': self.loss, 'metrics': self.metrics}
        #print(self.with_output_blocks, self.with_input_blocks, self.with_skip_connection)
        if self.with_input_blocks:
            self.inputs_op = [architect.Operation('input', shape=(1,), name='X_%i' % i) for i in range(self.num_inputs)]
        else:
            self.inputs_op = [architect.Operation('input', shape=(self.num_inputs,), name='X')]
        self.output_op = architect.Operation('Dense', units=1, activation='linear', name='output')
        # get a three-layer model space
        self.model_space = testing_utils.get_example_sparse_model_space(4)
        # get data (again)
        self.traindata, self.validdata, self.testdata = self.get_data(seed=111, blockify_inputs=False)
    
    def get_data(self, seed=111, blockify_inputs=True):
        n1 = 1000
        n2 = 500
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
    
    def get_model_builder(self, sess):
        model_fn = modeler.supernet.EnasAnnModelBuilder(
            session=sess,
            model_space=self.model_space,
            inputs_op=self.inputs_op,
            output_op=self.output_op,
            l1_reg=self.reg,
            l2_reg=self.reg,
            model_compile_dict=self.model_compile_dict,
            controller=None,
            with_output_blocks=self.with_output_blocks,
            with_input_blocks=self.with_input_blocks,
            with_skip_connections=self.with_skip_connection,
        )
        return model_fn

    def get_controller(self, sess):
        with tf.device("/cpu:0"):
            controller = architect.MultiIOController(
                model_space=self.model_space,
                session=sess,
                output_block_unique_connection=False,
                num_output_blocks=1,
                with_output_blocks=self.with_output_blocks,
                share_embedding={i: 0 for i in range(1, len(self.model_space))},
                with_skip_connection=self.with_skip_connection,
                with_input_blocks=self.with_input_blocks,
                num_input_blocks=4,
                input_block_unique_connection=True,
                skip_connection_unique_connection=False,
                lstm_size=32,
                kl_threshold=0.05,
                train_pi_iter=100,
                skip_weight=None,
                lr_init=0.001,
                buffer_size=1,
                batch_size=2
            )
        return controller

    def model_fit(self, model):
        model.fit(
            self.traindata[0], self.traindata[1],
            batch_size=200,
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
        with self.session() as sess:
            if self.with_input_blocks:
                # correct model: y = h1(x0,x1) + h2(x2,x3)
                arc1 = np.array([
                    0, 0, 0, 1, 1,
                    0, 1, 1, 0, 0, 0,
                    0, 0, 0, 0, 0, 1, 1,
                    0, 0, 0, 0, 0, 0, 0, 1
                ], dtype=np.int32)
                # fully-connected model: y = f(x0,x1,x2,x3)
                arc2 = np.array([
                    0, 1, 1, 1, 1,
                    0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 1,
                    0, 0, 0, 0, 0, 0, 0, 1
                ], dtype=np.int32)
                # add another (num_output x num_layers, 1) vector to represent the output block connections
                if self.with_output_blocks:
                    num_output_blocks = len(self.traindata[1]) if type(self.traindata[1]) in (list, tuple) else 1
                    output_bl = np.random.randint(0,2,num_output_blocks*len(self.model_space))
                    arc1 = np.concatenate([arc1, output_bl])
                    arc2 = np.concatenate([arc2, output_bl])
            else:
                # if no with_input_blocks, 
                # sample arcs from controller
                controller = self.get_controller(sess=sess)
                arc1, _ = controller.get_action()
                arc2, _ = controller.get_action()

            model_fn = self.get_model_builder(sess=sess)
            m1 = model_fn(arc1)
            m2 = model_fn(arc2)
            m1_old = m1.evaluate(*self.traindata)
            m2_old = m2.evaluate(*self.traindata)
            m1_train, m1_test = self.model_fit(m1)
            m2_train, m2_test = self.model_fit(m2)
            # m1 train loss should decrease after training fix-arc m1; test loss not necessarily decrease
            self.assertLess(m1_train[0], m1_old[0])
            # m2 train loss should also decrease after training fix-arc m2
            self.assertLess(m2_train[0], m2_old[0])

    def test_set_controller(self):
        with self.session() as sess:
            model_fn = self.get_model_builder(sess=sess)
            controller = self.get_controller(sess=sess)
            model_fn.set_controller(controller)
            # sample arch
            m = model_fn()
            old_loss = m.evaluate(*self.traindata)
            train_loss, test_loss = self.model_fit(m)
            new_loss = m.evaluate(*self.traindata)
            # sampled graph should also decrease loss over training, though it
            # may fail due to randomness in dropout
            self.assertLess(new_loss[0], old_loss[0])


class TestAnnModelIO(TestEnasAnnDAG):
    def test_model_save_load(self):
        tempdir = tempfile.TemporaryDirectory()
        with self.session() as sess:
            model_fn = self.get_model_builder(sess=sess)
            controller = self.get_controller(sess=sess)
            model_fn.set_controller(controller)
            arc, _ = controller.get_action()
            model = model_fn(arc)
            model.save(filepath=os.path.join(tempdir.name, "save.h5"))
            model.save_weights(filepath=os.path.join(tempdir.name, "save_weights.h5"))
            model.load_weights(filepath=os.path.join(tempdir.name, "save_weights.h5"))
        tempdir.cleanup()


if __name__ == '__main__':
    tf.test.main()
