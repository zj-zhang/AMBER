"""Test DAG modeler

These modelers will convert a DAG representation to a neural network
"""

import tensorflow as tf
import numpy as np
from parameterized import parameterized_class
from amber.utils import testing_utils
from amber import modeler
from amber import architect
import logging, sys
logging.disable(sys.maxsize)


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


# TODO: test Ann with a upstream feature model, e.g. CNN
@parameterized_class(attrs=('with_output_blocks',), input_values=[
    (True,),
    (False,)
])
class TestEnasAnnDAG(TestKerasDAG):
    with_output_blocks = False
    fit_epochs = 15

    def setUp(self):
        super().setUp()
        # get data (again)
        self.traindata, self.validdata, self.testdata = self.get_data(seed=111, blockify_inputs=False)

    def get_model_builder(self, sess):
        model_fn = modeler.EnasAnnModelBuilder(
            session=sess,
            model_space=self.model_space,
            inputs_op=self.inputs_op,
            output_op=self.output_op,
            l1_reg=0.001,
            l2_reg=0.0001,
            model_compile_dict=self.model_compile_dict,
            controller=None,
            use_node_dag=False,
            with_output_blocks=self.with_output_blocks,
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
                # buffer_type='ordinal',
                with_skip_connection=True,
                with_input_blocks=True,
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

    def test_multi_input(self):
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
        with self.session() as sess:
            model_fn = self.get_model_builder(sess=sess)
            m1 = model_fn(arc1)
            m2 = model_fn(arc2)
            m1_old = m1.evaluate(*self.traindata)
            m1_train, m1_test = self.model_fit(m1)
            # m1 train loss should decrease after training fix-arc m1; test loss not necessarily decrease
            self.assertLess(m1_train[0], m1_old[0])
            m2_old = m2.evaluate(*self.traindata)
            m2_train, m2_test = self.model_fit(m2)
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
            # sampled graph should also decrease loss over training, though it
            # may fail due to randomness
            self.assertLess(train_loss[0], old_loss[0])


if __name__ == '__main__':
    tf.test.main()
