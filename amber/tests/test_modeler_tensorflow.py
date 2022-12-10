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


@parameterized_class([
     {'loss': tf.keras.losses.MeanSquaredError()},
     {'loss': ['binary_crossentropy']},
     {'loss': 'binary_crossentropy'},
     {'reg': 0},
     {'reg': 1e-8},
     {'metrics': None},
     {'metrics': ['mae']},
     {'has_stem_conv': True},
     # not working - must have stem-conv
     #{'has_stem_conv': False},    
])
class TestEnasConvModeler(testing_utils.TestCase):
    loss = 'binary_crossentropy'
    metrics = None
    reg = 0
    has_stem_conv = True

    def setUp(self):
        self.sess = F.Session()
        self.input_op = [architect.Operation('input', shape=(10, 4), name="input")]
        self.output_op = architect.Operation('dense', units=1, activation='sigmoid', name="output")
        self.x = np.random.randn(2, 10, 4)
        self.y = np.random.sample(2).reshape((2, 1))
        self.model_space, _ = testing_utils.get_example_conv1d_space(num_layers=4, num_pool=2)
        #self.model_space, _ = testing_utils.get_example_conv1d_space()
        self.model_compile_dict = {'loss': self.loss, 'optimizer': 'sgd', 'metrics':self.metrics}
        self.controller = architect.GeneralController(
            model_space=self.model_space,
            buffer_type='ordinal',
            with_skip_connection=True,
            kl_threshold=0.05,
            buffer_size=15,
            batch_size=5,
            session=self.sess,
            train_pi_iter=2,
            lstm_size=32,
            lstm_num_layers=1,
            lstm_keep_prob=1.0,
            optim_algo="adam",
            skip_target=0.8,
            skip_weight=0.4,
        )
        self.decoder = modeler.architectureDecoder.ResConvNetArchitecture(model_space=self.model_space)
        self.target_arc = self.decoder.sample(seed=777)
        self.enas_modeler = modeler.supernet.EnasCnnModelBuilder(
            model_space=self.model_space,
            num_layers=len(self.model_space),
            inputs_op=self.input_op,
            output_op=self.output_op,
            model_compile_dict=self.model_compile_dict,
            session=self.sess,
            controller=self.controller,
            l1_reg=1e-8,
            l2_reg=self.reg,
            batch_size=1,
            stem_config= {
                    'has_stem_conv': self.has_stem_conv,
                    'fc_units': 5}
        )
        self.num_samps = 15
    
    def tearDown(self):
        try:
            self.sess.close()
        except:
            pass
        super().tearDown()

    def test_sample_arc_builder(self):
        model = self.enas_modeler()
        samp_preds = [model.predict(self.x).flatten()[0] for _ in range(self.num_samps)]
        # sampled loss can't be always identical
        self.assertNotEqual(len(set(samp_preds)), 1)
        old_loss = [model.evaluate(self.x, self.y)['val_loss'] for _ in range(self.num_samps)]
        model.fit(self.x, self.y, batch_size=1, epochs=100, verbose=0)
        new_loss = [model.evaluate(self.x, self.y)['val_loss'] for _ in range(self.num_samps)]
        self.assertLess(sum(new_loss), sum(old_loss))

    def test_fix_arc_builder(self):
        model = self.enas_modeler(arc_seq=self.target_arc)
        # fixed preds must always be identical
        fix_preds = [model.predict(self.x).flatten()[0] for _ in range(self.num_samps)]
        self.assertEqual(len(set(fix_preds)), 1)
        # record original loss
        old_loss = model.evaluate(self.x, self.y)['val_loss']
        # train weights with sampled arcs from model2
        model2 = self.enas_modeler()
        model2.fit(self.x, self.y, batch_size=1, epochs=100, verbose=0)
        # loss should reduce
        new_loss = model.evaluate(self.x, self.y)['val_loss']
        # XXX: this will often fail - may indicate dropouts are not turned off.
        self.assertLess(new_loss, old_loss)
        # fixed preds should still be identical
        fix_preds = [model.predict(self.x).flatten()[0] for _ in range(self.num_samps)]
        self.assertEqual(len(set(fix_preds)), 1)


class TestEnasConvModelerIO(TestEnasConvModeler):
    def test_model_save_load(self):
        tempdir = tempfile.TemporaryDirectory()
        model = self.enas_modeler(arc_seq=self.target_arc)
        model.save(filepath=os.path.join(tempdir.name, "save.h5"))
        model.save_weights(filepath=os.path.join(tempdir.name, "save_weights.h5"))
        model.load_weights(filepath=os.path.join(tempdir.name, "save_weights.h5"))
        tempdir.cleanup()


class TestSinglePathCnnSupernet(testing_utils.TestCase):
    def setUp(self):
        self.sess = F.Session()
        self.input_op = [architect.Operation('input', shape=(10, 4), name="input")]
        self.output_op = architect.Operation('dense', units=1, activation='sigmoid', name="output")
        self.model_compile_dict = {'loss': 'mse', 'optimizer': 'sgd'}
        self.x = np.random.choice(2, 40).reshape((1, 10, 4))
        self.y = np.random.sample(1).reshape((1, 1))
        self.model_space, _ = testing_utils.get_example_conv1d_space()
        self.decoder = modeler.architectureDecoder.ResConvNetArchitecture(model_space=self.model_space)
        self.num_samps = 15

    def tearDown(self):
        try:
            self.sess.close()
        except:
            pass
        super().tearDown()
    
    def test_single_path(self):
        arc = self.decoder.sample()
        enas_modeler = modeler.supernet.EnasCnnModelBuilder(
            fixed_arc=arc,
            train_fixed_arc=True,
            model_space=self.model_space,
            num_layers=len(self.model_space),
            inputs_op=self.input_op,
            output_op=self.output_op,
            model_compile_dict=self.model_compile_dict,
            session=self.sess,
            controller=None,
            batch_size=1,
            stem_config={'has_stem_conv': True,
                    'fc_units': 5}
        )        
        model = enas_modeler(arc_seq=arc)
        # test pred
        fix_preds = [model.predict(self.x).flatten()[0] for _ in range(self.num_samps)]
        #print(fix_preds) # indeed not the same. no fix yet. FZZ 20221209
        #self.assertEqual(len(set(fix_preds)), 1)
        # test train & eval
        old_loss = model.evaluate(self.x, self.y)['val_loss']
        model.fit(self.x, self.y, batch_size=1, epochs=10, verbose=0)
        new_loss = model.evaluate(self.x, self.y)['val_loss']
        # seems dropout is not turned off yet
        #self.assertLess(new_loss, old_loss)


# See also: test Ann with a upstream feature model, e.g. CNN, see `test_supernet_featmodel.py`
@parameterized_class(
    attrs=('with_output_blocks', 'with_input_blocks', 'use_pipe', 'use_node_dag'), input_values=[
        (True,  True,  True, False, False),
        (False, True,  True, False, False),
        (False, False, True, False, False),
        (True,  True,  True, True,  True),
])
class TestEnasAnnDAG(testing_utils.TestCase):
    with_output_blocks = False
    fit_epochs = 15
    with_input_blocks = True
    with_skip_connection = True
    use_pipe = False
    use_node_dag = False
    num_inputs = 4
    model_compile_dict = {'optimizer': 'sgd', 'loss': 'mse', 'metrics': ['mae']}

    def setUp(self):
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
        n1 = 1500
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
    
    def get_model_builder(self, sess):
        model_fn = modeler.supernet.EnasAnnModelBuilder(
            session=sess,
            model_space=self.model_space,
            inputs_op=self.inputs_op,
            output_op=self.output_op,
            l1_reg=0.001,
            l2_reg=0.0001,
            model_compile_dict=self.model_compile_dict,
            controller=None,
            use_node_dag=self.use_node_dag,
            use_pipe=self.use_pipe,
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
