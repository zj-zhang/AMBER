# !/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Testing amber.modeler.sparse_ffnn.FeatModelSparseFfnnModelBuilder

:class`EnasAnnModelBuilder` can optionally take a pretrained CNN backbone as a
`feature_model`, and perform a FFNN search after CNN feature map.
This can be very useful for Grad-CAM-type of tasks, e.g. VQA (visual question answering)
"""

import amber
from amber import backend as F
from amber import architect, modeler
from amber.utils import testing_utils
import warnings
import unittest
import numpy as np
import copy
import tensorflow as tf
from parameterized import parameterized_class


@parameterized_class(attrs=[
    {"feat_model_trainable": True},
    {'with_output_blocks': False},
    {'output_block_unique_connection': False},
    {'with_input_blocks': False},
    {'input_block_unique_connection': False},
    {'skip_connection_unique_connection': False},
    {'loss': 'mse'},
    {'loss': ['mse']*2},
    {'loss': [tf.keras.losses.MeanSquaredError()]}
])
class TestFeatModelSparseFfnnModelBuilder(testing_utils.TestCase):
    with_output_blocks = True
    output_block_unique_connection = True
    with_input_blocks = True
    input_block_unique_connection = True
    with_skip_connection = True
    skip_connection_unique_connection=True

    last_conv_channels = 9
    num_input_blocks = 3
    num_output_blocks = 2
    num_layers = 4
    n1 = 1000
    n2 = 500
    loss = 'mse'
    feat_model_trainable = False

    def setUp(self):
        #self.sess = F.Session()
        # somehow setting a session will cause precondition error
        # but not setting it and using the global default graph
        # will work. wierd. FZZ 2022.12.08
        self.sess = None
        self.base_cnn = self.get_base_feature_model()
        self.datas = self.get_data()
        self.model_space = self.get_model_space(num_layers=4)
        self.model_compile_dict = {'loss': self.loss, 'optimizer': 'sgd'}
        # feature_assign: a vector of integers, length is the 
        # total number of channels, each is a group assignment
        _groups = np.arange(self.num_input_blocks)
        self.feature_assign = np.array([_groups[i//self.num_input_blocks] for i in range(self.last_conv_channels)])
        # helper
        self.decoder = amber.modeler.architectureDecoder.MultiIOArchitecture(
            model_space=self.model_space,
            num_inputs=self.num_input_blocks,
            num_outputs=self.num_output_blocks
        )
        self.output_ops = [F.Operation('Dense', units=i, activation='linear', name='output_%i'%i)
            for i in np.arange(1, self.num_output_blocks+1) ]
        
    def tearDown(self):
        try:
            self.sess.close()
        except:
            pass
        return super().tearDown()

    def get_data(self, seed=111):
        np.random.seed(seed)
        X_train = np.random.randn(self.n1, 10, 4)
        X_valid = np.random.randn(self.n2, 10, 4)
        X_test = np.random.randn(self.n2, 10, 4)

        y_train = np.random.randn(self.n1,1)
        y_valid = np.random.randn(self.n2,1)
        y_test = np.random.randn(self.n2,1)
        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

    @staticmethod
    def get_model_space(num_layers):
        model_space = architect.ModelSpace()
        for i in range(num_layers):
            model_space.add_layer(i, [
                F.Operation('Dense', units=5, activation='relu'),
                F.Operation('Dense', units=10, activation='relu'),
                F.Operation('Dense', units=25, activation='relu'),
            ])
        return model_space

    def get_base_feature_model(self):
        mb = modeler.sequential.SequentialModelBuilder(
            inputs_op = F.Operation('Input', shape=(10,4)),
            output_op = F.Operation('Dense', units=1, activation='linear'),
            model_compile_dict = {'loss': 'mse', 'optimizer': 'adam'},
            model_space=None,
        )

        base_cnn = mb(model_states=[
            F.Operation('Conv1D', filters=5, kernel_size=3, activation='relu', padding="same", name="conv1_base"),
            F.Operation('Conv1D', filters=self.last_conv_channels, kernel_size=3, activation='relu', padding="same", name="conv2_base"),
            F.Operation('Flatten')
        ]) 
        return base_cnn

    def get_controller(self):
        controller = architect.optim.controller.MultiIOController(
            model_space=self.model_space,

            with_skip_connection=self.with_skip_connection,
            with_input_blocks=self.with_input_blocks,
            with_output_blocks=self.with_output_blocks,

            num_input_blocks=self.num_input_blocks,
            num_output_blocks=self.num_output_blocks,

            input_block_unique_connection=self.input_block_unique_connection,
            skip_connection_unique_connection=self.skip_connection_unique_connection,
            output_block_unique_connection=self.output_block_unique_connection,

            share_embedding={i: 0 for i in range(1, len(self.model_space))},
            skip_weight=None,
            buffer_size=1,  ## num of episodes saved
            batch_size=5,
            session=F.get_session()
        )
        return controller

    def test_cnn_feat_model(self):
        # feature_assing = array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        # cluster of channels into groups
        feat_model = modeler.sparse_ffnn.CnnFeatureModel(
            base_model=self.base_cnn, 
            feature_assign=self.feature_assign,
            trainable=self.feat_model_trainable,
            #session=self.sess
            )
        _, _, test_data = self.datas
        pred = feat_model.predict(test_data[0])
        assert pred.shape == (500, 10, 9)
        pred2 = feat_model.predict(test_data[0], keep_spatial=False)
        assert pred2.shape == (500, 90)

    def test_featmod_fixedarc(self):
        self.sess = F.Session()
        # XXX: Context manager as_default() is necessary
        with self.sess.as_default():
            feat_model = modeler.sparse_ffnn.CnnFeatureModel(
                base_model=self.base_cnn, 
                feature_assign=self.feature_assign,
                session=self.sess,
                trainable=self.feat_model_trainable
                )
            # test loss as a list
            model_compile_dict = copy.copy(self.model_compile_dict)
            model_compile_dict['loss'] = [model_compile_dict['loss'] for _  in range(self.num_output_blocks)]
            model_fn = modeler.sparse_ffnn.FeatModelSparseFfnnModelBuilder(
                session=self.sess,
                model_space=self.model_space,
                inputs_op=feat_model.input_node_for_nas,
                output_op=self.output_ops,
                l1_reg=1e-8,
                l2_reg=5e-7,
                model_compile_dict=self.model_compile_dict,
                controller=None,
                feature_model=feat_model,
                feature_model_trainable=self.feat_model_trainable,
                # configs
                with_output_blocks=self.with_output_blocks,
                with_input_blocks=self.with_input_blocks,
                with_skip_connection=self.with_skip_connection,
                )
            # test model creation by fixed arc
            #arc = self.decoder.sample()
            arc = np.random.binomial(n=1, p=0.5, size=len(model_fn.input_arc))
            model = model_fn(arc_seq=arc)
            # test pred
            train_data, val_data, test_data = self.datas
            pred = model.predict(test_data[0])
            assert len(pred) == self.num_output_blocks
            for i, p in enumerate(pred):
                self.assertEqual(p.shape, (500, i+1))
            # test train
            y1_blocks = [np.random.randn(self.n1, i+1) for i in range(self.num_output_blocks)]
            y2_blocks = [np.random.randn(self.n2, i+1) for i in range(self.num_output_blocks)]
            hist = model.fit(
                x=train_data[0], y=y1_blocks, 
                batch_size=256, 
                validation_data=[val_data[0], y2_blocks], epochs=2,
                verbose=False)
            # test evalute
            loss1 = model.evaluate(x=train_data[0], y=y1_blocks)

    def test_featmodel_samplearc(self):
        self.sess = F.Session()
        with self.sess.as_default():
            controller = self.get_controller()
            feat_model = modeler.sparse_ffnn.CnnFeatureModel(
                base_model=self.base_cnn, 
                feature_assign=np.repeat(np.arange(3), 3),
                session=self.sess,
                trainable=self.feat_model_trainable
                )
            model_fn = modeler.sparse_ffnn.FeatModelSparseFfnnModelBuilder(
                session=self.sess,
                model_space=self.model_space,
                inputs_op=feat_model.input_node_for_nas,
                output_op=self.output_ops,
                model_compile_dict=self.model_compile_dict,
                controller=controller,
                feature_model=feat_model,
                # configs
                feature_model_trainable=self.feat_model_trainable,
                with_output_blocks=self.with_output_blocks,
                with_input_blocks=self.with_input_blocks,
                with_skip_connection=self.with_skip_connection,
                )
            #model_fn.set_controller(controller)
            # test model creation by sampled arc
            model = model_fn(arc_seq=None)
            # test pred
            train_data, val_data, test_data = self.datas
            pred = model.predict(test_data[0])
            assert len(pred) == self.num_output_blocks
            for i, p in enumerate(pred):
                    self.assertEqual(p.shape, (500, i+1))
            # test train
            y1_blocks = [np.random.randn(self.n1, i+1) for i in range(self.num_output_blocks)]
            y2_blocks = [np.random.randn(self.n2, i+1) for i in range(self.num_output_blocks)]
            hist = model.fit(
                x=train_data[0], y=y1_blocks, 
                batch_size=256, 
                validation_data=[val_data[0], y2_blocks], epochs=2)
            # test evalute
            loss1 = model.evaluate(x=train_data[0], y=y1_blocks)


if __name__ == '__main__' and not amber.utils.run_from_ipython():
    unittest.main()
