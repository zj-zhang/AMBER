"""Testing train environment test for architects
This is different from other helper tests, because train environment's work depend on helpers being functioning in the
expected behaviors
"""

import logging
import os
import sys
import tempfile
import unittest

import numpy as np
from parameterized import parameterized_class

import amber.backend as F
from amber import architect, modeler
from amber.utils import testing_utils

logging.disable(sys.maxsize)


def get_random_data(num_samps=1000):
    x = np.random.sample(10*4*num_samps).reshape((num_samps, 10, 4))
    y = np.random.sample(num_samps)
    return x, y


def get_class_name(*args):
    id = args[1]
    map = {
        0 : 'TestGeneralEnv',
        1 : 'TestEnasEnv'
    }
    return map[id]


@parameterized_class(attrs=('foo', 'manager_getter', 'controller_getter', 'modeler_getter', 'trainenv_getter'), input_values=[
    (0, architect.GeneralManager, architect.GeneralController, modeler.resnet.ResidualCnnBuilder, architect.ControllerTrainEnvironment),
    (1, architect.EnasManager, architect.GeneralController, modeler.supernet.EnasCnnModelBuilder, architect.EnasTrainEnv)
], class_name_func=get_class_name)
class TestEnvDryRun(testing_utils.TestCase):
    """Test dry-run will only aim to construct a train env class w/o examining its behaviors; however, this will
    serve as the scaffold for other tests
    """
    manager_getter = architect.GeneralManager
    controller_getter = architect.GeneralController
    modeler_getter = modeler.resnet.ResidualCnnBuilder
    trainenv_getter = architect.ControllerTrainEnvironment
    save_controller_every = 1

    def __init__(self, *args, **kwargs):
        super(TestEnvDryRun, self).__init__(*args, **kwargs)
        self.train_data = get_random_data(50)
        self.val_data = get_random_data(10)
        self.model_space, _ = testing_utils.get_example_conv1d_space(out_filters=8, num_layers=2)
        self.reward_fn = architect.reward.LossReward()
        self.store_fn = architect.store.get_store_fn('minimal')

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.sess = F.Session()
        self.controller = self.controller_getter(
            model_space=self.model_space,
            buffer_type='ordinal',
            with_skip_connection=True,
            kl_threshold=0.05,
            buffer_size=1,
            batch_size=3,
            session=self.sess,
            train_pi_iter=10,
            lstm_size=16,
            lstm_num_layers=1,
            optim_algo="adam",
            skip_target=0.8,
            skip_weight=0.4,
        )
        self.model_fn = self.modeler_getter(
            model_space=self.model_space,
            inputs_op=architect.Operation('input', shape=(10, 4)),
            output_op=architect.Operation('dense', units=1, activation='sigmoid'),
            fc_units=5,
            flatten_mode='gap',
            model_compile_dict={'optimizer': 'adam', 'loss': 'mse'},
            batch_size=10,
            session=self.sess,
            controller=self.controller,
            verbose=0
        )
        self.manager = self.manager_getter(
            train_data=self.train_data,
            validation_data=self.val_data,
            model_fn=self.model_fn,
            reward_fn=self.reward_fn,
            store_fn=self.store_fn,
            working_dir=self.tempdir.name,
            child_batchsize=10,
            epochs=1,
            verbose=0
        )


    def tearDown(self):
        super(TestEnvDryRun, self).tearDown()
        try:
            self.sess.close()
        except:
            pass
        self.tempdir.cleanup()

    def test_build(self):
        self.env = self.trainenv_getter(
            self.controller,
            self.manager,
            max_episode=3,
            max_step_per_ep=1,
            logger=None,
            resume_prev_run=False,
            should_plot=True,
            working_dir=self.tempdir.name,
            with_skip_connection=True,
            save_controller_every=self.save_controller_every
        )
        self.env.train()
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir.name, 'controller_weights.h5')))
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir.name, 'train_history.csv')))
        # test resume
        self.env_resume = self.trainenv_getter(
            self.controller,
            self.manager,
            max_episode=5,
            max_step_per_ep=1,
            logger=None,
            resume_prev_run=True,
            should_plot=True,
            working_dir=self.tempdir.name,
            with_skip_connection=True
        )
        self.env_resume.train()

# https://github.com/zj-zhang/AMBER/blob/89cdd45f8803014cadb131159d8bea804bfefcbc/examples/AMBIENT/sim_data/zero_shot_nas.sim_data.py
@parameterized_class(attrs=('manager_getter', 'controller_getter', 'modeler_getter', 'trainenv_getter', 'child_warmup_epochs'), input_values=[
    (architect.GeneralManager, architect.ZeroShotController, modeler.resnet.ResidualCnnBuilder, architect.MultiManagerEnvironment, 0),
    (architect.EnasManager, architect.ZeroShotController, modeler.supernet.EnasCNNwDataDescriptor, architect.MultiManagerEnvironment, 1),
])
@unittest.skipIf(F.mod_name!='tensorflow_1', "only implemented in TF1 backend")
class TestMultiManagerEnv(TestEnvDryRun):
    manager_getter = architect.GeneralManager
    controller_getter = architect.ZeroShotController
    modeler_getter = modeler.resnet.ResidualCnnBuilder
    trainenv_getter = architect.MultiManagerEnvironment
    data_description = np.eye(2)
    data_description_len = 2
    child_warm_up_epochs = 0

    def __init__(self, *args, **kwargs):
        super(TestEnvDryRun, self).__init__(*args, **kwargs)
        self.model_space, _ = testing_utils.get_example_conv1d_space(out_filters=8, num_layers=2)
        self.datasets ={
            'dataset1':
                {'descriptor': np.array([[1,0]]), 'train': get_random_data(50), 'valid': get_random_data(10), 'test': get_random_data(10)},
            'dataset2':
                {'descriptor': np.array([[0,1]]), 'train': get_random_data(50), 'valid': get_random_data(10), 'test': get_random_data(10)}
        }

    # controller should have been tested elsewhere; e.g. architect_optimize_test.py
    def get_controller(self, sess):
        controller = self.controller_getter(
            data_description_config={
                "length": self.data_description_len,
                "hidden_layer": {"units": 4, "activation": "relu"},
                "regularizer": {"l1":  1e-8}
            },
            model_space=self.model_space,
            buffer_type='MultiManager',
            with_skip_connection=True,
            kl_threshold=0.05,
            buffer_size=1,
            batch_size=3,
            session=sess,
            train_pi_iter=10,
            lstm_size=16,
            lstm_num_layers=1,
            optim_algo="adam",
        )
        return controller

    # manager should have been tested elsewhere; e.g. architect_helper_test.py
    def get_manager(self, sess, controller, dataset_key):
        model_fn = self.modeler_getter(
            # specific to EnasCnnModelBuilder; will be ignored by KerasResidualCnnModelBuilder
            **{
                'with_skip_connection': True,
                'add_conv1_under_pool': False,
                'stem_config': {
                    'has_stem_conv': True,
                    'flatten_op': 'flatten',
                    'fc_units': 30
                },
                'data_description': self.datasets[dataset_key]['descriptor'],
                'name': dataset_key
            },
            # End
            model_space=self.model_space,
            inputs_op=architect.Operation('input', shape=(10, 4)),
            output_op=architect.Operation('dense', units=1, activation='sigmoid'),
            fc_units=5,
            flatten_mode='gap',
            model_compile_dict={'optimizer': 'adam', 'loss': 'mse'},
            batch_size=10,
            session=sess,
            controller=controller,
            verbose=0
        )
        reward_fn = architect.reward.LossReward()
        store_fn = 'minimal'
        manager = self.manager_getter(
            train_data=self.datasets[dataset_key]['train'],
            validation_data=self.datasets[dataset_key]['valid'],
            model_fn=model_fn,
            reward_fn=reward_fn,
            store_fn=store_fn,
            working_dir=os.path.join(self.tempdir.name, dataset_key),
            child_batchsize=10,
            epochs=1,
            verbose=0,
            devices=None,
            train_data_kwargs=None,
            validate_data_kwargs=None
        )
        return manager

    def test_build(self):
        is_enas = self.modeler_getter in (modeler.supernet.EnasAnnModelBuilder, modeler.supernet.EnasCnnModelBuilder)
        sess = F.Session()
        F.set_session(sess)
        controller = self.get_controller(sess=sess)
        managers = []
        for dataset_key in self.datasets:
            managers.append(self.get_manager(
                sess=sess,
                controller=controller,
                dataset_key=dataset_key
            ))
        env = self.trainenv_getter(
            data_descriptive_features=self.data_description,
            controller=controller,
            manager=managers,
            max_episode=3,
            max_step_per_ep=1,
            working_dir=self.tempdir.name,
            time_budget="0:03:00",
            with_input_blocks=False,
            with_skip_connection=False,
            child_warm_up_epochs=2 if is_enas else 0,
            should_plot=False,
            save_controller=True
        )
        env.train()
        # test resume
        env2 = self.trainenv_getter(
            data_descriptive_features=self.data_description,
            controller=controller,
            manager=managers,
            max_episode=5,
            max_step_per_ep=1,
            working_dir=self.tempdir.name,
            time_budget="0:03:00",
            with_input_blocks=False,
            with_skip_connection=False,
            child_warm_up_epochs=2 if is_enas else 0,
            should_plot=False,
            save_controller=True,
            resume_prev_run=True
        )
        sess.close()
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir.name, 'controller_weights.h5')))
        self.assertTrue(os.path.isfile(os.path.join(self.tempdir.name, 'train_history.csv')))

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tempdir.cleanup()


if __name__ == '__main__':
    unittest.main()
