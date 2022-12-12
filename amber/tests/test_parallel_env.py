import os
from amber import getter, architect, modeler
from amber import backend as F
from amber.utils import testing_utils
import unittest, pytest, tempfile
import numpy as np


def get_rand_data(n=1):
    return np.random.randn(n, 10, 4), np.random.randn(n,1)

@pytest.mark.parametrize(['env_type', 'num_process'], [
    ['ParallelMultiManagerEnvironment', 2],
    ['ParallelMultiManagerEnvironment', 1],
])
def test_parallel_env(env_type, num_process):
    tempdir = tempfile.TemporaryDirectory()
    sess = F.Session()
    with F.session_scope(sess):
        model_space, _ = testing_utils.get_example_conv1d_space(num_layers=2, num_pool=1)
        controller = architect.optim.controller.ZeroShotController(
            model_space=model_space,
            session=sess,
            data_description_config= {'length':2}         
        )
        model_fn = modeler.base.BaseModelBuilder(
            inputs_op=F.Operation('input'),
            output_op=F.Operation('output')
        )
        manager_list = []
        reward_fn = architect.base.BaseReward()
        for _ in range(2):
            manager = architect.manager.DistributedGeneralManager(
                model_fn=model_fn, 
                working_dir=os.path.join(tempdir.name, "manager_%i"%_),
                reward_fn=reward_fn, 
                train_data=get_rand_data,
                train_data_kwargs={'n':10},
                validation_data=get_rand_data,
                validate_data_kwargs={'n':5}
            )
            manager_list.append(manager)
        
        data_descriptor = np.eye(2) # must by shape (n_manager, ?)
        env = getter.get_train_env(
            env_type=env_type, 
            controller=controller, 
            manager=manager_list, 
            working_dir=tempdir.name,
            # specific to multi-manager
            data_descriptive_features=data_descriptor,
            devices=['/cpu:0'],
            processes=num_process,
            # reduce steps for test
            max_episode=3,
            max_steps_per_ep=1
            )
        env.train()
        tempdir.cleanup()

if __name__ == '__main__':
    test_parallel_env(env_type='ParallelMultiManagerEnvironment', num_process=2)
    pass
