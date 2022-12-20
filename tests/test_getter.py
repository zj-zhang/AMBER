import os
import amber
from amber import getter, architect, modeler
from amber import backend as F
from amber.utils import testing_utils
import unittest, pytest, tempfile
import numpy as np, h5py, pickle


def test_data_to_parse():
    tempdir = tempfile.TemporaryDirectory()
    in_mem_data = np.random.randn(100, 20, 4)
    # save by existing packages
    with open(os.path.join(tempdir.name, 'example.npy'), 'wb') as f:
        np.save(f, in_mem_data)
    with open(os.path.join(tempdir.name, 'example.pkl'), 'wb') as f:
        pickle.dump(in_mem_data, f)
    with h5py.File(os.path.join(tempdir.name, 'example.h5'), 'w') as store:
        store.create_dataset('X', data=in_mem_data)
    # load from npy
    data = getter.DataToParse(os.path.join(tempdir.name, 'example.npy'))
    arr = data.unpack()
    assert np.array_equal(arr, in_mem_data)
    # the same syntax also works if the file is pickle
    data = getter.DataToParse(os.path.join(tempdir.name, 'example.pkl'))
    arr = data.unpack()
    assert np.array_equal(arr, in_mem_data)
    # and also h5py
    data = getter.DataToParse(os.path.join(tempdir.name, 'example.h5'))
    with data.unpack() as store:
        arr = store['X'][()]
    assert np.array_equal(arr, in_mem_data)
   # finally, you can lazy init by passing a callable
    def read_ith_data(index):
        return in_mem_data[index]
    data = getter.DataToParse(obj=read_ith_data, kws={'index': [8,9,10]})
    arr = data.unpack()
    assert np.array_equal(arr, in_mem_data[[8,9,10]])
    # load_data_dict provides an iterative way to load and unpack data
    # of mixed types
    data_dict = {
        'train_data': in_mem_data,  
        "validation_data": data,
        "test_data": os.path.join(tempdir.name, 'example.npy'),
        "knowledge_data": os.path.join(tempdir.name, 'example.pkl'),
        }
    data_unpacked = getter.load_data_dict(data_dict)
    # cleanup
    tempdir.cleanup()
    return data_unpacked
    
@pytest.mark.parametrize("env_type", ['ControllerTrainEnv', 'EnasTrainEnv',])
def test_get_train_env(env_type):
    tempdir = tempfile.TemporaryDirectory()
    model_space, _ = testing_utils.get_example_conv1d_space(num_layers=2, num_pool=1)
    #controller = architect.optim.controller.GeneralController(model_space=model_space, session=F.Session())
    controller = architect.optim.controller.BaseController(model_space=model_space)
    controller.get_action = unittest.mock.MagicMock(return_value=[
        np.array([0,0,1]), 
        [np.ones((1,4))/4, np.ones((1,4))/4, np.ones((1,1,2))/2]
        ])
    controller.buffer = architect.buffer.get_buffer('ordinal')(max_size=10, is_squeeze_dim=True)
    controller.train = unittest.mock.MagicMock(return_value=0)
    controller.save_weights = unittest.mock.MagicMock()
    model_fn = modeler.base.BaseModelBuilder(
        inputs_op=F.Operation('input'),
        output_op=F.Operation('output')
    )
    manager = architect.base.BaseNetworkManager(model_fn=model_fn, working_dir=tempdir.name)
    setattr(manager.model_fn, 'controller', controller)
    manager.get_rewards = unittest.mock.MagicMock(return_value=[1, {}])
    env = getter.get_train_env(env_type=env_type, controller=controller, manager=manager, working_dir=tempdir.name)
    env.train()
    tempdir.cleanup()


@pytest.mark.parametrize("env_type", ['MultiManagerEnvironment',])
def test_get_train_env_multi_manager(env_type):
    # parallel env is not working for MockMagic, see
    # https://stackoverflow.com/questions/60627827/picklingerror-when-using-mock-to-check-function-is-called
    # to test parallel env, see `test_parallel_env.py`
    tempdir = tempfile.TemporaryDirectory()
    model_space, _ = testing_utils.get_example_conv1d_space(num_layers=2, num_pool=1)
    controller = architect.optim.controller.BaseController(model_space=model_space)
    controller.get_action = unittest.mock.MagicMock(return_value=[
        np.array([0,0,1]), 
        [np.ones((1,4))/4, np.ones((1,4))/4, np.ones((1,1,2))/2]
        ])
    controller.buffer = architect.buffer.get_buffer('multimanager')(max_size=10, is_squeeze_dim=True)
    controller.train = unittest.mock.MagicMock(return_value=0)
    controller.save_weights = unittest.mock.MagicMock()
    model_fn = modeler.base.BaseModelBuilder(
        inputs_op=F.Operation('input'),
        output_op=F.Operation('output')
    )
    manager_list = []
    for _ in range(2):
        manager = architect.base.BaseNetworkManager(model_fn=model_fn, working_dir=tempdir.name)
        setattr(manager.model_fn, 'controller', controller)
        manager.get_rewards = unittest.mock.MagicMock(return_value=[1, {'loss':0, 'val_loss':0}])
        manager_list.append(manager)
    
    data_descriptor = np.eye(2) # must by shape (n_manager, ?)
    env = getter.get_train_env(
        env_type=env_type, 
        controller=controller, 
        manager=manager_list, 
        working_dir=tempdir.name,
        # specific to multi-manager
        data_descriptive_features=data_descriptor,
        )
    env.train()
    tempdir.cleanup()

@pytest.mark.parametrize(["controller_type","init_kws"], [
    ['GeneralController',{}], 
    ['OperationController',{}], 
    ['MultiIOController',{}],
    ['ZeroShotController', {'data_description_config': {'length':2}}]
])
def test_get_controller(controller_type, init_kws):
    model_space, _ = testing_utils.get_example_conv1d_space(num_layers=12, num_pool=4)
    sess = F.Session()
    with F.session_scope(sess):
        controller = getter.get_controller(
            controller_type=controller_type, 
            model_space=model_space,
            session=sess,
            **init_kws
        )
        arc, prob = controller.get_action()


@pytest.mark.parametrize('arg', [
    [[F.Operation('conv1d'), F.Operation('maxpool1d')], [F.Operation('conv1d'), F.Operation('maxpool1d')]],
    testing_utils.get_bionas_model_space()
])
def test_get_model_space(arg):
    ms = getter.get_model_space(arg)
    assert issubclass(type(ms), architect.base.BaseModelSpace)


@pytest.mark.parametrize(['manager_type', 'init_kws'], [
    ['GeneralManager', {}],
    ['EnasManager', {}],
    ['MockManager', {'history_fn_list':[os.path.join(os.path.dirname(__file__), "mock_black_box/tmp_%i/train_history.csv.gz" % i)
                               for i in range(1, 3)], 'model_compile_dict':{'metrics': ['acc']}}],
    ['DistributedManager', {}],
    [architect.manager.GeneralManager, {}]
])
def test_get_manager(manager_type, init_kws):
    data_dict = test_data_to_parse()
    model_fn = modeler.base.BaseModelBuilder(
        inputs_op=F.Operation('input'),
        output_op=F.Operation('output')
    )
    reward_fn = architect.base.BaseReward()
    sess = F.Session()
    with F.session_scope(sess):
        manager = getter.get_manager(
            manager_type=manager_type,
            model_fn=model_fn,
            reward_fn=reward_fn,
            data_dict=data_dict,
            session=sess,
            **init_kws
        )
        assert issubclass(type(manager), architect.base.BaseNetworkManager)


@pytest.mark.parametrize(["model_fn_type", "ms_type"], [
    ['SupernetCnn', 'cnn'],
    ['SequentialModelBuilder', 'cnn'],
    [modeler.sequential.SequentialModelBuilder, 'cnn'],

    ['SupernetDnn', 'dnn'],
    ['SparseFfnnModelBuilder', 'dnn'],
    ['SequentialMultiIOModelBuilder', 'dnn'], # multiIO needs more than one i/o

    ['SequentialBranchModelBuilder', 'branched'],
])
def test_get_modeler(model_fn_type, ms_type):
    if ms_type == 'cnn':
        model_space, _ = testing_utils.get_example_conv1d_space()
        inputs_op = [F.Operation('input', shape=(10,4))]
    elif ms_type == 'dnn':
        model_space, _ = testing_utils.get_example_sparse_model_space()
        inputs_op = [F.Operation('input', shape=(5,))]
    elif ms_type == 'branched':
        branch1, _ = testing_utils.get_example_conv1d_space(num_layers=4, num_pool=2)
        branch2, _ = testing_utils.get_example_conv1d_space(num_layers=2, num_pool=1)
        stem, _ = testing_utils.get_example_sparse_model_space(num_layers=2)
        model_space = architect.modelSpace.BranchedModelSpace(
            subspaces=[[branch1, branch2], stem]
        )
        inputs_op = [
            F.Operation('input', shape=(10,4)), # for branch 1
            F.Operation('input', shape=(10,4)), # for branch 2
        ]
    else:
        raise ValueError(f'unknown modelspace type: {ms_type}')
    sess = F.Session()
    outputs_op = [F.Operation('dense', units=1)]
    model_compile_dict = {'loss':'mse', 'optimizer':'adam'}
    model_fn = getter.get_modeler(
        model_fn_type=model_fn_type, 
        model_space=model_space,
        session=sess,
        inputs_op=inputs_op,
        outputs_op=outputs_op,
        model_compile_dict=model_compile_dict
        )
    assert callable(model_fn)
    # TODO: test model building by model_fn.__call__


@pytest.mark.parametrize('reward_fn_type', [
    'LossReward', 'KnowledgeReward', 
    "LossAucReward", "SparseCategoricalReward",
])
def test_get_reward_fn(reward_fn_type):
    # because knowledge_fn is currently not tested and deprecated
    knowledge_fn = None
    reward_fn = getter.get_reward_fn(reward_fn_type=reward_fn_type, knowledge_fn=knowledge_fn)



if __name__ == '__main__':
    #test_get_train_env(env_type='ControllerTrainEnv')
    #test_get_manager(manager_type='DistributedManager', init_kws={})
    #test_get_train_env_multi_manager(env_type='MultiManagerEnvironment')
    #test_get_modeler('SequentialBranchModelBuilder', 'branched')
    test_get_reward_fn('LossReward')
    pass
