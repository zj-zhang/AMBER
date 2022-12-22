import os
import pickle
import warnings
import tempfile

import numpy as np
from sklearn.model_selection import train_test_split

from amber import architect, modeler
from amber import backend
from amber import Amber, AmberSpecifications, DataToParse


warnings.filterwarnings("ignore")
CONFIG_FP = os.path.join(os.path.realpath(os.path.dirname(__file__)), 'test_conv2d_specs.pkl')


def get_data(data_type="all"):
    # Model / data parameters
    num_classes = 10
    n_samples = 2000
    data = np.random.randn(n_samples, 8, 8)
    target = np.random.multinomial(n=num_classes, pvals=np.ones(num_classes)/num_classes, size=n_samples)
    # flatten the images
    data = data.reshape((n_samples, 8, 8, 1))

    # Split data into 80% train and 20% test subsets
    X_trainvalid, X_test, y_trainvalid, y_test = train_test_split(
        data, target, test_size=0.2, shuffle=False, random_state=111
    )
    if data_type == "test":
        return X_test, y_test
    elif data_type == "trainvalid":
        return X_trainvalid, y_trainvalid
    # Split into 90% train 10% valid
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainvalid, y_trainvalid, test_size=0.2, shuffle=True, random_state=777
    )
    if data_type == "train":
        return X_train, y_train
    elif data_type == "valid":
        return X_valid, y_valid
    else:
        assert data_type == "all"
        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)


def get_model_space():
    conv_layers = lambda m: [
        {"Layer_type": "conv2d", "filters": 16 * m, "kernel_size": (1, 1), "activation": "relu"},
        {"Layer_type": "conv2d", "filters": 16 * m, "kernel_size": (3, 3), "activation": "relu"},
        {"Layer_type": "conv2d", "filters": 16 * m, "kernel_size": (1, 1), "activation": "tanh"},
        {"Layer_type": "conv2d", "filters": 16 * m, "kernel_size": (3, 3), "activation": "tanh"},
        {"Layer_type": "identity"},
    ]
    model_space = architect.ModelSpace.from_dict(
        [
            # conv layer 1
            conv_layers(1),
            # conv layer 2
            conv_layers(2),
            # pooling
            [
                {"Layer_type": "maxpool2d", "pool_size": (2, 2)},
                {"Layer_type": "avgpool2d", "pool_size": (2, 2)},
                {"Layer_type": "identity"},
            ],
            # flatten
            [
                {"Layer_type": "flatten"},
                {"Layer_type": "globalmaxpool2d"},
                {"Layer_type": "globalavgpool2d"},
            ],
            # dropout
            [
                {"Layer_type": "dropout", "rate": 0.5},
                {"Layer_type": "dropout", "rate": 0.1},
                {"Layer_type": "identity"},
            ],
            # dense
            [
                {"Layer_type": "dense", "units": 32, "activation": "relu"},
                {"Layer_type": "dense", "units": 16, "activation": "relu"},
                {"Layer_type": "identity"},
            ],
        ]
    )
    return model_space

def get_verbose_specs():
    # First, define the components we need to use
    type_dict = {
        "controller_type": "GeneralController",
        "modeler_type": "KerasModelBuilder",
        "knowledge_fn_type": "zero",
        "reward_fn_type": "SparseCategoricalReward",
        "manager_type": "GeneralManager",
        "env_type": "ControllerTrainEnv",
    }

    model_space = get_model_space()
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data()

    specs = {
        "model_space": model_space,
        "controller": {
            #"share_embedding": {1: 0},
            "with_skip_connection": False,
            "num_input_blocks": 1,
            "lstm_size": 32,
            "lstm_num_layers": 1,
            "kl_threshold": 0.01,
            "train_pi_iter": 10,
            "optim_algo": "adam",
            "temperature": 2.0,
            "lr_init": 0.001,
            "tanh_constant": 1.5,
            "buffer_size": 5,
            "batch_size": 5,
        },
        "model_builder": {
            "batch_size": 128,
            "inputs_op": [architect.Operation("input", shape=(8, 8, 1), name="input")],
            "outputs_op": [architect.Operation("dense", units=10, activation="softmax")],
            "model_compile_dict": {
                "loss": "categorical_crossentropy",
                "optimizer": "adam",
            },
        },
        "knowledge_fn": {"data": None, "params": {}},
        "reward_fn": {"method": accuracy},
        "manager": {
            "data": {
                "train_data": DataToParse(obj=get_data, kws={"data_type": "train"}),
                "validation_data": DataToParse(
                    obj=get_data, kws={"data_type": "valid"}
                ),
            },
            "params": {
                "epochs": 25,
                "child_batchsize": 128,
                "store_fn": "minimal",
                "working_dir": "./outputs/",
                "verbose": 0,
            },
        },
        "train_env": {
            "max_episode": 30,
            "max_step_per_ep": 3,
            "working_dir": "./outputs/",
            "time_budget": "00:10:00",  # 5min
            "with_input_blocks": False,
            "with_skip_connection": False,
        },
    }
    return type_dict, specs


def accuracy(y_true, y_score):
    return np.mean(y_true.argmax(axis=1) == y_score.argmax(axis=1))


def test_amber_config_io():
    tempdir = tempfile.TemporaryDirectory()
    type_dict, specs = get_verbose_specs()
    # save a config
    amber_config = AmberSpecifications(name="digits", types=type_dict)
    amber_config.from_dict(specs)
    amber_config.save_pickle(os.path.join(tempdir.name, "conv2d_specs.pkl"))
    # test load
    with open(os.path.join(tempdir.name, "conv2d_specs.pkl"), "rb") as f:
        amber_config = pickle.load(f)
    # NOTE: if you want to change the content, you can export the dict by d=amber_config.to_dict(), modify the dict,
    # then add it back by amber_config.from_dict(d)
    d = amber_config.to_dict()
    d['train_env']['max_episode'] = 20
    amber_config = amber_config.from_dict(d)

    tempdir.cleanup()
    return amber_config


def test_amber_wrapper_build():
    with open(CONFIG_FP, "rb") as f:
        amber_config = pickle.load(f)
    amb = Amber(types=amber_config.types, specs=amber_config.to_dict())
    assert issubclass(type(amb.controller), architect.base.BaseSearcher)
    assert issubclass(type(amb.env), architect.base.BaseSearchEnvironment)
    assert issubclass(type(amb.manager), architect.base.BaseNetworkManager)
    assert issubclass(type(amb.manager.model_fn), modeler.base.BaseModelBuilder)
