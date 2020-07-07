#!/usr/bin/env python

"""
Testing for ZeroShotNAS, aka data description driven nas
ZZ
May 14, 2020
"""

import tensorflow as tf
import numpy as np
import os
import copy
import logging
import pickle
import pandas as pd
import argparse

from amber.modeler import EnasCnnModelBuilder
from amber.architect.controller import ZeroShotController
from amber.architect.model_space import State, ModelSpace
from amber.architect.common_ops import count_model_params, unpack_data

from amber.architect.train_env import MultiManagerEnvironment, ParallelMultiManagerEnvironment
from amber.architect.reward import LossAucReward, LossReward
from amber.plots import plot_controller_hidden_states
from amber.utils import run_from_ipython, get_available_gpus
from amber.utils.logging import setup_logger
from amber.utils.data_parser import get_data_from_simdata

from amber.modeler.modeler import build_sequential_model, KerasModelBuilder
from amber.architect.manager import GeneralManager, DistributedGeneralManager
from amber.architect.model_space import get_layer_shortname

from amber.utils.sequences import EncodedGenome
from amber.utils.sequences import EncodedHDF5Genome
from amber.utils.sampler import BatchedBioIntervalSequence


def get_controller(model_space, session, data_description_len=3):
    with tf.device("/cpu:0"):
        controller = ZeroShotController(
            data_description_config={
                "length": data_description_len,
                "hidden_layer": {"units":8, "activation": "relu"},
                "regularizer": {"l1":1e-8 }
                },
            model_space=model_space,
            session=session,
            with_skip_connection=False,
            skip_weight=None,
            skip_target=0.2,
            lstm_size=32,
            lstm_num_layers=1,
            kl_threshold=0.01,
            train_pi_iter=100,
            optim_algo='adam',
            temperature=2.,
            tanh_constant=1.5,
            buffer_type="MultiManager",
            buffer_size=5,
            batch_size=5,
            use_ppo_loss=True,
            rescale_advantage_by_reward=True
        )
    return controller


def get_model_space_common():

    # Setup and params.
    state_space = ModelSpace()
    default_params = {"kernel_initializer": "glorot_uniform",
                      "activation": "relu"}
    param_list = [
            # Block 1:
            [{"filters": 100, "kernel_size": 8},
             {"filters": 100, "kernel_size": 14},
             {"filters": 100, "kernel_size": 20}],
            # Block 2:
            [{"filters": 200, "kernel_size": 8},
             {"filters": 200, "kernel_size": 14},
             {"filters": 200, "kernel_size": 20}],
            # Block 3:
            [{"filters": 300, "kernel_size": 8},
             {"filters": 300, "kernel_size": 14},
             {"filters": 300, "kernel_size": 20}],
        ]

    # Build state space.
    conv_seen = 0
    for i in range(len(param_list)):
        # Build conv states for this layer.
        conv_states = [State("Identity")]
        for j in range(len(param_list[i])):
            d = copy.deepcopy(default_params)
            for k, v in param_list[i][j].items():
                d[k] = v
            conv_states.append(State('conv1d', name="conv{}".format(conv_seen), **d))
        state_space.add_layer(conv_seen * 2, conv_states)
        conv_seen += 1

        # Add pooling states.
        if i < len(param_list) - 1:
            pool_states = [State('Identity'),
                           State('maxpool1d', pool_size=4, strides=4),
                           State('avgpool1d', pool_size=4, strides=4)]
        else:
            pool_states = [State('Flatten'),
                           State('GlobalMaxPool1D'),
                           State('GlobalAvgPool1D')]
        state_space.add_layer(conv_seen * 2 - 1, pool_states)

    # Add final classifier layer.
    state_space.add_layer(conv_seen * 2, [
            State('Dense', units=30, activation='relu'),
            State('Dense', units=100, activation='relu'),
            State('Identity')
        ])
    return state_space


def get_model_space_with_long_model():
    # Allow option for convolution to be smaller but have two layers instead of just 1.

    # Setup and params.
    state_space = ModelSpace()
    default_params = {"kernel_initializer": "glorot_uniform",
                      "activation": "relu"}
    param_list = [
            # Block 1:
            [{"filters": 100, "kernel_size": 8},
             {"filters": 100, "kernel_size": 14},
             {"filters": 100, "kernel_size": 20}],
            # Block 2:
            [{"filters": 200, "kernel_size": 8},
             {"filters": 200, "kernel_size": 14},
             {"filters": 200, "kernel_size": 20}],
            # Block 3:
            [{"filters": 300, "kernel_size": 8},
             {"filters": 300, "kernel_size": 14},
             {"filters": 300, "kernel_size": 20}]
        ]

    # Build state space.
    conv_seen = 0
    for i in range(len(param_list)):
        # Build conv states for this layer.
        conv_states = [State("Identity")]
        second_conv_states = [State("Identity")]
        for j in range(len(param_list[i])):
            d = copy.deepcopy(default_params)
            for k, v in param_list[i][j].items():
                d[k] = v
            conv_states.append(State('conv1d', name="conv{}.1".format(conv_seen), **d))
            d["kernel_size"] = d["kernel_size"] // 2
            conv_states.append(State('conv1d', name="conv{}.1".format(conv_seen), **d))
            second_conv_states.append(State("conv1d", name="conv{}.2".format(conv_seen), **d))
        state_space.add_layer(conv_seen * 3, conv_states)
        state_space.add_layer(conv_seen * 3 + 1, second_conv_states)
        conv_seen += 1

        # Add pooling states.
        if i < len(param_list) - 1:
            pool_states = [State('Identity'),
                           State('maxpool1d', pool_size=4, strides=4),
                           State('avgpool1d', pool_size=4, strides=4)]
        else:
            pool_states = [State('Flatten'),
                           State('GlobalMaxPool1D'),
                           State('GlobalAvgPool1D')]
        state_space.add_layer(conv_seen * 3 - 1, pool_states)

    # Add final classifier layer.
    state_space.add_layer(conv_seen * 2, [
            State('Dense', units=30, activation='relu'),
            State('Dense', units=100, activation='relu'),
            State('Identity')
        ])
    return state_space


def get_model_space_with_dilation():

    # Setup and params.
    state_space = ModelSpace()
    default_params = {"kernel_initializer": "glorot_uniform",
                      "activation": "relu"}
    param_list = [
            # Block 1:
            [{"filters": 100, "kernel_size": 8},
             {"filters": 100, "kernel_size": 14},
             {"filters": 100, "kernel_size": 20},
             {"filters": 100, "kernel_size": 8, "dilation_rate": 2},
             {"filters": 100, "kernel_size": 14, "dilation_rate": 2},
             {"filters": 100, "kernel_size": 20, "dilation_rate": 2}],
            # Block 2:
            [{"filters": 200, "kernel_size": 8},
             {"filters": 200, "kernel_size": 14},
             {"filters": 200, "kernel_size": 20},
             {"filters": 200, "kernel_size": 8, "dilation_rate": 2},
             {"filters": 200, "kernel_size": 14, "dilation_rate": 2},
             {"filters": 200, "kernel_size": 20, "dilation_rate": 2}],
            # Block 3:
            [{"filters": 300, "kernel_size": 8},
             {"filters": 300, "kernel_size": 14},
             {"filters": 300, "kernel_size": 20},
             {"filters": 300, "kernel_size": 8, "dilation_rate": 2},
             {"filters": 300, "kernel_size": 14, "dilation_rate": 2},
             {"filters": 300, "kernel_size": 20, "dilation_rate": 2}],
        ]

    # Build state space.
    conv_seen = 0
    for i in range(len(param_list)):
        # Build conv states for this layer.
        conv_states = [State("Identity")]
        for j in range(len(param_list[i])):
            d = copy.deepcopy(default_params)
            for k, v in param_list[i][j].items():
                d[k] = v
            conv_states.append(State('conv1d', name="conv{}".format(conv_seen), **d))
        state_space.add_layer(conv_seen * 2, conv_states)
        conv_seen += 1

        # Add pooling states.
        if i < len(param_list) - 1:
            pool_states = [State('Identity'),
                           State('maxpool1d', pool_size=4, strides=4),
                           State('avgpool1d', pool_size=4, strides=4)]
        else:
            pool_states = [State('Flatten'),
                           State('GlobalMaxPool1D'),
                           State('GlobalAvgPool1D')]
        state_space.add_layer(conv_seen * 2 - 1, pool_states)

    # Add final classifier layer.
    state_space.add_layer(conv_seen * 2, [
            State('Dense', units=30, activation='relu'),
            State('Dense', units=100, activation='relu'),
            State('Identity')
        ])
    return state_space


def get_model_space_with_long_model_and_dilation():
    # Allow option for convolution to be smaller but have two layers instead of just 1.

    # Setup and params.
    state_space = ModelSpace()
    default_params = {"kernel_initializer": "glorot_uniform",
                      "activation": "relu"}
    param_list = [
            # Block 1:
            [{"filters": 100, "kernel_size": 8},
             {"filters": 100, "kernel_size": 14},
             {"filters": 100, "kernel_size": 20},
             {"filters": 100, "kernel_size": 8, "dilation_rate": 2},
             {"filters": 100, "kernel_size": 14, "dilation_rate": 2},
             {"filters": 100, "kernel_size": 20, "dilation_rate": 2}],
            # Block 2:
            [{"filters": 200, "kernel_size": 8},
             {"filters": 200, "kernel_size": 14},
             {"filters": 200, "kernel_size": 20},
             {"filters": 200, "kernel_size": 8, "dilation_rate": 2},
             {"filters": 200, "kernel_size": 14, "dilation_rate": 2},
             {"filters": 200, "kernel_size": 20, "dilation_rate": 2}],
            # Block 3:
            [{"filters": 300, "kernel_size": 8},
             {"filters": 300, "kernel_size": 14},
             {"filters": 300, "kernel_size": 20},
             {"filters": 300, "kernel_size": 8, "dilation_rate": 2},
             {"filters": 300, "kernel_size": 14, "dilation_rate": 2},
             {"filters": 300, "kernel_size": 20, "dilation_rate": 2}],
        ]

    # Build state space.
    conv_seen = 0
    for i in range(len(param_list)):
        # Build conv states for this layer.
        conv_states = [State("Identity")]
        second_conv_states = [State("Identity")]
        for j in range(len(param_list[i])):
            d = copy.deepcopy(default_params)
            for k, v in param_list[i][j].items():
                d[k] = v
            conv_states.append(State('conv1d', name="conv{}.1".format(conv_seen), **d))
            d["kernel_size"] = d["kernel_size"] // 2
            conv_states.append(State('conv1d', name="conv{}.1".format(conv_seen), **d))
            second_conv_states.append(State("conv1d", name="conv{}.2".format(conv_seen), **d))
        state_space.add_layer(conv_seen * 3, conv_states)
        state_space.add_layer(conv_seen * 3 + 1, second_conv_states)
        conv_seen += 1

        # Add pooling states.
        if i < len(param_list) - 1:
            pool_states = [State('Identity'),
                           State('maxpool1d', pool_size=4, strides=4),
                           State('avgpool1d', pool_size=4, strides=4)]
        else:
            pool_states = [State('Flatten'),
                           State('GlobalMaxPool1D'),
                           State('GlobalAvgPool1D')]
        state_space.add_layer(conv_seen * 3 - 1, pool_states)

    # Add final classifier layer.
    state_space.add_layer(conv_seen * 2, [
            State('Dense', units=30, activation='relu'),
            State('Dense', units=100, activation='relu'),
            State('Identity')
        ])
    return state_space


def get_manager_distributed(train_data, val_data, controller, model_space, wd, data_description, verbose=0,
                            devices=None, train_data_kwargs=None, **kwargs):
    reward_fn = LossAucReward(method='auc')
    input_node = State('input', shape=(1000, 4), name="input", dtype='float32')
    output_node = State('dense', units=1, activation='sigmoid')
    model_compile_dict = {
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
        'metrics': ['acc']
    }
    #mb = KerasModelBuilder(inputs=input_node, outputs=output_node, model_compile_dict=model_compile_dict, model_space=model_space,  gpus=devices)
    mb = KerasModelBuilder(inputs=input_node, outputs=output_node, model_compile_dict=model_compile_dict, model_space=model_space)
    manager = DistributedGeneralManager(
        devices=devices,
        train_data_kwargs = train_data_kwargs or None,
        train_data=train_data,
        validation_data=unpack_data(val_data, unroll_generator=True),
        epochs=50,
        child_batchsize=1000,
        reward_fn=reward_fn,
        model_fn=mb,
        store_fn='model_plot',
        model_compile_dict=model_compile_dict,
        working_dir=wd,
        verbose=verbose,
        save_full_model=False,
        model_space=model_space
    )
    return manager


def get_manager_common(train_data, val_data, controller, model_space, wd, data_description, verbose=2, n_feats=1, **kwargs):
    input_node = State('input', shape=(1000, 4), name="input", dtype='float32')
    output_node = State('dense', units=n_feats, activation='sigmoid')
    model_compile_dict = {
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
        'metrics': ['acc']
    }
    session = controller.session

    reward_fn = LossAucReward(method='auc')
    gpus = get_available_gpus()
    num_gpus = len(gpus)
    mb = KerasModelBuilder(inputs=input_node, outputs=output_node, model_compile_dict=model_compile_dict, model_space=model_space, gpus=num_gpus)
 
    child_batch_size = 500*num_gpus
    manager = GeneralManager(
        train_data=train_data,
        validation_data=val_data,
        epochs=50,
        child_batchsize=child_batch_size,
        reward_fn=reward_fn,
        model_fn=mb,
        store_fn='minimal',
        model_compile_dict=model_compile_dict,
        working_dir=wd,
        verbose=verbose,
        save_full_model=False,
        model_space=model_space,
        fit_kwargs={'workers': 8, 'max_queue_size': 100, 'use_multiprocessing':False}
    )
    return manager


def get_samples_controller(dfeatures, controller, model_space, T=100):
    res = []
    for i in range(len(dfeatures)):
        dfeature = dfeatures[[i]]
        prob_arr = [ np.zeros((T, len(model_space[i]))) for i in range(len(model_space)) ]
        for t in range(T):
            a, p = controller.get_action(dfeature)
            for i, p in enumerate(p):
                prob_arr[i][t] = p.flatten()
        res.append(prob_arr)
    return res


def convert_to_dataframe(res, model_space, data_names):
    probs = []
    layer = []
    description = []
    operation = []
    for i in range(len(res)):
        for j in range(len(model_space)):
            for k in range(len(model_space[j])):
                o = get_layer_shortname(model_space[j][k])
                p = res[i][j][:, k]
                # extend the array
                T = p.shape[0]
                probs.extend(p)
                description.extend([data_names[i]]*T)
                layer.extend([j]*T)
                operation.extend([o]*T)
    df = pd.DataFrame({
        'description':description,
        'layer': layer,
        'operation': operation,
        'prob': probs
        })
    return df


def reload_trained_controller(arg):
    wd = arg.wd #wd = "./outputs/zero_shot/"
    #model_space = get_model_space_common()
    model_space = get_model_space_with_long_model_and_dilation()
    try:
        session = tf.Session()
    except AttributeError:
        session = tf.compat.v1.Session()
    controller = get_controller(model_space=model_space, session=session, data_description_len=2)
    controller.load_weights(os.path.join(wd, "controller_weights.h5"))

    dfeatures = np.array([[1,0], [0,1]])  # one-hot encoding
    res = get_samples_controller(dfeatures, controller, model_space, T=1000)
   
    import seaborn as sns
    import matplotlib.pyplot as plt
    df = convert_to_dataframe(res, model_space, data_names=['MYC_known10', 'CTCF_known1'])

    for i in range(len(model_space)):
        sub_df = df.loc[ (df.layer==i) ]
        plt.clf()
        plt.tight_layout()
        ax = sns.boxplot(x="operation", y="prob",
            hue="description", palette=["m", "g"],
            data=sub_df)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.savefig(os.path.join(wd, "layer_%i.png"%i), bbox_inches="tight")
    return res


def train_nas(arg):
    dfeature_names = list()
    with open(arg.dfeature_name_file, "r") as read_file:
        for line in read_file:
            line = line.strip()
            if line:
                dfeature_names.append(line)
    wd = arg.wd
    verbose = 1
    model_space = get_model_space_common()
    try:
        session = tf.Session()
    except AttributeError:
        session = tf.compat.v1.Session()

    controller = get_controller(model_space=model_space, session=session, data_description_len=len(dfeature_names))
    # Re-load previously saved weights, if specified
    if arg.resume:
        try:
            controller.load_weights(os.path.join(wd, "controller_weights.h5"))
            print("loaded existing weights")
        except Exception as e:
            print("cannot load controller weights because of %s"%e)
    # Load in datasets and configurations for them.
    if arg.config_file.endswith("tsv"):
        sep = "\t"
    else:
        sep = ","
    configs = pd.read_csv(arg.config_file, sep=sep)
    tmp = list(configs.columns) # Because pandas doesn't have infer quotes...
    if any(["\"" in x for x in tmp]):
        configs = pd.read_csv(arg.config_file, sep=sep, quoting=2)
        print("Re-read with quoting")
    configs = configs.to_dict(orient='index')
    # Get available gpus for parsing to DistributedManager
    gpus = get_available_gpus()
    gpus_ = gpus * len(configs)

    # Build genome. This only works under the assumption that all configs use same genome.
    k = list(configs.keys())[0]
    genome = EncodedHDF5Genome(input_path=arg.genome_file, in_memory=False)

    manager_getter = get_manager_distributed if arg.parallel else get_manager_common
    config_keys = list()
    for i, k in enumerate(configs.keys()):
        # Build datasets for train/test/validate splits.
        for x in ["train", "validate"]:
            if x == "train":
                n = arg.n_train
            elif x == "test":
                n = arg.n_test
            elif x == "validate":
                n = arg.n_validate
            else:
                s = "Unknown mode: {}".format(x)
                raise ValueError(s)
            d = {
                        'example_file': configs[k][x + "_file"],
                        'reference_sequence': arg.genome_file,
                        'batch_size': 500, 
                        'seed': 1337, 
                        'shuffle': True,
                        'n_examples': n,
                        'pad': 400
                    }
            if x == "train":
                configs[k][x] = BatchedBioIntervalSequence
                configs[k]['train_data_kwargs'] = d
            else:
                configs[k][x] = BatchedBioIntervalSequence(**d)

        # Build covariates and manager.
        configs[k]["dfeatures"] = np.array(
            [configs[k][x] for x in dfeature_names]) # TODO: Make cols dynamic.
        configs[k]["manager"] = manager_getter(
            devices=[gpus_[i]],
            train_data=configs[k]["train"],
            val_data=configs[k]["validate"],
            controller=controller,
            model_space=model_space,
            wd=os.path.join(wd, "manager_%s"%k),
            data_description=configs[k]["dfeatures"],
            dag_name="AmberDAG{}".format(k),
            verbose=0,
            n_feats=configs[k]["n_feats"],
            train_data_kwargs=configs[k]['train_data_kwargs']
            )
        config_keys.append(k)

    logger = setup_logger(wd, verbose_level=logging.INFO)

    env = ParallelMultiManagerEnvironment(
        processes=len(gpus) if arg.parallel else 1,
        data_descriptive_features=np.stack([configs[k]["dfeatures"] for k in config_keys]),
        controller=controller,
        manager=[configs[k]["manager"] for k in config_keys],
        logger=logger,
        max_episode=200,
        max_step_per_ep=15,
        working_dir=wd,
        time_budget="150:00:00",
        save_controller_every=1,
        with_input_blocks=False,
        with_skip_connection=False,
    )

    try:
        env.train()
    except KeyboardInterrupt:
        print("user interrupted training")
        pass
    controller.save_weights(os.path.join(wd, "controller_weights.h5"))


if __name__ == "__main__":
    if not run_from_ipython():
        parser = argparse.ArgumentParser(description="experimental zero-shot nas")
        parser.add_argument("--analysis", type=str, choices=['train', 'reload'], required=True, help="analysis type")
        parser.add_argument("--wd", type=str, default="./outputs/zero_shot/", help="working dir")
        parser.add_argument("--parallel", default=False, action="store_true", help="Use parallel")
        parser.add_argument("--resume", default=False, action="store_true", help="resume previous run")
        parser.add_argument("--config-file", type=str, required=True, help="Path to the config file to use.")
        parser.add_argument("--genome-file", type=str, required=True, help="Path to genome file to use.")
        parser.add_argument("--dfeature-name-file", type=str, required=True, help="Path to file with dataset feature names listed one per line.")
        parser.add_argument("--n-test", type=int, required=True, help="Number of test examples.")
        parser.add_argument("--n-train", type=int, required=True, help="Number of train examples.")
        parser.add_argument("--n-validate", type=int, required=True, help="Number of validation examples.")

        arg = parser.parse_args()

        if arg.analysis == "train":
            train_nas(arg)
        elif arg.analysis == "reload":
            reload_trained_controller(arg)
        else:
            raise Exception("Unknown analysis type: %s"% arg.analysis)
