#!/usr/bin/env python

"""
Testing for ZeroShotNAS, aka data description driven nas
ZZ
May 14, 2020
"""

import tensorflow as tf
import numpy as np
import os
import logging
import pickle
import pandas as pd
import argparse

from amber.modeler import EnasCnnModelBuilder
from amber.architect.controller import ZeroShotController
from amber.architect.model_space import State, ModelSpace
from amber.architect.common_ops import count_model_params

#from amber.architect.manager import EnasManager
from amber.architect.train_env import MultiManagerEnvironment
from amber.architect.reward import LossAucReward, LossReward
from amber.plots import plot_controller_hidden_states
from amber.utils import run_from_ipython
from amber.utils.logging import setup_logger
from amber.utils.data_parser import get_data_from_simdata

from amber.modeler.modeler import build_sequential_model
from amber.architect.manager import GeneralManager
from amber.architect.model_space import get_layer_shortname

from amber.utils.sequences import EncodedGenome
from amber.utils.sequences import EncodedHDF5Genome
from amber.utils.sampler import BatchedBioIntervalSequence


def get_controller(model_space, session, data_description_len=3):
    with tf.device("/cpu:0"):
        controller = ZeroShotController(
            data_description_len=data_description_len,
            model_space=model_space,
            session=session,
            #share_embedding={i:0 for i in range(1, len(model_space))},
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
        #controller.buffer.rescale_advantage_by_reward = False
        controller.buffer.rescale_advantage_by_reward = True
    return controller


def get_model_space_enas(out_filters=16, num_layers=3, num_pool=3):
    state_space = ModelSpace()
    if num_pool == 4:
        expand_layers = [num_layers//4-1, num_layers//4*2-1, num_layers//4*3-1]
    elif num_pool == 3:
        expand_layers = [num_layers//3-1, num_layers//3*2-1]
    else:
        raise Exception("Unsupported pooling num: %i"%num_pool)
    for i in range(num_layers):
        state_space.add_layer(i, [
            State('conv1d', filters=out_filters, kernel_size=4, activation='relu'),
            State('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
            State('conv1d', filters=out_filters, kernel_size=12, activation='relu'),
            # max/avg pool has underlying 1x1 conv
            State('maxpool1d', filters=out_filters, pool_size=4, strides=1),
            State('avgpool1d', filters=out_filters, pool_size=4, strides=1),
            State('identity', filters=out_filters),
      ])
        if i in expand_layers:
            out_filters *= 2
    return state_space


def get_model_space_common():
    state_space = ModelSpace()
    state_space.add_layer(0, [
        State('conv1d', filters=3, kernel_size=8, kernel_initializer='glorot_uniform', activation='relu',
              name="conv1"),
        State('conv1d', filters=3, kernel_size=14, kernel_initializer='glorot_uniform', activation='relu',
              name="conv1"),
        State('conv1d', filters=3, kernel_size=20, kernel_initializer='glorot_uniform', activation='relu',
              name="conv1"),
    ])
    state_space.add_layer(1, [
        State('Identity'),
        State('maxpool1d', pool_size=8, strides=8),
        State('avgpool1d', pool_size=8, strides=8),

    ])
    state_space.add_layer(2, [
        State('Flatten'),
        State('GlobalMaxPool1D'),
        State('GlobalAvgPool1D'),
    ])
    state_space.add_layer(3, [
        State('Dense', units=3, activation='relu'),
        State('Dense', units=10, activation='relu'),
        State('Identity')
    ])
    return state_space


def get_manager_common(train_data, val_data, controller, model_space, wd, data_description, verbose=2, n_feats=1, **kwargs):
    input_node = State('input', shape=(1000, 4), name="input", dtype='float32')
    output_node = State('dense', units=n_feats, activation='sigmoid')
    model_compile_dict = {
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
        'metrics': ['acc']
    }
    session = controller.session

    #reward_fn = LossAucReward(method='auc')
    reward_fn = LossReward() # TODO: IMplement LossAucReward for generator.

    child_batch_size = 500
    # TODO: convert functions in `_keras_modeler.py` to classes, and wrap up this Lambda function step
    model_fn = lambda model_arc: build_sequential_model(
        model_states=model_arc, input_state=input_node, output_state=output_node, model_compile_dict=model_compile_dict,
        model_space=model_space)
    manager = GeneralManager(
        train_data=train_data,
        validation_data=val_data,
        epochs=30,
        child_batchsize=child_batch_size,
        reward_fn=reward_fn,
        model_fn=model_fn,
        store_fn='model_plot',
        model_compile_dict=model_compile_dict,
        working_dir=wd,
        verbose=0,
        save_full_model=False,
        model_space=model_space
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
    model_space = get_model_space_common()
    session = tf.Session()
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
    verbose = 2
    model_space = get_model_space_common()
    session = tf.Session()
    controller = get_controller(model_space=model_space, session=session, data_description_len=len(dfeature_names))

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

    # Build genome. This only works under the assumption that all configs use same genome.
    k = list(configs.keys())[0]
    genome = EncodedHDF5Genome(input_path=arg.genome_file, in_memory=False)
    #genome = EncodedGenome(input_path=configs[k]["genome_file"], in_memory=True)


    config_keys = list()
    for k in configs.keys():
        # Build datasets for train/test/validate splits.
        for x in ["train", "test", "validate"]:
            if x == "train":
                n = arg.n_train
            elif x == "test":
                n = arg.n_test
            elif x == "validate":
                n = arg.n_validate
            else:
                s = "Unknown mode: {}".format(x)
                raise ValueError(s)
            #in_memory=(x == "train")),
            configs[k][x] = BatchedBioIntervalSequence(
                configs[k][x + "_file"],
                genome,
                batch_size=500, seed=1337, shuffle=(x == "train"),
                n_examples=n)
            configs[k][x].set_pad(400) # 1000 total bp = 200 + 400 * 2

        # Build covariates and manager.
        configs[k]["dfeatures"] = np.array(
            [configs[k][x] for x in dfeature_names]) # TODO: Make cols dynamic.
        #print(configs[k]["dfeatures"])
        configs[k]["manager"] = get_manager_common(
            train_data=configs[k]["train"],
            val_data=configs[k]["validate"],
            controller=controller,
            model_space=model_space,
            wd=wd,
            data_description=configs[k]["dfeatures"],
            dag_name="AmberDAG{}".format(k),
            verbose=verbose,
            n_feats=configs[k]["n_feats"])
        config_keys.append(k)

    logger = setup_logger(wd, verbose_level=logging.INFO)

    env = MultiManagerEnvironment(
        data_descriptive_features=np.stack([configs[k]["dfeatures"] for k in config_keys]),
        controller=controller,
        manager=[configs[k]["manager"] for k in config_keys],
        logger=logger,
        max_episode=200,
        max_step_per_ep=15,
        working_dir=wd,
        time_budget="1:00:00",
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
