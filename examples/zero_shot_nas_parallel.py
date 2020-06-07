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

from amber.architect.controller import ZeroShotController
from amber.architect.model_space import State, ModelSpace

#from amber.architect.manager import EnasManager
from amber.architect.train_env import MultiManagerEnvironment, ParallelMultiManagerEnvironment
from amber.architect.reward import LossAucReward
from amber.plots import plot_controller_hidden_states
from amber.utils import run_from_ipython, get_available_gpus
from amber.utils.logging import setup_logger

from amber.modeler.modeler import build_sequential_model
from amber.architect.manager import GeneralManager, DistributedGeneralManager
from amber.architect.model_space import get_layer_shortname

from zero_shot_nas import get_model_space_common, read_data, get_samples_controller, convert_to_dataframe


model_space = get_model_space_common()
input_node = State('input', shape=(1000, 4), name="input", dtype='float32')
output_node = State('dense', units=1, activation='sigmoid')
model_compile_dict = {
    'loss': 'binary_crossentropy',
    'optimizer': 'adam',
    'metrics': ['acc']
}

reward_fn = LossAucReward(method='auc')
child_batch_size = 500


def model_fn(model_arc):
    return build_sequential_model(
        model_states=model_arc, input_state=input_node, output_state=output_node, model_compile_dict=model_compile_dict,
        model_space=model_space)


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
            use_ppo_loss=True
        )
        controller.buffer.rescale_advantage_by_reward = False
    return controller


def get_manager_distributed(train_data, val_data, controller, model_space, wd, data_description, verbose=0,
                            devices=None, **kwargs):
    manager = DistributedGeneralManager(
        devices=devices,
        train_data=train_data,
        validation_data=val_data,
        epochs=5,
        child_batchsize=child_batch_size,
        reward_fn=reward_fn,
        model_fn=model_fn,
        store_fn='model_plot',
        model_compile_dict=model_compile_dict,
        working_dir=wd,
        verbose=verbose,
        save_full_model=False,
        model_space=model_space
    )
    return manager


def get_manager_mock(train_hist_fn, model_space, **kwargs):
    pass


def reload_trained_controller(arg):
    wd = arg.wd #wd = "./outputs/zero_shot/"
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
    wd = arg.wd #wd = "./outputs/zero_shot/"
    verbose = 1
    session = tf.Session()
    controller = get_controller(model_space=model_space, session=session, data_description_len=2)

    dataset1, dataset2 = read_data()
    dfeatures = np.array([[1,0], [0,1]])  # one-hot encoding
    gpus = get_available_gpus()
    if len(gpus) > 1:
        dev1 = gpus[:len(gpus)//2]
        dev2 = gpus[len(gpus)//2 :]
    else:
        dev1 = dev2 = None
    manager1 = get_manager_distributed(train_data=dataset1['train'], val_data=dataset1['val'], controller=controller,
            model_space=model_space, wd=os.path.join(wd, "manager1"),
            data_description=dfeatures[[0]], dag_name="EnasDAG1", verbose=0,
            devices=dev1)
    manager2 = get_manager_distributed(train_data=dataset2['train'], val_data=dataset2['val'], controller=controller,
            model_space=model_space, wd=os.path.join(wd, "manager2"),
            data_description=dfeatures[[1]], dag_name="EnasDAG2", verbose=0,
            devices=dev2)

    logger = setup_logger(wd, verbose_level=logging.INFO)
    try:
        # only for enas
        vars_list1 = [v for v in tf.trainable_variables() if v.name.startswith(manager1.model_fn.dag.name)]
        vars_list2 = [v for v in tf.trainable_variables() if v.name.startswith(manager2.model_fn.dag.name)]
        # remove optimizer related vars (e.g. momentum, rms)
        vars_list1 = [v for v in vars_list1 if not v.name.startswith("%s/compile"%manager1.model_fn.dag.name)]
        vars_list2 = [v for v in vars_list2 if not v.name.startswith("%s/compile"%manager2.model_fn.dag.name)]
        logger.info("total model1 params: %i" % count_model_params(vars_list1))
        logger.info("total model2 params: %i" % count_model_params(vars_list2))
        with open(os.path.join(wd,"tensor_vars.txt"), "w") as f:
            for v in vars_list1 + vars_list2:
                f.write("%s\t%i\n"%(v.name, int(np.prod(v.shape).value) ))
    except AttributeError:
        pass


    env = ParallelMultiManagerEnvironment(
        processes=2,
        data_descriptive_features= dfeatures,
        controller=controller,
        manager=[manager1, manager2],
        logger=logger,
        max_episode=12,
        max_step_per_ep=3,
        working_dir=wd,
        time_budget="12:00:00",
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

        arg = parser.parse_args()

        if arg.analysis == "train":
            train_nas(arg)
        elif arg.analysis == "reload":
            reload_trained_controller(arg)
        else:
            raise Exception("Unknown analysis type: %s"% arg.analysis)
