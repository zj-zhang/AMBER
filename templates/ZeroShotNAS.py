#!/usr/bin/env python

"""
Testing for ZeroShotNAS, aka data description driven nas
ZZ
May 14, 2020
"""

import tensorflow as tf
import numpy as np
import keras.backend as K
import os
import logging
import pickle

from amber.modeler import EnasCnnModelBuilder
from amber.architect.controller import ZeroShotController
from amber.architect.model_space import State, ModelSpace
from amber.architect.common_ops import count_model_params

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam, SGD

from amber.architect.manager import EnasManager
from amber.architect.train_env import MultiManagerEnasEnvironment
from amber.architect.reward import LossAucReward
from amber.plots import plot_controller_hidden_states
from amber.utils import run_from_ipython
from amber.utils.logging import setup_logger
from amber.utils.data_parser import get_data_from_simdata


def get_controller(model_space, session, data_description_len=3):
    with tf.device("/cpu:0"):
        controller = ZeroShotController(
            data_description_len=data_description_len,
            model_space=model_space,
            session=session,
            share_embedding={i:0 for i in range(1, len(model_space))},
            with_skip_connection=True,
            skip_weight=0.5,
            skip_target=0.2,
            lstm_size=64,
            lstm_num_layers=1,
            kl_threshold=0.01,
            train_pi_iter=10,
            #optim_algo=SGD(lr=lr, momentum=True),
            optim_algo='adam',
            temperature=2.,
            tanh_constant=1.5,
            buffer_type="MultiManager",
            buffer_size=5,
            batch_size=20
        )
        controller.buffer.rescale_advantage_by_reward = False
    return controller


def get_model_space(out_filters=16, num_layers=3, num_pool=3):
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


def read_data():
    dataset1 = get_data_from_simdata(
        positive_file="./data/zero_shot/DensityEmbedding_prefix-MYC_known10_motifs-MYC_known10_min-1_max-10_mean-5_zeroProb-0_seqLength-1000_numSeqs-10000.simdata",
        negative_file="./data/zero_shot/EmptyBackground_prefix-empty_bg_seqLength-1000_numSeqs-10000.simdata",
        targets=["MYC"])
    dataset2 = get_data_from_simdata(
        positive_file="./data/zero_shot/DensityEmbedding_prefix-CTCF_known1_motifs-CTCF_known1_min-1_max-1_mean-1_zeroProb-0_seqLength-1000_numSeqs-10000.simdata",
        negative_file="./data/zero_shot/EmptyBackground_prefix-empty_bg_seqLength-1000_numSeqs-10000.simdata",
        targets=["CTCF"])

    num_seqs = 20000
    train_idx = np.arange(0, int(num_seqs*0.8))
    val_idx = np.arange(int(num_seqs*0.8), int(num_seqs*0.9) )
    test_idx = np.arange(int(num_seqs*0.9), num_seqs )
    dictionarize = lambda dataset: {
            "train": (dataset[0][train_idx], dataset[1][train_idx]),
            "val": (dataset[0][val_idx], dataset[1][val_idx]),
            "test": (dataset[0][test_idx], dataset[1][test_idx]),
            }

    dataset1 = dictionarize(dataset1)
    dataset2 = dictionarize(dataset2)

    return dataset1, dataset2


def get_manager(train_data, val_data, controller, model_space, wd, data_description, dag_name, verbose=2):
    input_node = State('input', shape=(1000, 4), name="input", dtype=tf.float32)
    output_node = State('dense', units=1, activation='sigmoid')
    model_compile_dict = {
        'loss': 'binary_crossentropy',
        'optimizer': 'adam',
    }
    session = controller.session

    reward_fn = LossAucReward(method='auc')
    
    child_batch_size = 500
    model_fn = EnasCnnModelBuilder(
        dag_func='EnasConv1DwDataDescrption',
        batch_size=child_batch_size,
        session=session,
        model_space=model_space,
        inputs_op=[input_node],
        output_op=[output_node],
        num_layers=len(model_space),
        l1_reg=1e-8,
        l2_reg=5e-7,
        model_compile_dict=model_compile_dict,
        controller=controller,
        dag_kwargs={
            'stem_config':{
                'flatten_op': 'flatten',
                'fc_units': 10
                },
            'name': dag_name,
            'data_description': data_description
            }
    )
    
    manager = EnasManager(
        train_data=train_data,
        validation_data=val_data,
        epochs=1,
        child_batchsize=child_batch_size,
        reward_fn=reward_fn,
        model_fn=model_fn,
        store_fn='minimal',
        model_compile_dict=model_compile_dict,
        working_dir=wd,
        verbose=verbose
        )
    return manager



def main():
    wd = "./outputs/zero_shot/"
    verbose = 1
    model_space = get_model_space()
    session = tf.Session()
    controller = get_controller(model_space=model_space, session=session, data_description_len=2)

    dataset1, dataset2 = read_data()
    dfeatures = np.array([[1,0], [0,1]])  # one-hot encoding
    manager1 = get_manager(train_data=dataset1['train'], val_data=dataset1['val'], controller=controller,
            model_space=model_space, wd=wd, data_description=dfeatures[[0]], dag_name="EnasDAG1", verbose=verbose)
    manager2 = get_manager(train_data=dataset2['train'], val_data=dataset2['val'], controller=controller,
            model_space=model_space, wd=wd, data_description=dfeatures[[1]], dag_name="EnasDAG2", verbose=verbose)


    logger = setup_logger(wd, verbose_level=logging.DEBUG)
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

    env = MultiManagerEnasEnvironment(
        data_descriptive_features= dfeatures,
        controller=controller,
        manager=[manager1, manager2],
        logger=logger,
        max_episode=300,
        max_step_per_ep=100,
        working_dir=wd,
        time_budget="2:00:00",
        with_input_blocks=False,
        with_skip_connection=True,
    )

    try:
        env.train()
    except KeyboardInterrupt:
        print("user interrupted training")
        pass
    controller.save_weights(os.path.join(wd, "controller_weights.h5"))
    plot_controller_hidden_states(controller, "%s/controller_states.png" % wd)


if __name__ == "__main__":
    if not run_from_ipython():
        main()
