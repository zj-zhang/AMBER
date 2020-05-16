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
import logging
import pickle
from amber.utils import run_from_ipython


def get_controller(model_space, session, data_description_len=3):
    with tf.device("/cpu:0"):
        controller = ZeroShotController(
            data_description_len=data_description_len,
            model_space=model_space,
            session=session,
            share_embedding={i:0 for i in range(1, len(model_space))},
            with_skip_connection=True,
            skip_connection_unique_connection=False,
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


def main():
    model_space = get_model_space()
    session = tf.Session()
    controller = get_controller(model_space=model_space, session=session, data_description_len=3)

    d1 = [[0,0,0]]
    action, prob = controller.get_action(description_feature=d1)
    for _ in range(5):
        controller.store(prob=prob, action=action, reward=1, description=d1)

    d2 = [[1,1,1]]
    action, prob = controller.get_action(description_feature=d2)
    for _ in range(5):
        controller.store(prob=prob, action=action, reward=-1, description=d2)

    buffer = controller.buffer
    buffer.finish_path(model_space, 1, ".")

    g = buffer.get_data(bs=3)
    batch_data = next(g)

    p_batch, a_batch, ad_batch, nr_batch = \
        [batch_data[x] for x in ['prob', 'action', 'advantage', 'reward']]
    desc_batch = batch_data['description']
    feed_dict = {controller.input_arc[i]: a_batch[:, [i]]
                 for i in range(a_batch.shape[1])}
    feed_dict.update({controller.advantage: ad_batch})
    feed_dict.update({controller.old_probs[i]: p_batch[i]
                      for i in range(len(controller.old_probs))})
    feed_dict.update({controller.reward: nr_batch})
    feed_dict.update({controller.data_descriptive_feature: desc_batch})

    controller.session.run(controller.train_op, feed_dict=feed_dict)
    curr_loss, curr_kl, curr_ent = controller.session.run([controller.loss, controller.kl_div, controller.ent], feed_dict=feed_dict)


if __name__ == "__main__":
    if not run_from_ipython():
        main()