#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example wrapper `Amber` use for searching DeepSEA 2015
ZZ, May 5, 2020
"""

from amber import Amber
from amber.architect import ModelSpace, Operation

def get_model_space(out_filters=64, num_layers=9):
    model_space = ModelSpace()
    num_pool = 4
    for i in range(num_layers):
        model_space.add_layer(i, [
            Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu'),
            Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu'),
            Operation('conv1d', filters=out_filters, kernel_size=8, activation='relu', dilation=10),
            Operation('conv1d', filters=out_filters, kernel_size=4, activation='relu', dilation=10),
            # max/avg pool has underlying 1x1 conv
            Operation('maxpool1d', filters=out_filters, pool_size=4, strides=1),
            Operation('avgpool1d', filters=out_filters, pool_size=4, strides=1),
            Operation('identity', filters=out_filters),
      ])
        if i in expand_layers:
            out_filters *= 2
    return model_space

# First, define the components we need to use
type_dict = {
    'architect_type': 'GeneralController',
    'modeler_type': 'EnasCnnModelBuilder',
    'knowledge_fn_type': 'zero',
    'reward_fn_type': 'LossAucReward',
    'manager_type': 'EnasManager',
    'env_type': 'EnasTrainEnv'
}


# Next, define the specifics
input_node = Operation('input', shape=(1000, 4), name="input", dtype=tf.float32)
output_node = Operation('dense', units=919, activation='sigmoid')
model_compile_dict = {
    'loss': 'binary_crossentropy',
    'optimizer': 'adam',
}
specs = {
    'model_space': get_model_space(out_filters=32, num_layers=12, num_pool=4),

    'model_builder': {
        'dag_func': 'EnasConv1dDAG',
        'batch_size': 1000,
        'inputs_op': [input_node],
        'output_op': [output_node],
        'model_compile_dict': model_compile_dict,
         'dag_kwargs': {
            'stem_config': {
                'flatten_op': 'flatten',
                'fc_units': 925
            }
        }
    },

    'knowedge_fn': {'data': None, 'params': None},

    'reward_fn': {'method': 'auc'},

    'manager': {
        'data': {
            'train_data': 'data/train_mat.h5',
            'val_data': 'data/val_mat.h5'
        },
        'params': {
            'epochs': 1,
            'child_batchsize': 1000,
            'store_fn': 'minimal',
            'working_dir': wd,
            'verbose': 2
        }
    },

    'train_env': {
        'max_episode': 300,
        'max_step_per_ep': 100,
        'working_dir': wd,
        'time_budget': "24:00:00",
        'with_input_blocks': False,
        'with_skip_connection': True,
        'child_train_steps': 500,
        'child_warm_up_epochs': 1
    }
}