"""
This script documents all viable configurations for Zero-Shot controller
ZZ
2020.7.30
"""

import os
from collections import OrderedDict
import itertools
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from amber.architect import State, ModelSpace


def get_model_space_common():

    # Setup and params.
    state_space = ModelSpace()
    default_params = {"kernel_initializer": "glorot_uniform",
                      "activation": "relu"}
    param_list = [
            # Block 1:
            #[
            #    {"filters": 16, "kernel_size": 8},
            #    {"filters": 16, "kernel_size": 14},
            #    {"filters": 16, "kernel_size": 20}
            #],
            # Block 2:
            #[
            #    {"filters": 32, "kernel_size": 8},
            #    {"filters": 32, "kernel_size": 14},
            #    {"filters": 32, "kernel_size": 20}
            #],
            # Block 3:
            [
                {"filters": 256, "kernel_size": 8},
                {"filters": 256, "kernel_size": 14},
                {"filters": 256, "kernel_size": 20}
            ],
        ]

    # Build state space.
    layer_embedding_sharing = {}
    conv_seen = 0
    for i in range(len(param_list)):
        # Build conv states for this layer.
        conv_states = [State("Identity")]
        for j in range(len(param_list[i])):
            d = copy.deepcopy(default_params)
            for k, v in param_list[i][j].items():
                d[k] = v
            conv_states.append(State('conv1d', name="conv{}".format(conv_seen), **d))
        state_space.add_layer(conv_seen*3, conv_states)
        if i > 0:
            layer_embedding_sharing[conv_seen*3] = 0
        conv_seen += 1

        # Add pooling states.
        if i < len(param_list) - 1:
            pool_states = [State('Identity'),
                           State('maxpool1d', pool_size=4, strides=4),
                           State('avgpool1d', pool_size=4, strides=4)]
            if i > 0:
                layer_embedding_sharing[conv_seen*3-2] = 1
        else:
            pool_states = [
                    State('Flatten'),
                    State('GlobalMaxPool1D'),
                    State('GlobalAvgPool1D')
                ]
        state_space.add_layer(conv_seen*3-2, pool_states)


        # Add dropout
        state_space.add_layer(conv_seen*3-1, [
            State('Identity'),
            State('Dropout', rate=0.1),
            State('Dropout', rate=0.3),
            State('Dropout', rate=0.5)
            ])
        if i > 0:
            layer_embedding_sharing[conv_seen*3-1] = 2

    # Add final classifier layer.
    state_space.add_layer(conv_seen*3, [
            State('Dense', units=30, activation='relu'),
            State('Dense', units=100, activation='relu'),
            State('Identity')
        ])
    return state_space, layer_embedding_sharing



def get_zs_controller_configs():
    _holder = OrderedDict({
            'lstm_size': [32, 128],
            'temperature': [0.5, 1, 2],
            #'descriptor_l1': [1e-1, 1e-8],
            'use_ppo_loss': [True, False],
            #'kl_threshold': [0.05, 0.1],
            #'max_episodes': [200, 400],
            #'max_step_per_ep': [15, 30],
            #'batch_size': [5, 15]
    })


    _rollout = [x for x in itertools.product(*_holder.values())]

    _keys = [k for k in _holder]
    configs_all = [
            { _keys[i]:x[i] for i in range(len(x))  } for x in _rollout
    ]
    return configs_all


def analyze(wd):
    df = pd.read_table(os.path.join(wd, "sum_df.tsv"))
    d = json.loads(df.iloc[0]['config_str'].replace("'", '"').replace('True', 'true').replace('False', 'false'))
    config_keys = [k for k in d]
    configs = [[] for _ in range(len(config_keys))]
    # efficiency is the sum of target manager median AUC; i.e. how well the NAS works for both 
    efficiency= []
    # specificity is the difference of the manager median AUC under different descriptors; i.e. how well ZS-NAS can distinguish the managers
    specificity = []
    config_index = []
    for i in range(0, df.shape[0], 2):
        d = json.loads(df.iloc[i]['config_str'].replace("'", '"').replace('True', 'true').replace('False', 'false'))
        for k in range(len(config_keys)):
           configs[k].append(d[config_keys[k]])
        efficiency.append( df.iloc[[i,i+1]]['target_median'].sum() )
        m1_sp = df.iloc[i]['target_median'] - df.iloc[i+1]['other_median']
        m2_sp = df.iloc[i+1]['target_median'] - df.iloc[i]['other_median']
        specificity.append( m1_sp + m2_sp )
        config_index.append(df.iloc[i]['c'])


    data_dict = {
                "config_index": config_index,
                "efficiency": efficiency,
                "specificity": specificity
                }
    data_dict.update({config_keys[i]:configs[i] for i in range(len(config_keys))})
    eval_df = pd.DataFrame(data_dict, columns=['config_index'] + config_keys + ['efficiency', 'specificity'])
    eval_df.groupby("config_index").mean().sort_values(by="efficiency", ascending=False).to_csv(os.path.join(wd, "eval_df.tsv"), sep="\t", index=False, float_format="%.4f")
    eval_df.sort_values(by="efficiency", ascending=False).to_csv(os.path.join(wd, "eval_df.ungrouped.tsv"), sep="\t", index=False, float_format="%.4f")


