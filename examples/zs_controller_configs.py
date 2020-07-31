"""
This script documents all viable configurations for Zero-Shot controller
ZZ
2020.7.30
"""

from collections import OrderedDict
import itertools

_holder = OrderedDict({
        'lstm_size': [32, 128],
        'temperature': [0.5, 2],
        #'descriptor_l1': [1e-1, 1e-8],
        'use_ppo_loss': [True, False],
        'kl_threshold': [0.05, 0.1],
        'max_episodes': [200, 400],
        'max_step_per_ep': [15, 30],
        'batch_size': [5, 15]
})


_rollout = [x for x in itertools.product(*_holder.values())]

_keys = [k for k in _holder]
configs_all = [
        { _keys[i]:x[i] for i in range(len(x))  } for x in _rollout
]
