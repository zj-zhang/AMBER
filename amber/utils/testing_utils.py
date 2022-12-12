"""
Testing utilities for amber
"""

import unittest
from .. import backend as F
from .. import architect

class TestCase(F.TestCase):
    def tearDown(self):
        F.clear_session()
        super(TestCase, self).tearDown()


class PseudoModel:
    def __init__(self, pred_retr, eval_retr):
        self.pred_retr = pred_retr
        self.eval_retr = eval_retr

    def predict(self, *args, **kwargs):
        return self.pred_retr

    def evaluate(self, *args, **kwargs):
        return self.eval_retr

    def fit(self, *args, **kwargs):
        pass

    def load_weights(self, *args, **kwargs):
        pass


class PseudoKnowledge:
    def __init__(self, k_val):
        self.k_val = k_val

    def __call__(self, *args, **kwargs):
        return self.k_val


class PseudoConv1dModelBuilder:
    def __init__(self, input_shape, output_units, model_compile_dict=None):
        self.input_shape = input_shape
        self.output_units = output_units
        self.model_compile_dict = model_compile_dict or {'optimizer':'sgd', 'loss':'mse'}
        self.session = None

    def __call__(self, *args, **kwargs):
        model = F.Sequential()
        model.add(F.get_layer(op=F.Operation('Conv1D', filters=4, kernel_size=1, input_shape=self.input_shape)))
        model.add(F.get_layer(op=F.Operation('Flatten')))
        model.add(F.get_layer(op=F.Operation('Dense', units=self.output_units)))
        model.compile(**self.model_compile_dict)
        return model


class PseudoCaller:
    def __init__(self, retr_val=0):
        self.retr_val = retr_val

    def __call__(self, *args, **kwargs):
        return self.retr_val


class PseudoReward(PseudoCaller):
    def __init__(self, retr_val=0):
        self.retr_val = retr_val
        self.knowledge_function = None

    def __call__(self, *args, **kwargs):
        return self.retr_val, [self.retr_val], None


def get_example_conv1d_space(out_filters=8, num_layers=4, num_pool=2):
    """Model space for stacking the same conv-pool-id layers"""
    model_space = architect.ModelSpace()
    expand_layers = [num_layers//num_pool-1 + i*(num_layers//num_pool) for i in range(num_pool-1)]
    layer_sharing = {}
    for i in range(num_layers):
        model_space.add_layer(i, [
            architect.Operation('conv1d', filters=out_filters, kernel_size=7, activation='relu'),
            architect.Operation('maxpool1d', filters=out_filters, pool_size=5, strides=1),
            architect.Operation('avgpool1d', filters=out_filters, pool_size=5, strides=1),
            architect.Operation('identity', filters=out_filters)
      ])
        if i in expand_layers:
            out_filters *= 2
        if i > 0:
            layer_sharing[i] = 0
    return model_space, layer_sharing


def get_example_sparse_model_space(num_layers=4):
    """Model space for multi-input/output sparse feed-forward nets"""
    state_space = architect.ModelSpace()
    layer_sharing = {}
    for i in range(num_layers):
        state_space.add_layer(i, [
            architect.Operation('Dense', units=3, activation='relu'),
            architect.Operation('Dense', units=10, activation='relu'),
        ])
        if i > 0:
            layer_sharing[i] = 0
    return state_space, layer_sharing


def get_bionas_model_space():
    """architect.Operation_space is the place we define all possible operations (called `architect.Operations`) on each layer to stack a neural net.
    The state_space is defined in a layer-by-layer manner, i.e. first define the first layer (layer 0), then layer 1,
    so on and so forth. See below for how to define all possible choices for a given layer.

    Returns
    -------
    a pre-defined state_space object

    Notes
    ------
    Use key-word arguments to define layer-specific attributes.

    Adding `Identity` state to a layer is basically omitting a layer by performing no operations.
    """
    state_space = architect.ModelSpace()
    state_space.add_layer(0, [
        architect.Operation('conv1d', filters=3, kernel_size=8, kernel_initializer='glorot_uniform', activation='relu',
              name="conv1"),
        architect.Operation('conv1d', filters=3, kernel_size=14, kernel_initializer='glorot_uniform', activation='relu',
              name="conv1"),
        architect.Operation('conv1d', filters=3, kernel_size=20, kernel_initializer='glorot_uniform', activation='relu',
              name="conv1"),
        architect.Operation('denovo', filters=3, kernel_size=8, lambda_pos=1e-4,
              lambda_l1=1e-4, lambda_filter=1e-8, name='conv1'),
        architect.Operation('denovo', filters=3, kernel_size=14, lambda_pos=1e-4,
              lambda_l1=1e-4, lambda_filter=1e-8, name='conv1'),
        architect.Operation('denovo', filters=3, kernel_size=20, lambda_pos=1e-4,
              lambda_l1=1e-4, lambda_filter=1e-8, name='conv1'),
    ])
    state_space.add_layer(1, [
        architect.Operation('Identity'),
        architect.Operation('maxpool1d', pool_size=8, strides=8),
        architect.Operation('avgpool1d', pool_size=8, strides=8),

    ])
    state_space.add_layer(2, [
        architect.Operation('Flatten'),
        architect.Operation('GlobalMaxPool1D'),
        architect.Operation('GlobalAvgPool1D'),
        architect.Operation('SFC', output_dim=10, symmetric=True, smoothness_penalty=1., smoothness_l1=True,
              smoothness_second_diff=True, curvature_constraint=10., name='sfc'),
    ])
    state_space.add_layer(3, [
        architect.Operation('Dense', units=3, activation='relu'),
        architect.Operation('Dense', units=10, activation='relu'),
        architect.Operation('Identity')
    ])
    return state_space

