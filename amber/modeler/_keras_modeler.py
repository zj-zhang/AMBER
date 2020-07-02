import tensorflow.keras as keras
from tensorflow.keras.models import Model
from ..architect import Operation
from .dag import get_layer
import numpy as np
from ._enas_modeler import ModelBuilder
import tensorflow as tf


class KerasModelBuilder(ModelBuilder):
    def __init__(self, inputs, outputs, model_compile_dict, model_space=None, gpus=None, **kwargs):
        self.model_compile_dict = model_compile_dict
        self.input_node = inputs
        self.output_node = outputs
        self.model_space = model_space
        self.gpus = gpus

    def __call__(self, model_states):
        if self.gpus is None or self.gpus == 1:
            model = build_sequential_model(
                        model_states=model_states,
                        input_state=self.input_node,
                        output_state=self.output_node,
                        model_compile_dict=self.model_compile_dict,
                        model_space=self.model_space
                        )
        else:
             model = build_multi_gpu_sequential_model(
                        model_states=model_states,
                        input_state=self.input_node,
                        output_state=self.output_node,
                        model_compile_dict=self.model_compile_dict,
                        model_space=self.model_space,
                        gpus=self.gpus
                        )
        return model


def build_sequential_model(model_states, input_state, output_state, model_compile_dict, **kwargs):
    """
    Args:
        model_states: a list of operators sampled from operator space
        input_state:
        output_state: specifies the output tensor, e.g. Dense(1, activation='sigmoid')
        model_compile_dict: a dict of `loss`, `optimizer` and `metrics`
    Returns:
        Keras.Model instance
    """
    inp = get_layer(None, input_state)
    x = inp
    model_space = kwargs.pop("model_space", None)
    for i, state in enumerate(model_states):
        if issubclass(type(state), Operation):
            x = get_layer(x, state)
        elif issubclass(type(state), int) or np.issubclass_(type(state), np.integer):
            assert model_space is not None, "if provided integer model_arc, must provide model_space in kwargs"
            x = get_layer(x, model_space[i][state])
        else:
            raise Exception("cannot understand %s of type %s" % (state, type(state)))
    out = get_layer(x, output_state)
    model = Model(inputs=inp, outputs=out)
    if not kwargs.pop('stop_compile', False):
        model.compile(**model_compile_dict)
    return model


def build_multi_gpu_sequential_model(model_states, input_state, output_state, model_compile_dict, gpus=4, **kwargs):
    try:
        from tensorflow.keras.utils import multi_gpu_model
    except Exception as e:
        raise Exception("multi gpu not supported in keras. check your version. Error: %s" % e)
    with tf.device('/cpu:0'):
        vanilla_model = build_sequential_model(model_states, input_state, output_state, model_compile_dict, stop_compile=True, **kwargs)
    model = multi_gpu_model(vanilla_model, gpus=gpus)
    model.compile(**model_compile_dict)
    return model


def build_sequential_model_from_string(model_states_str, input_state, output_state, state_space, model_compile_dict):
    """build a sequential model from a string of states
    """
    assert len(model_states_str) == len(state_space)
    str_to_state = [[str(state) for state in state_space[i]] for i in range(len(state_space))]
    try:
        model_states = [state_space[i][str_to_state[i].index(model_states_str[i])] for i in range(len(state_space))]
    except ValueError:
        raise Exception("model_states_str not found in state-space")
    return build_sequential_model(model_states, input_state, output_state, model_compile_dict)


def build_multi_gpu_sequential_model_from_string(model_states_str, input_state, output_state, state_space,
                                                 model_compile_dict):
    """build a sequential model from a string of states
    """
    assert len(model_states_str) == len(state_space)
    str_to_state = [[str(state) for state in state_space[i]] for i in range(len(state_space))]
    try:
        model_states = [state_space[i][str_to_state[i].index(model_states_str[i])] for i in range(len(state_space))]
    except ValueError:
        raise Exception("model_states_str not found in state-space")
    return build_multi_gpu_sequential_model(model_states, input_state, output_state, model_compile_dict)
