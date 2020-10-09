import tensorflow.keras as keras
from ..architect import Operation
from .dag import get_layer
import numpy as np
from ._enas_modeler import ModelBuilder
import tensorflow as tf
from .architecture_decoder import MultiIOArchitecture
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
import copy
import numpy as np


class KerasModelBuilder(ModelBuilder):
    def __init__(self, inputs_op, output_op, model_compile_dict, model_space=None, gpus=None, **kwargs):
        self.model_compile_dict = model_compile_dict
        self.input_node = inputs_op
        self.output_node = output_op
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
        elif type(self.gpus) is int:
            model = build_multi_gpu_sequential_model(
                        model_states=model_states,
                        input_state=self.input_node,
                        output_state=self.output_node,
                        model_compile_dict=self.model_compile_dict,
                        model_space=self.model_space,
                        gpus=self.gpus
                        )
        elif type(self.gpus) is list:
            mirrored_strategy = tf.distribute.MirroredStrategy(devices=self.gpus)
            with mirrored_strategy.scope():
                model = build_sequential_model(
                            model_states=model_states,
                            input_state=self.input_node,
                            output_state=self.output_node,
                            model_compile_dict=self.model_compile_dict,
                            model_space=self.model_space
                            )
        return model


class KerasMultiIOModelBuilder(ModelBuilder):
    """
    Note:
        Still not working if num_outputs=0
    """
    def __init__(self, inputs_op, output_op, model_compile_dict, model_space, with_input_blocks, with_output_blocks, dropout_rate=0.2, wsf=1, **kwargs):
        self.model_compile_dict = model_compile_dict
        self.inputs = inputs_op
        self.outputs = output_op
        self.model_space = model_space
        self.num_inputs = len(inputs_op) if type(inputs_op) in (list, tuple) else 0
        self.num_outputs = len(output_op) if type(output_op) in (list, tuple) else 0
        assert not (self.num_inputs==0 & self.num_outputs==0), "MultiIO cannot have single input and single output at the same time"
        self.with_input_blocks = with_input_blocks
        self.with_output_blocks = with_output_blocks
        if self.with_input_blocks: assert self.num_inputs > 0, "you specified with_input_blocks=True for KerasMultiIOModelBuilder, but only provided 1 num_inputs"
        self.decoder = MultiIOArchitecture(num_layers=len(self.model_space), num_inputs=self.num_inputs*self.with_input_blocks, num_outputs=self.num_outputs*self.with_output_blocks)

    def __call__(self, model_states):
        model = self._convert(model_states)
        if model is not None:
            model.compile(**self.model_compile_dict)
        return model

    def _convert(self, arc_seq, with_bn=True, wsf=1):
        inputs = [get_layer(x=None, state=x) for x in self.inputs] if self.num_inputs>0 else [get_layer(x=None, state=self.inputs)]
        op, inp, skp, out = self.decoder.decode(arc_seq)
        out_rowsum = np.apply_along_axis(np.sum, 1, out)
        out_colsum = np.apply_along_axis(np.sum, 0, out)
        skp_rowsum = np.array([1] + [sum(x) for x in skp])
        with_input_blocks = self.with_input_blocks
        # missing output connection
        if any(out_rowsum==0):
            print("invalid model: unconnected output")
            return None
        # missing output with skip connection
        if self.with_input_blocks is False and any( (skp_rowsum==0)&(out_colsum!=0) ):
            print("invalid model: output connected to layer with no input")
            return None

        # Build the model until outputs
        prev_layers = []
        for layer_id in range(len(self.model_space)):
            this_op = op[layer_id]
            # Prepare the inputs
            if with_input_blocks:
                this_inputs = [inputs[i] for i in np.where(inp[layer_id])[0]]
            else:
                this_inputs = inputs if layer_id == 0 else []
            if layer_id > 0:
                this_inputs += [ prev_layers[i] for i in np.where(skp[layer_id-1])[0] if prev_layers[i] is not None ]

            # Connect tensors
            model_op = copy.deepcopy(self.model_space[layer_id][this_op])
            if 'units' in model_op.Layer_attributes:
                model_op.Layer_attributes['units'] *= wsf
            elif 'filters' in model_op.Layer_attributes:
                model_op.Layer_attributes['filters'] *= wsf
            else:
                raise Exception("Cannot use wsf")
 
            if len(this_inputs) > 1:
                input_tensor = Concatenate()(this_inputs)
                layer = get_layer(x=input_tensor, state=model_op, with_bn=with_bn)
                prev_layers.append(layer)
            elif len(this_inputs) == 1:
                input_tensor = this_inputs[0]
                layer = get_layer(x=input_tensor, state=model_op, with_bn=with_bn)
                prev_layers.append(layer)
            else:
                prev_layers.append(None)  # skipped a layer

        # Build the outputs
        outputs_inputs = []
        for m, o in enumerate(out):
            idx = [i for i in np.where(o)[0] if prev_layers[i] is not None]
            if len(idx) > 1:
                outputs_inputs.append( Concatenate()([prev_layers[i] for i in idx])  )
            elif len(idx) == 1:
                outputs_inputs.append(prev_layers[idx[0]] )
            else:
                #raise Exception("Unconnected output %i"%m)
                print("Secondary unconnected output %i"%m)
                return None
        outputs = [get_layer(x=outputs_inputs[i], state=self.outputs[i]) for i in range(self.num_outputs)  ]
        model = Model(inputs=inputs, outputs=outputs)
        return model
     


def build_sequential_model(model_states, input_state, output_state, model_compile_dict, **kwargs):
    """
    Args:
        model_states: a list of _operators sampled from operator space
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
