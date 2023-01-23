import torch
from ..base import BaseModelBuilder
from amber import backend as F
import numpy as np
import copy


class SequentialModelBuilder(BaseModelBuilder):
    def __init__(self, inputs_op, output_op, model_compile_dict, model_space, custom_objects=None, gpus=None, session=None, **kwargs):
        self.model_compile_dict = model_compile_dict
        if isinstance(inputs_op, (tuple, list)):
            assert len(inputs_op) == 1, "SequentialModelBuilder only support one inputs_op"
            self.input_node = inputs_op[0]
        elif isinstance(inputs_op, F.Operation):
            self.input_node = inputs_op
        else:
            raise TypeError("unknown type for inputs_op")
        if isinstance(output_op, (tuple, list)):
            assert len(output_op) == 1, "SequentialModelBuilder only support one inputs_op"
            self.output_node = output_op[0]
        elif isinstance(output_op, F.Operation):
            self.output_node = output_op
        else:
            raise TypeError("unknown type for output_node")
        self.model_space = model_space
        self.custom_objects = custom_objects or {}
        self.gpus = gpus
        self.session = session
        assert self.session is None or isinstance(self.session, F.SessionType)

    def build(self, model_states):
        layers = []
        curr_shape = self.input_node.Layer_attributes['shape']
        x = torch.empty(*curr_shape, device='cpu')
        # add a batch dim
        x = torch.unsqueeze(x, dim=0)
        # permute x if not a vector to match channel_last data format
        if len(curr_shape) > 1:
            dims = [0, len(curr_shape)] + np.arange(1, len(curr_shape)).tolist()
            layer = F.get_layer(op=F.Operation('permute', dims=dims))
            layers.append(layer)
            x = layer(x)
        for i, state in enumerate(model_states):
            if issubclass(type(state), int) or np.issubclass_(type(state), np.integer):
                op = self.model_space[i][state]
            elif isinstance(state, F.Operation) or callable(state):
                op = state
            else:
                raise Exception(
                    "cannot understand %s of type %s" % (state, type(state))
                )
            layer = F.get_layer(torch.squeeze(x, dim=0), op, custom_objects=self.custom_objects)
            x = layer(x)
            layers.append(layer)
        out = F.get_layer(torch.squeeze(x, dim=0), op=self.output_node, custom_objects=self.custom_objects)
        layers.append(out)
        model = F.Sequential(layers=layers)
        return model
    
    def __call__(self, model_states):
        if self.session is not None:
            F.set_session(self.session)
        if self.gpus is None or self.gpus == 1:
            model = self.build(model_states=model_states)
        else:
            raise ValueError(f"cannot parse gpus: {self.gpus}")
        model_compile_dict = copy.deepcopy(self.model_compile_dict)
        opt = model_compile_dict.pop("optimizer")
        if callable(opt): opt = opt()
        metrics = [
            x() if callable(x) else x for x in model_compile_dict.pop("metrics", [])
        ]
        model.compile(optimizer=opt, metrics=metrics, **model_compile_dict)
        return model
