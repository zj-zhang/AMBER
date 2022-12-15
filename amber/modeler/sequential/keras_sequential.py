from ... import backend as F
from ...backend import get_layer, Model  # type: ignore
import numpy as np
from ..base import BaseModelBuilder
from ..architectureDecoder import MultiIOArchitecture
import copy
from ...architect.modelSpace import BranchedModelSpace
import tensorflow as tf
from tensorflow.keras.utils import multi_gpu_model


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
        inp = get_layer(None, self.input_node, custom_objects=self.custom_objects)
        x = inp
        for i, state in enumerate(model_states):
            if issubclass(type(state), int) or np.issubclass_(type(state), np.integer):
                op = self.model_space[i][state]
            elif isinstance(state, F.Operation) or callable(state):
                op = state
            else:
                raise Exception(
                    "cannot understand %s of type %s" % (state, type(state))
                )
            x = get_layer(x, op, custom_objects=self.custom_objects)

        out = get_layer(x, self.output_node, custom_objects=self.custom_objects)
        model = F.Model(inputs=inp, outputs=out)
        return model
    
    def __call__(self, model_states):
        if self.session is not None:
            F.set_session(self.session)
        if self.gpus is None or self.gpus == 1:
            model = self.build(model_states=model_states)
        elif isinstance(self.gpus, int):
            with F.device_scope('/cpu:0'):
                vanilla_model = self.build(model_states)
            model = multi_gpu_model(vanilla_model, gpus=self.gpus)
        elif isinstance(self.gpus, list):
            mirrored_strategy = tf.distribute.MirroredStrategy(
                devices=self.gpus)
            with mirrored_strategy.scope():
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


class SequentialBranchModelBuilder(BaseModelBuilder):
    def __init__(self, inputs_op, output_op, model_compile_dict,
                 model_space=None, with_bn=False, **kwargs):
        assert isinstance(model_space, BranchedModelSpace)
        assert len(inputs_op) == len(model_space.subspaces[0])
        self.inputs_op = inputs_op
        if type(output_op) in (tuple, list):
            assert len(output_op) == 1
            self.output_op = output_op[0]
        else:
            self.output_op = output_op
        self.model_space = model_space
        self.model_compile_dict = model_compile_dict
        self.with_bn = with_bn
        self._branch_to_layer = self.model_space.branch_to_layer

    def _build_branch(self, input_op, model_states, model_space):
        if issubclass(type(input_op), F.Operation):
            inp = get_layer(None, input_op)
        else:
            inp = input_op
        x = inp
        assert len(model_states) > 0
        for i, state in enumerate(model_states):
            if issubclass(type(state), F.Operation):
                x = get_layer(x, state)
            elif issubclass(type(state), int) or np.issubclass_(type(state), np.integer):
                assert model_space is not None, "if provided integer model_arc, must provide model_space in kwargs"
                x = get_layer(x, model_space[i][state], with_bn=self.with_bn)
            else:
                raise Exception(
                    "cannot understand %s of type %s" %
                    (state, type(state)))
        return inp, x

    def __call__(self, model_states, **kwargs):
        inps = []
        branches = []
        # build branch sequentially
        for i in range(len(self.inputs_op)):
            inp, out = self._build_branch(
                input_op=self.inputs_op[i],
                model_states=[model_states[j]
                              for j in self._branch_to_layer[(0, i)]],
                model_space=self.model_space.subspaces[0][i]
            )
            inps.append(inp)
            branches.append(out)
        # merge branches
        if self.model_space.concat_op == 'concatenate':
            branch_merge = get_layer(
                x=branches, op=F.Operation('concatenate'))
        else:
            raise ValueError(
                'Model builder cannot understand model space concat op: %s' %
                self.model_space.conat_op)
        # build stem
        _, h = self._build_branch(
            input_op=branch_merge,
            model_states=[model_states[j]
                          for j in self._branch_to_layer[(1, None)]],
            model_space=self.model_space.subspaces[1]
        )
        out = get_layer(x=h, op=self.output_op)
        model = Model(inputs=inps, outputs=out)
        model.compile(**self.model_compile_dict)
        return model


class SequentialMultiIOModelBuilder(BaseModelBuilder):
    def __init__(self, inputs_op, output_op, model_compile_dict, model_space,
                 with_input_blocks=False, with_output_blocks=False, dropout_rate=0.2, wsf=1, **kwargs):
        self.model_compile_dict = model_compile_dict
        self.inputs = inputs_op
        self.outputs = output_op
        self.model_space = model_space
        self.num_inputs = len(inputs_op) if type(
            inputs_op) in (list, tuple) else 0
        self.num_outputs = len(output_op) if type(
            output_op) in (list, tuple) else 0
        assert not (self.num_inputs == 0 & self.num_outputs == 0), "MultiIO cannot have single input and single output at " \
            "the same time "
        self.with_input_blocks = with_input_blocks
        self.with_output_blocks = with_output_blocks
        if self.with_input_blocks:
            assert self.num_inputs > 0, "you specified with_input_blocks=True for " \
                "KerasMultiIOModelBuilder, but only provided 1 " \
                "num_inputs "
        self.decoder = MultiIOArchitecture(
            model_space=self.model_space,
            num_inputs=self.num_inputs * self.with_input_blocks,
            num_outputs=self.num_outputs * self.with_output_blocks)

    def __call__(self, model_states):
        model = self._convert(model_states)
        if model is not None:
            model.compile(**self.model_compile_dict)
        return model

    def _convert(self, arc_seq, with_bn=True, wsf=1):
        inputs = [
            get_layer(
                x=None,
                op=x) for x in self.inputs] if self.num_inputs > 0 else [
            get_layer(
                x=None,
                op=self.inputs)]
        op, inp, skp, out = self.decoder.decode(arc_seq)
        out_rowsum = np.apply_along_axis(np.sum, 1, out)
        out_colsum = np.apply_along_axis(np.sum, 0, out)
        skp_rowsum = np.array([1] + [sum(x) for x in skp])
        with_input_blocks = self.with_input_blocks
        # missing output connection
        if any(out_rowsum == 0):
            print("invalid model: unconnected output")
            return None
        # missing output with skip connection
        if self.with_input_blocks is False and any(
                (skp_rowsum == 0) & (out_colsum != 0)):
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
                this_inputs += [prev_layers[i]
                                for i in np.where(skp[layer_id - 1])[0] if prev_layers[i] is not None]

            # Connect tensors
            model_op = copy.deepcopy(self.model_space[layer_id][this_op])
            if 'units' in model_op.Layer_attributes:
                model_op.Layer_attributes['units'] *= wsf
            elif 'filters' in model_op.Layer_attributes:
                model_op.Layer_attributes['filters'] *= wsf
            else:
                raise Exception("Cannot use wsf")

            if len(this_inputs) > 1:
                input_tensor = F.concat(this_inputs)
                layer = get_layer(
                    x=input_tensor,
                    op=model_op,
                    with_bn=with_bn)
                prev_layers.append(layer)
            elif len(this_inputs) == 1:
                input_tensor = this_inputs[0]
                layer = get_layer(
                    x=input_tensor,
                    op=model_op,
                    with_bn=with_bn)
                prev_layers.append(layer)
            else:
                prev_layers.append(None)  # skipped a layer

        # Build the outputs
        outputs_inputs = []
        for m, o in enumerate(out):
            idx = [i for i in np.where(o)[0] if prev_layers[i] is not None]
            if len(idx) > 1:
                outputs_inputs.append(F.concat(
                    [prev_layers[i] for i in idx]))
            elif len(idx) == 1:
                outputs_inputs.append(prev_layers[idx[0]])
            else:
                #raise Exception("Unconnected output %i"%m)
                print("Secondary unconnected output %i" % m)
                return None
        outputs = [
            get_layer(
                x=outputs_inputs[i],
                op=self.outputs[i]) for i in range(
                self.num_outputs)]
        model = Model(inputs=inputs, outputs=outputs)
        return model
