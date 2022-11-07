"""represent neural network computation graph
as a directed-acyclic graph from a list of 
architecture selections

"""

import numpy as np
import warnings
try:
    import torch
    from torch.nn import Module
    has_torch = True
except ImportError:
    Module = object
    has_torch = False
from ..utils import corrected_tf as tf

# for general child
from .dag import EnasConv1dDAG
from .architectureDecoder import ResConvNetArchitecture


def get_torch_layer(fn_str):
    if fn_str == "relu":
        return torch.nn.ReLU()
    elif fn_str == "softmax":
        return torch.nn.Softmax()
    elif fn_str == "sigmoid":
        return torch.nn.Sigmoid()
    else:
        raise Exception("cannot get tensorflow layer for: %s" % fn_str)


class LambdaLayer(Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


mro = [Module, EnasConv1dDAG] if has_torch else [EnasConv1dDAG]
class EnasConv1dDAGpyTorch(*mro):
    def __init__(
        self,
        model_space,
        input_node,
        output_node,
        model_compile_dict,
        with_skip_connection=True,
        batch_size=128,
        keep_prob=0.9,
        l1_reg=0.0,
        l2_reg=0.0,
        reduction_factor=4,
        controller=None,
        child_train_op_kwargs=None,
        stem_config=None,
        data_format="NWC",
        train_fixed_arc=False,
        fixed_arc=None,
        name="EnasDAG",
        **kwargs,
    ):
        """EnasCnnDAG is a DAG model builder for using the weight sharing framework.

        This class deals with the Convolutional neural network.

        Parameters
        ----------
        model_space: amber.architect.ModelSpace
            model space to search network architectures from
        input_node: amber.architect.Operation, or list
            defines the input shapes and activations
        output_node: amber.architect.Operation, or list
            defines the output shapes and activations
        model_compile_dict: dict
            compile dict for child models
        train_fixed_arc: bool
            boolean indicator for whether is the final stage; if is True, must provide `fixed_arc` and not connect
            to a controller
        fixed_arc: list-like
            the architecture for final stage training
        name: str
            a name identifier for this instance

        Example
        --------
        .. code-block:: python

            %run amber/modeler/dag_pytorch.py
            from amber import architect
            from amber.utils import testing_utils
            input_op = [architect.Operation('input', shape=(100, 4), name="input")]
            output_op = architect.Operation('dense', units=1, activation='sigmoid', name="output")
            model_space, _ = testing_utils.get_example_conv1d_space(num_layers=12, num_pool=4)
            tf_sess = tf.Session()
            controller = architect.GeneralController(model_space=model_space, buffer_type='ordinal', with_skip_connection=True, session=tf_sess)
            arc, p = controller.get_action()
            x = torch.randn((10, 100, 4))
            dag = EnasConv1dDAGpyTorch(model_space=model_space, input_node=input_op,  output_node=output_op, model_compile_dict={}, reduction_factor=2)
            y_pred = dag(arc, x)


        """
        # init Modules, and manually call EnasConv1dDAG
        super().__init__()
        EnasConv1dDAG.__init__(
            self,
            model_space=model_space,
            input_node=input_node,
            output_node=output_node,
            session=tf.Session(),
            model_compile_dict={},
            reduction_factor=reduction_factor,
        )
        # ensure no tf variables are created
        assert (
            len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)) == 0
        )
        layers = self._build_dag()
        # register with ModuleDict in PyTorch
        self.layers = torch.nn.ModuleDict(layers)
        # helpers
        self.decoder = ResConvNetArchitecture(model_space=self.model_space)

    def forward(self, arc_seq, x):
        """use forward to compute predictions on the given arc_seq and
        datapoints of (input, label) pairs
        Forward pass should parse arc_seq
        """
        ops, skip_cons = self.decoder.decode(arc_seq)
        # pytorch conv1d expects data format of NCW
        if self.data_format == "NWC":
            x = torch.permute(x, (0, 2, 1))
        x = self.layers["stem"](x)
        hs = {}
        curr_pool_block = 0
        for layer_id in range(self.num_layers):
            # print(f'b{curr_pool_block},l{layer_id},o{ops[layer_id]}')
            x = self.layers[f"b{curr_pool_block},l{layer_id},o{ops[layer_id]}"](x)
            hs[f"b{curr_pool_block},l{layer_id}"] = x
            # connect residual layers
            if self.with_skip_connection is True and layer_id > 0:
                skip_bool = skip_cons[layer_id - 1]
                res_layers = [
                    hs[f"b{curr_pool_block},l{prev_layer}"]
                    for prev_layer in range(layer_id)
                    if skip_bool[prev_layer] == 1
                ]
                if len(res_layers):
                    res_layers = torch.stack(res_layers, dim=0)
                    x = x + res_layers.sum(dim=0)
                    x = self.layers[f"b{curr_pool_block},l{layer_id},bn"](x)
            # perform pooling & update block
            if layer_id in self.pool_layers:
                next_pool_block = curr_pool_block + 1
                # update each prev_layer from curr_pool_block to next_pool_block
                if self.with_skip_connection:
                    for prev_layer in range(layer_id):
                        hs[f"b{next_pool_block},l{prev_layer}"] = self.layers[
                            f"b{next_pool_block},l{prev_layer}"
                        ](hs[f"b{curr_pool_block},l{prev_layer}"])
                # update curr layer from curr_pool_block to next_pool_block
                curr_pool_block = next_pool_block
                x = self.layers[f"b{curr_pool_block},l{layer_id}"](x)
                hs[f"b{curr_pool_block},l{layer_id}"] = x
        x = self.layers["flatten_op"](x)
        x = self.layers["fc"](x)
        out = self.layers["out"](x)
        return out

    def _build_dag(self, *args, **kwargs):
        """use this to init all torch.nn.Layers based on model_space"""
        # for making connections with name IDs
        # key = (b'pool_block', l'layer_id', o'op_id')
        layers = {}
        pool_block = 0  # skip connections should always connect within a pool block
        # to ensure the number of channels are matched
        # build stems
        has_stem_conv = self.stem_config.get("has_stem_conv", True)
        if has_stem_conv:
            stem_kernel_size = self.stem_config.get("stem_kernel_size", 8)
            stem_filters = self.out_filters[0]
            layers["stem"] = torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=self.input_node.Layer_attributes["shape"][1],
                    out_channels=stem_filters,
                    kernel_size=stem_kernel_size,
                    padding="same",
                ),
                torch.nn.BatchNorm1d(num_features=stem_filters),
            )
        else:
            layers["stem"] = torch.nn.Sequential()

        refactorized_layers = {}
        for layer_id in range(self.num_layers):
            x = self._layer(layer_id)
            # x is a list of length = len(self.model_space[layer_id])
            for i, xx in enumerate(x):
                layers[f"b{pool_block},l{layer_id},o{i}"] = xx
            # add bn layer after residual sum
            if self.with_skip_connection is True:
                layers[f"b{pool_block},l{layer_id},bn"] = torch.nn.BatchNorm1d(
                    self.out_filters[layer_id]
                )
            # refactor channels at pool layers
            if (self.with_skip_connection is True) and (layer_id in self.pool_layers):
                pool_block += 1
                tmp = {}
                for name, layer in layers.items():
                    if name == "stem":
                        continue
                    this_pb, this_layer, this_op = name.split(",")
                    # each pool should only have one
                    if f"b{pool_block},{this_layer}" in refactorized_layers:
                        continue
                    x = self._refactorized_channels_for_skipcon(
                        self.out_filters[layer_id], self.out_filters[layer_id + 1]
                    )
                    refactorized_layers[f"b{pool_block},{this_layer}"] = x
        layers.update(refactorized_layers)
        flatten_op = (
            self.stem_config["flatten_op"]
            if "flatten_op" in self.stem_config
            else "flatten"
        )
        if flatten_op == "global_avg_pool" or flatten_op == "gap":
            inp_c = self.out_filters[-1]
            layers["flatten_op"] = LambdaLayer(lambda x: torch.mean(x, dim=-1))
        elif flatten_op == "flatten":
            inp_c = (
                self.input_node.Layer_attributes["shape"][0]
                // (self.reduction_factor ** len(self.pool_layers))
                * self.out_filters[-1]
            )
            layers["flatten_op"] = torch.nn.Flatten()
        else:
            raise Exception("cannot understand flatten_op: %s" % flatten_op)

        fc_units = (
            self.stem_config["fc_units"] if "fc_units" in self.stem_config else 1000
        )
        layers["fc"] = torch.nn.Sequential(
            torch.nn.Dropout(1 - self.keep_prob),
            torch.nn.Linear(inp_c, fc_units),
            torch.nn.BatchNorm1d(fc_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(1 - self.keep_prob),
        )
        layers["out"] = torch.nn.Sequential(
            torch.nn.Linear(fc_units, self.output_node.Layer_attributes["units"]),
            get_torch_layer(self.output_node.Layer_attributes["activation"]),
        )
        return layers

    def _refactorized_channels_for_skipcon(self, in_channels, out_channels):
        """for dealing with mismatch-dimensions in skip connections: use a linear transformation"""
        x = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            ),
            torch.nn.MaxPool1d(
                kernel_size=self.reduction_factor, stride=self.reduction_factor
            ),
        )
        return x

    def _layer(self, layer_id):
        # If prev_layer is pool layer, in_channels will already be updated by `_refactorized_channels_for_skipcon`
        # in_channels = self.out_filters[layer_id] if (layer_id-1) in self.pool_layers else self.out_filters[max(0, layer_id-1)]
        # Therefore, `_layer` will always deal with the same in_channels and out_channels operations
        in_channels = self.out_filters[layer_id]
        out_channels = self.out_filters[layer_id]
        branches = []
        strides = []
        for i in range(len(self.model_space[layer_id])):
            if self.model_space[layer_id][i].Layer_type == "conv1d":
                # print('%i, conv1d' % layer_id)
                y = self._conv_branch(
                    in_channels=in_channels,
                    layer_attr=self.model_space[layer_id][i].Layer_attributes,
                )
                branches.append(y)
            elif self.model_space[layer_id][i].Layer_type == "maxpool1d":
                # print('%i, maxpool1d' % layer_id)
                y = self._pool_branch(
                    in_channels=in_channels,
                    avg_or_max="max",
                    layer_attr=self.model_space[layer_id][i].Layer_attributes,
                )
                branches.append(y)
                strides.append(
                    self.model_space[layer_id][i].Layer_attributes["strides"]
                )
            elif self.model_space[layer_id][i].Layer_type == "avgpool1d":
                # print('%i, avgpool1d' % layer_id)
                y = self._pool_branch(
                    in_channels=in_channels,
                    avg_or_max="avg",
                    layer_attr=self.model_space[layer_id][i].Layer_attributes,
                )
                branches.append(y)
                strides.append(
                    self.model_space[layer_id][i].Layer_attributes["strides"]
                )
            elif self.model_space[layer_id][i].Layer_type == "identity":
                y = self._identity_branch()
                branches.append(y)
            else:
                raise Exception("Unknown layer: %s" % self.model_space[layer_id][i])
        if len(strides) > 0:
            assert len(set(strides)) == 1, (
                "If you set strides!=1 (i.e. a reduction layer), then all candidate operations must have the same strides to keep the shape identical; got %s"
                % strides
            )
        return branches

    def _conv_branch(self, in_channels, layer_attr):
        kernel_size = layer_attr["kernel_size"]
        activation_fn = layer_attr["activation"]
        dilation = layer_attr.get("dilation", 1)
        filters = layer_attr["filters"]
        x = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=kernel_size,
                dilation=dilation,
                padding="same",
            ),
            torch.nn.BatchNorm1d(filters),
            get_torch_layer(activation_fn),
        )
        return x

    def _pool_branch(self, in_channels, avg_or_max, layer_attr):
        pool_size = layer_attr["pool_size"]
        strides = layer_attr["strides"]
        filters = layer_attr["filters"]
        assert pool_size % 2, ValueError("PyTorch MaxPool1d must have an uneven pool_size to keep constant shapes in supernet; got %i" % pool_size)
        x = []
        if self.add_conv1_under_pool:
            x.extend(
                [
                    torch.nn.Conv1d(
                        in_channels, filters, kernel_size=1, padding="same"
                    ),
                    torch.nn.BatchNorm1d(filters),
                    torch.nn.ReLU(),
                ]
            )

        if avg_or_max == "avg":
            x.append(
                torch.nn.AvgPool1d(
                    kernel_size=pool_size,
                    stride=strides,
                    padding=int((pool_size - 1) / 2),
                )
            )
        elif avg_or_max == "max":
            x.append(
                torch.nn.MaxPool1d(
                    kernel_size=pool_size,
                    stride=strides,
                    padding=int((pool_size - 1) / 2),
                )
            )
        else:
            raise ValueError("Unknown pool {}".format(avg_or_max))
        x = torch.nn.Sequential(*x)
        return x
    
    def _identity_branch(self):
        return torch.nn.Sequential()
    
    def _verify_args(self):
        super()._verify_args()
        self.input_ph = None
        self.label_ph = None

    def _build_sample_arc(self, input_tensor=None, label_tensor=None, **kwargs):
        pass

    def _build_fixed_arc(self, input_tensor=None, label_tensor=None, **kwargs):
        pass

    def _compile(self, model_output, labels=None, is_training=True, var_scope=None):
        pass
