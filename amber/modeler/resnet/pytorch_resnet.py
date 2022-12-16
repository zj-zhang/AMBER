import pytorch_lightning as pl
import torch
from typing import Tuple, List, Union
from collections import OrderedDict
import copy
from argparse import Namespace
from typing import Optional, Union, List, Dict, Any
from ... import backend as F
from ..base import BaseModelBuilder
from ..architectureDecoder import ResConvNetArchitecture
from ..supernet.pytorch_supernet import ConcatLayer, GlobalAveragePooling1DLayer, get_torch_layer


class ResidualCnnBuilder(BaseModelBuilder):
    """Class for converting an architecture sequence tokens to a PyTorch-lightning model

    Parameters
    ----------
    inputs_op : amber.architect.modelSpace.Operation
    output_op : amber.architect.modelSpace.Operation
    fc_units : int
        number of units in the fully-connected layer
    flatten_mode : {'GAP', 'Flatten'}
        the flatten mode to convert conv layers to fully-connected layers.
    model_compile_dict : dict
    model_space : amber.architect.modelSpace.ModelSpace
    dropout_rate : float
        dropout rate, must be 0<dropout_rate<1
    wsf : int
        width scale factor
    reduction_factor : int
        reduce the feature dimension by this factor at each pool layer
        
    Returns
    --------
    amber.modeler.base.LightningResNet : 
        a pytorch_lightning model
    """

    def __init__(self, inputs_op, output_op, fc_units, flatten_mode, model_compile_dict, model_space,
                 dropout_rate=0.2, reduction_factor=4, wsf=1, add_conv1_under_pool=True, verbose=1, **kwargs):
        self.model_compile_dict = model_compile_dict
        self.inputs = inputs_op
        self.outputs = output_op
        self.fc_units = fc_units
        self.verbose = verbose
        assert flatten_mode.lower() in {
            'gap', 'flatten'}, "Unknown flatten mode: %s" % flatten_mode
        self.flatten_mode = flatten_mode.lower()
        self.model_space = model_space
        self.dropout_rate = dropout_rate
        self.wsf = wsf
        self.reduction_factor = reduction_factor
        self.add_conv1_under_pool = add_conv1_under_pool
        self.decoder = ResConvNetArchitecture(model_space=model_space)

    @staticmethod
    def get_out_filters(model_space):
        out_filters = []
        pool_layers = []
        layerid_to_block = {}
        block_to_filters = {}
        pool_block = 0
        for layer_id in range(len(model_space)):
            layer = model_space[layer_id]
            this_out_filters = [l.Layer_attributes['filters'] for l in layer]
            assert len(
                set(this_out_filters)) == 1, "EnasConv1dDAG only supports one identical number of filters per layer," \
                                             "but found %i in layer %s" % (
                                                 len(set(this_out_filters)), layer)
            if len(out_filters) and this_out_filters[0] != out_filters[-1]:
                pool_layers.append(layer_id - 1)
                block_to_filters[pool_block] = out_filters[-1]
                pool_block += 1
            layerid_to_block[layer_id] = pool_block
            out_filters.append(this_out_filters[0])
        block_to_filters[pool_block] = out_filters[-1]
        # print(out_filters)
        # print(pool_layers)
        return out_filters, pool_layers, layerid_to_block, block_to_filters

    def __call__(self, model_states):
        model = self._convert(model_states, verbose=self.verbose)
        if model is not None:
            model.compile(**self.model_compile_dict)
        return model
    
    def _convert(self, arc_seq, verbose=False):
        # init a new Model instance
        out_filters, pool_layers, layerid_to_block, block_to_filters = self.get_out_filters(self.model_space)
        layers = []
        ops, skip_cons = self.decoder.decode(arc_seq)
        # for making connections with name IDs
        # key = f"layer_{layer_id}-b{pool_block}"
        # build conv stem
        stem = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=self.inputs.Layer_attributes["shape"][1],
                out_channels=out_filters[0] * self.wsf,
                kernel_size=7,
                padding="same",
            ),
            torch.nn.BatchNorm1d(num_features=out_filters[0] * self.wsf),
        )
        #model.add
        layers.append(('stem', stem))
        # build conv trunk
        # skip connections should always connect within a pool block
        # to ensure the number of channels are matched
        for layer_id in range(len(self.model_space)):
            pool_block = layerid_to_block[layer_id]
            if verbose:
                print("layer id=%i, out_filters=%i x %i" % (
                    layer_id, out_filters[layer_id], self.wsf))
            this_layer = self.model_space[layer_id][ops[layer_id]]
            if verbose:
                print(this_layer)
            # build conv
            conv_op = self.res_layer(
                layer=this_layer, 
                width_scale_factor=self.wsf, 
                in_channels=out_filters[layer_id] * self.wsf)
            #model.add
            layers.append((
                f"layer_{layer_id}-b{pool_block}", 
                conv_op,
            ))
            # build skip connections
            if layer_id > 0:
                skip = skip_cons[layer_id - 1]
                skip_layers = [i for i in range(len(skip)) if skip[i] == 1]
                # examine if skip layers in the same block has been built
                # and trace back to all prev blocks
                for skip_layer in skip_layers:
                    start_block = layerid_to_block[skip_layer]
                    for prev_block in range(start_block+1, pool_block+1):
                        skip_id = f"layer_{skip_layer}-b{prev_block}"
                        #if skip_id not in model.layers:
                        model_layers = set([layer[0] for layer in layers])
                        if skip_id not in model_layers:
                            skip_op = self.factorized_reduction_layer(
                                in_channels=block_to_filters[prev_block-1] * self.wsf,
                                out_channels=block_to_filters[prev_block] * self.wsf,
                                reduction_factor = self.reduction_factor
                                )
                            if verbose:
                                print(skip_id, f"layer_{skip_layer}-b{prev_block-1}")
                            #model.add
                            layers.append((
                                skip_id, 
                                skip_op, 
                                f"layer_{skip_layer}-b{prev_block-1}"
                            ))
                # add residual sum to model, if any
                if len(skip_layers):
                    res_sum = torch.nn.Sequential(
                        ConcatLayer(take_sum=True),
                        torch.nn.BatchNorm1d(num_features=out_filters[layer_id]*self.wsf)
                    )
                    #model.add
                    layers.append((
                        f"resid_{layer_id}", 
                        res_sum, 
                        [f"layer_{layer_id}-b{pool_block}"] + \
                            [f"layer_{skip_layer}-b{pool_block}" for skip_layer in skip_layers]
                    ))
            # add a fatorize layer if in pool_layers
            if layer_id in pool_layers:
                factor_op = self.factorized_reduction_layer(
                        in_channels=out_filters[layer_id] * self.wsf,
                        out_channels=out_filters[layer_id+1] * self.wsf,
                        reduction_factor = self.reduction_factor
                        )
                #model.add
                layers.append((
                    f"pool_from_{layer_id}_to_{layer_id+1}",
                    factor_op
                ))
        
        if self.flatten_mode == "global_avg_pool" or self.flatten_mode == "gap":
            inp_c = out_filters[-1] * self.wsf
            flatten_op = GlobalAveragePooling1DLayer()
        elif self.flatten_mode == "flatten":
            inp_c = (
                self.inputs.Layer_attributes["shape"][0]
                // (self.reduction_factor ** len(pool_layers))
                * out_filters[-1] * self.wsf
            )
            flatten_op = torch.nn.Flatten()
        else:
            raise Exception("cannot understand flatten_op: %s" % self.flatten_mode)
        #model.add
        layers.append(("flatten", flatten_op))
        
        #model.add
        layers.append(("fc", torch.nn.Sequential(
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(inp_c, self.fc_units),
            torch.nn.BatchNorm1d(self.fc_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_rate),
        )))
        #model.add
        layers.append(("out", torch.nn.Sequential(
            torch.nn.Linear(self.fc_units, self.outputs.Layer_attributes["units"]),
            get_torch_layer(self.outputs.Layer_attributes["activation"]),
        )))
        model = LightningResNet(layers=layers, model_compile_dict=self.model_compile_dict)
        return model


    @staticmethod
    def res_layer(layer, width_scale_factor, in_channels, add_conv1_under_pool=True):
        if layer.Layer_type == 'conv1d':
            activation = layer.Layer_attributes['activation']
            filters = width_scale_factor * layer.Layer_attributes['filters']
            kernel_size = layer.Layer_attributes['kernel_size']
            dilation = layer.Layer_attributes.get('dilation', 1)
            conv_op = torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding="same",
                ),
                torch.nn.BatchNorm1d(filters),
                get_torch_layer(activation),
            )
        elif layer.Layer_type in ('maxpool1d', 'avgpool1d'):
            pool_size = layer.Layer_attributes["pool_size"]
            strides = layer.Layer_attributes["strides"]
            filters = layer.Layer_attributes["filters"]
            assert pool_size % 2, ValueError("PyTorch MaxPool1d must have an uneven pool_size to keep constant shapes in supernet; got %i" % pool_size)
            x = []
            if add_conv1_under_pool:
                x.extend(
                    [
                        torch.nn.Conv1d(
                            in_channels, filters, kernel_size=1, padding="same"
                        ),
                        torch.nn.BatchNorm1d(filters),
                        torch.nn.ReLU(),
                    ]
                )

            if layer.Layer_type == "avgpool1d":
                x.append(
                    torch.nn.AvgPool1d(
                        kernel_size=pool_size,
                        stride=strides,
                        padding=int((pool_size - 1) / 2),
                    )
                )
            elif layer.Layer_type == "maxpool1d":
                x.append(
                    torch.nn.MaxPool1d(
                        kernel_size=pool_size,
                        stride=strides,
                        padding=int((pool_size - 1) / 2),
                    )
                )
            else:
                raise ValueError("Unknown pool {}".format(layer.Layer_type))
            conv_op = torch.nn.Sequential(*x)
        elif layer.Layer_type == 'identity':
            conv_op = torch.nn.Sequential()
        return conv_op

    @staticmethod
    def factorized_reduction_layer(in_channels, out_channels, reduction_factor=4):
        x = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same",
            ),
            torch.nn.MaxPool1d(
                kernel_size=reduction_factor, stride=reduction_factor
            ),
        )
        return x


class LightningResNet(F.Model):
    """LightningResNet is a subclass of pytorch_lightning.LightningModule

    It implements a basic functions of `step`, `configure_optimizers` but provides a similar
    user i/o arguments as tensorflow.keras.Model 

    A module builder will add `torch.nn.Module`s to this instance, and define its forward 
    pass function. Then this instance is responsible for training and evaluations.
    add module: use torch.nn.Module.add_module
    define forward pass: private __forward_tracker list
    """
    def __init__(self, layers=None, data_format='NWC', *args, **kwargs):
        super().__init__()
        self.__forward_pass_tracker = []
        self.layers = torch.nn.ModuleDict()
        self.hs = {}
        self.is_compiled = False
        self.criterion = None
        self.optimizer = None
        self.metrics = {}
        self.trainer = None
        self.data_format = data_format
        layers = layers or []
        for layer in layers:
            layer_id, operation, input_ids = layer[0], layer[1], layer[2] if len(layer)>2 else None
            self.add(layer_id=layer_id, operation=operation, input_ids=input_ids)
        self.save_hyperparameters()
    
    @property
    def forward_tracker(self):
        # return a read-only view
        return copy.copy(self.__forward_pass_tracker)
    
    def add(self, layer_id: str, operation, input_ids: Union[str, List, Tuple] = None):
        self.layers[layer_id] = operation
        self.__forward_pass_tracker.append((layer_id, input_ids))
    
    def forward(self, x, verbose=False):
        """Scaffold forward-pass function that follows the operations in 
        the pre-set in self.__forward_pass_tracker
        """
        # permute input, if data_format has channel last
        if self.data_format == 'NWC':
            x = torch.permute(x, (0,2,1))
        # intermediate outputs, for branching models
        self.hs = {}
        # layer_id : current layer name
        # input_ids : if None,       take the output from prev layer as input
        #             if tuple/list, expect a list of layer_ids (str)
        for layer_id, input_ids in self.__forward_pass_tracker:
            assert layer_id in self.layers
            if verbose:
                print(layer_id)
                print([self.hs[layer_id].shape for layer_id in self.hs])
                print(input_ids)
            this_inputs = x if input_ids is None else self.hs[input_ids] if type(input_ids) is str else [self.hs[i] for i in input_ids]
            out = self.layers[layer_id](this_inputs)
            self.hs[layer_id] = out
            x = out
        return out


