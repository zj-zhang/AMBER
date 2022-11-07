from .base import ModelBuilder, BaseTorchModel
from .architectureDecoder import ResConvNetArchitecture
from .dag_pytorch import LambdaLayer, get_torch_layer
try:
    import torch
    has_torch = True
except ImportError:
    has_torch = False


class PytorchResidualCnnBuilder(ModelBuilder):
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
    amber.modeler.base.BaseTorchModel : 
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
        model = BaseTorchModel(model_compile_dict=self.model_compile_dict)
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
        model.add('stem', stem)
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
            model.add(
                layer_id=f"layer_{layer_id}-b{pool_block}", 
                operation=conv_op,
                )
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
                        if skip_id not in model.layers:
                            skip_op = self.factorized_reduction_layer(
                                in_channels=block_to_filters[prev_block-1] * self.wsf,
                                out_channels=block_to_filters[prev_block] * self.wsf,
                                reduction_factor = self.reduction_factor
                                )
                            if verbose:
                                print(skip_id, f"layer_{skip_layer}-b{prev_block-1}")
                            model.add(
                                layer_id=skip_id, 
                                operation=skip_op, 
                                input_ids=f"layer_{skip_layer}-b{prev_block-1}"
                            )
                # add residual sum to model, if any
                if len(skip_layers):
                    res_sum = torch.nn.Sequential(
                        LambdaLayer(lambda x: torch.stack(x, dim=0).sum(dim=0)),
                        torch.nn.BatchNorm1d(num_features=out_filters[layer_id]*self.wsf)
                    )
                    model.add(
                        layer_id=f"resid_{layer_id}", 
                        operation=res_sum, 
                        input_ids=[f"layer_{layer_id}-b{pool_block}"] + \
                            [f"layer_{skip_layer}-b{pool_block}" for skip_layer in skip_layers])
            # add a fatorize layer if in pool_layers
            if layer_id in pool_layers:
                factor_op = self.factorized_reduction_layer(
                        in_channels=out_filters[layer_id] * self.wsf,
                        out_channels=out_filters[layer_id+1] * self.wsf,
                        reduction_factor = self.reduction_factor
                        )
                model.add(
                    layer_id=f"pool_from_{layer_id}_to_{layer_id+1}",
                    operation=factor_op
                )
        
        if self.flatten_mode == "global_avg_pool" or self.flatten_mode == "gap":
            inp_c = out_filters[-1] * self.wsf
            flatten_op = LambdaLayer(lambda x: torch.mean(x, dim=-1))
        elif self.flatten_mode == "flatten":
            inp_c = (
                self.inputs.Layer_attributes["shape"][0]
                // (self.reduction_factor ** len(pool_layers))
                * out_filters[-1] * self.wsf
            )
            flatten_op = torch.nn.Flatten()
        else:
            raise Exception("cannot understand flatten_op: %s" % self.flatten_mode)
        model.add("flatten", flatten_op)
        
        model.add("fc", torch.nn.Sequential(
            torch.nn.Dropout(self.dropout_rate),
            torch.nn.Linear(inp_c, self.fc_units),
            torch.nn.BatchNorm1d(self.fc_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.dropout_rate),
        ))
        model.add("out", torch.nn.Sequential(
            torch.nn.Linear(self.fc_units, self.outputs.Layer_attributes["units"]),
            get_torch_layer(self.outputs.Layer_attributes["activation"]),
        ))
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