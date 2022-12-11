from amber import backend as F
import numpy as np
from ..base import BaseModelBuilder
from ..architectureDecoder import ResConvNetArchitecture
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Add, Dense, Conv1D, MaxPooling1D, AveragePooling1D, \
    GlobalAveragePooling1D, Flatten, BatchNormalization, LeakyReLU, Dropout, Activation, Lambda
from tensorflow.keras import regularizers
from tensorflow.keras import constraints
from tensorflow.keras.models import Model


class ResidualCnnBuilder(BaseModelBuilder):
    """Function class for converting an architecture sequence tokens to a Keras model

    Parameters
    ----------
    inputs_op : amber.architect.modelSpace.F.Operation
    output_op : amber.architect.modelSpace.F.Operation
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
    ---------
    tensorflow.keras.models.Model : 
        a compiled keras model
    """

    def __init__(self, inputs_op, output_op, fc_units, flatten_mode, model_compile_dict, model_space,
                 dropout_rate=0.2, wsf=1, reduction_factor=4, add_conv1_under_pool=True, verbose=1, **kwargs):
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

    def __call__(self, model_states):
        model = self._convert(model_states, verbose=self.verbose)
        if model is not None:
            model.compile(**self.model_compile_dict)
        return model

    def _convert(self, arc_seq, verbose=True):
        out_filters, pool_layers = self.get_out_filters(self.model_space)

        inp = F.get_layer(x=None, op=self.inputs)
        # this is assuming all choices have the same out_filters
        stem_conv = F.Operation(
            'conv1d',
            kernel_size=8,
            filters=out_filters[0],
            activation="linear")
        x = self.res_layer(stem_conv, self.wsf, inp, name="stem_conv",
                           add_conv1_under_pool=self.add_conv1_under_pool)

        start_idx = 0
        layers = []
        for layer_id in range(len(self.model_space)):
            if verbose:
                print("start_idx=%i, layer id=%i, out_filters=%i x %i" % (
                    start_idx, layer_id, out_filters[layer_id], self.wsf))
            count = arc_seq[start_idx]
            this_layer = self.model_space[layer_id][count]
            if verbose:
                print(this_layer)
            if layer_id == 0:
                x = self.res_layer(this_layer, self.wsf, x, name="L%i" % layer_id,
                                   add_conv1_under_pool=self.add_conv1_under_pool)
            else:
                x = self.res_layer(this_layer, self.wsf, layers[-1], name="L%i" % layer_id,
                                   add_conv1_under_pool=self.add_conv1_under_pool)

            if layer_id > 0:
                skip = arc_seq[start_idx + 1: start_idx + layer_id + 1]
                skip_layers = [layers[i]
                               for i in range(len(layers)) if skip[i] == 1]
                if verbose:
                    print("skip=%s" % skip)
                if len(skip_layers):
                    skip_layers.append(x)
                    x = Add(name="L%i_resAdd" % layer_id)(skip_layers)
                x = BatchNormalization(name="L%i_resBn" % layer_id)(x)

            if self.dropout_rate != 0:
                x = Dropout(
                    self.dropout_rate,
                    name="L%i_dropout" %
                    layer_id)(x)

            layers.append(x)
            if layer_id in pool_layers:
                pooled_layers = []
                for i, layer in enumerate(layers):
                    pooled_layers.append(
                        self.factorized_reduction_layer(
                            layer,
                            out_filters[layer_id + 1] * self.wsf,
                            name="pool_at_%i_from_%i" % (layer_id, i),
                            reduction_factor = self.reduction_factor)
                    )
                if verbose:
                    print("pooled@%i, %s" % (layer_id, pooled_layers))
                layers = pooled_layers

            start_idx += 1 + layer_id
            if verbose:
                print('-' * 80)

        # fully-connected layer
        if self.flatten_mode == 'gap':
            x = GlobalAveragePooling1D()(x)
        elif self.flatten_mode == 'flatten':
            x = Flatten()(x)
        else:
            raise Exception("Unknown flatten mode: %s" % self.flatten_mode)
        if self.dropout_rate != 0:
            x = Dropout(self.dropout_rate)(x)
        x = Dense(units=self.fc_units, activation="relu")(x)

        out = F.get_layer(x=x, op=self.outputs)

        model = Model(inputs=inp, outputs=out)
        return model

    @staticmethod
    def factorized_reduction_layer(inp, out_filter, name, reduction_factor=4):
        x = Conv1D(out_filter,
                   kernel_size=1,
                   strides=1,
                   kernel_initializer='he_normal',
                   use_bias=False,
                   padding="same",
                   name=name
                   )(inp)
        x = MaxPooling1D(
            pool_size=reduction_factor,
            strides=reduction_factor,
            padding="same")(x)
        return x

    @staticmethod
    def res_layer(layer, width_scale_factor, inputs, l2_reg=5e-7,
                  name="layer", add_conv1_under_pool=True):
        if layer.Layer_type == 'conv1d':
            activation = layer.Layer_attributes['activation']
            num_filters = width_scale_factor * \
                layer.Layer_attributes['filters']
            kernel_size = layer.Layer_attributes['kernel_size']
            dilation = layer.Layer_attributes.get('dilation', 1)
            x = Conv1D(num_filters,
                       kernel_size=kernel_size,
                       strides=1,
                       padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=regularizers.l2(l2_reg),
                       kernel_constraint=constraints.max_norm(0.9),
                       use_bias=False,
                       name="%s_conv" % name if dilation == 1 else "%s_conv_d%i" % (
                           name, dilation),
                       dilation_rate=dilation
                       )(inputs)
            x = BatchNormalization(name="%s_bn" % name)(x)
            if activation in ("None", "linear"):
                pass
            elif activation in ("relu", "sigmoid", "tanh", "softmax", "elu"):
                x = Activation(
                    activation, name="%s_%s" %
                    (name, activation))(x)
            elif activation == "leaky_relu":
                x = LeakyReLU(alpha=0.2, name="%s_%s" % (name, activation))(x)
            else:
                raise Exception("Unknown activation: %s" % activation)
        elif layer.Layer_type == 'maxpool1d' or layer.Layer_type == 'avgpool1d':
            num_filters = width_scale_factor * \
                layer.Layer_attributes['filters']
            pool_size = layer.Layer_attributes['pool_size']
            if add_conv1_under_pool:
                x = Conv1D(num_filters,
                           kernel_size=1,
                           strides=1,
                           padding='same',
                           kernel_initializer='he_normal',
                           use_bias=False,
                           name="%s_maxpool_conv" % name
                           )(inputs)
                x = BatchNormalization(name="%s_bn" % name)(x)
                x = Activation("relu", name="%s_relu" % name)(x)
            else:
                x = inputs

            if layer.Layer_type == 'maxpool1d':
                x = MaxPooling1D(
                    pool_size=pool_size,
                    strides=1,
                    padding='same',
                    name="%s_maxpool" %
                    name)(x)
            elif layer.Layer_type == 'avgpool1d':
                x = AveragePooling1D(
                    pool_size=pool_size,
                    strides=1,
                    padding='same',
                    name="%s_avgpool" %
                    name)(x)
            else:
                raise Exception("Unknown pool: %s" % layer.Layer_type)

        elif layer.Layer_type == 'identity':
            x = Lambda(lambda t: t, name="%s_id" % name)(inputs)
        else:
            raise Exception("Unknown type: %s" % layer.Layer_type)
        return x

    @staticmethod
    def get_out_filters(model_space):
        out_filters = []
        pool_layers = []
        for layer_id in range(len(model_space)):
            layer = model_space[layer_id]
            this_out_filters = [l.Layer_attributes['filters'] for l in layer]
            assert len(
                set(this_out_filters)) == 1, "EnasConv1dDAG only supports one identical number of filters per layer," \
                                             "but found %i in layer %s" % (
                                                 len(set(this_out_filters)), layer)
            if len(out_filters) and this_out_filters[0] != out_filters[-1]:
                pool_layers.append(layer_id - 1)

            out_filters.append(this_out_filters[0])
        # print(out_filters)
        # print(pool_layers)
        return out_filters, pool_layers
