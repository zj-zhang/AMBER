import torch
import numpy as np
from ..amber_ops import Operation


class LambdaLayer(torch.nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class ConcatLayer(torch.nn.Module):
    def __init__(self, take_sum=False):
        super(ConcatLayer, self).__init__()
        self.take_sum = take_sum

    def forward(self, x):
        out = torch.stack(x, dim=0)
        if self.take_sum:
            out = out.sum(dim=0)
        return out

class PermuteLayer(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)        


class GlobalAveragePooling1DLayer(torch.nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=-1)


class GlobalMaxPooling1DLayer(torch.nn.Module):
    def forward(self, x):
        return torch.max(x, dim=-1).values


def get_layer(x=None, op=None, custom_objects=None, with_bn=False):
    """Getter method for a Torch layer, including native pytorch implementation and custom layers that are not included.

    Parameters
    ----------
    x : torch.Tensor, torch.nn.Module or None
        The running dummy Tensor, or input Torch layer
    op : amber.architect.Operation, or callable
        The target layer to be built
    custom_objects : dict, or None
        Allow stringify custom objects by parsing a str->class dict
    with_bn : bool, optional
        If true, add batch normalization layers before activation

    Returns
    -------
    x : torch.nn.Module
        The built target layer connected to input x
    """
    custom_objects = custom_objects or {}
    if callable(op):
        layer = op()
    elif op.Layer_type == 'activation':
        actv_fn = op.Layer_attributes['activation']
        if actv_fn == 'relu':
            layer = torch.nn.ReLU()
        elif actv_fn == 'softmax':
            layer = torch.nn.Softmax()
        elif actv_fn == 'tanh':
            layer = torch.nn.Tanh()
        elif actv_fn == 'sigmoid':
            layer = torch.nn.Sigmoid()
        elif actv_fn == 'linear':
            layer = torch.nn.Identity()
        else:
            raise ValueError("unknown activation layer: %s" % actv_fn)

    elif op.Layer_type == 'dense':
        actv_fn = op.Layer_attributes.get('activation', 'linear')
        curr_shape = np.array(x.shape) if isinstance(x, torch.Tensor) else x.out_features
        assert len(curr_shape)==1, ValueError("dense layer must have 1-d prev layers")
        _list = [torch.nn.Linear(in_features=curr_shape[0], out_features=op.Layer_attributes['units'])]
        if with_bn:   _list.append(get_layer(x=x, op=Operation("batchnorm")))
        _list.append(get_layer(op=Operation("activation", activation=actv_fn)))
        layer = torch.nn.Sequential(* _list)

    elif op.Layer_type == 'conv1d':
        assert x is not None
        actv_fn = op.Layer_attributes.get('activation', 'linear')
        curr_shape = np.array(x.shape) if isinstance(x, torch.Tensor) else x.out_features
        assert len(curr_shape)==2, ValueError("conv1d layer must have 2-d prev layers")
        # XXX: pytorch assumes channel_first, unlike keras
        _list = [torch.nn.Conv1d(in_channels=curr_shape[-2], out_channels=op.Layer_attributes['filters'], 
                    kernel_size=op.Layer_attributes.get('kernel_size',1),
                    stride=op.Layer_attributes.get('strides', 1),
                    padding=op.Layer_attributes.get('padding', 0),
                    dilation=op.Layer_attributes.get('dilation_rate', 1),
                )]
        if with_bn:   _list.append(get_layer(x=x, op=Operation("batchnorm")))
        _list.append(get_layer(op=Operation("activation", activation=actv_fn)))
        layer = torch.nn.Sequential(* _list)

    # elif op.Layer_type == 'conv2d':
    #     if with_bn is True:
    #         assert x is not None
    #         actv_fn = op.Layer_attributes.get('activation', 'linear')
    #         x = tf.keras.layers.Conv2D(**op.Layer_attributes)(x)
    #         x = tf.keras.layers.BatchNormalization()(x)
    #         x = tf.keras.layers.Activation(actv_fn)(x)
    #         return x
    #     else:
    #         layer = tf.keras.layers.Conv2D(**op.Layer_attributes)
    
    elif op.Layer_type == 'batchnorm' or op.Layer_type == 'BatchNormalization'.lower():
        curr_shape = np.array(x.shape) if isinstance(x, torch.Tensor) else x.out_features
        if len(curr_shape) <= 2:
            layer = torch.nn.BatchNorm1d()
        elif len(curr_shape) == 3:
            layer = torch.nn.BatchNorm2d()
        elif len(curr_shape) == 4:
            layer = torch.nn.BatchNorm3d()

    elif op.Layer_type in ('maxpool1d','maxpooling1d'):
        layer = torch.nn.MaxPool1d(kernel_size=op.Layer_attributes.get('pool_size',2), stride=op.Layer_attributes.get('strides', None), padding=op.Layer_attributes.get('padding', 0))

    elif op.Layer_type in ('maxpool2d','maxpooling2d'):
        layer = torch.nn.MaxPool2d(kernel_size=op.Layer_attributes.get('pool_size',2), stride=op.Layer_attributes.get('strides', None), padding=op.Layer_attributes.get('padding', 0))

    elif op.Layer_type in ('avgpool1d', 'avgpooling1d', 'averagepooling1d'):
        layer = torch.nn.AvgPool1d(kernel_size=op.Layer_attributes.get('pool_size',2), stride=op.Layer_attributes.get('strides', None), padding=op.Layer_attributes.get('padding', 0))

    elif op.Layer_type in ('avgpool2d', 'avgpooling2d', 'averagepooling2d'):
        layer = torch.nn.AvgPool2d(kernel_size=op.Layer_attributes.get('pool_size',2), stride=op.Layer_attributes.get('strides', None), padding=op.Layer_attributes.get('padding', 0))

    elif op.Layer_type == 'flatten':
        layer = torch.nn.Flatten()

    elif op.Layer_type in ('globalavgpool1d', 'GlobalAveragePooling1D'.lower()):
        layer = GlobalAveragePooling1DLayer()

    elif op.Layer_type in ('globalavgpool2d', 'GlobalAveragePooling2D'.lower()):
        curr_shape = np.array(x.shape) if isinstance(x, torch.Tensor) else x.out_features
        layer = torch.nn.AdaptiveAvgPool2d(output_size=curr_shape[-3])

    elif op.Layer_type in ('globalmaxpool1d', 'GlobalMaxPooling1D'.lower()):
        layer = GlobalMaxPooling1DLayer()

    # elif op.Layer_type in ('globalmaxpool2d', 'GlobalMaxPooling2D'.lower()):
    #     layer =  tf.keras.layers.GlobalMaxPooling2D(**op.Layer_attributes)

    elif op.Layer_type == 'dropout':
        layer = torch.nn.Dropout(p=op.Layer_attributes.get('rate'))
    
    elif op.Layer_type == 'identity':
        layer = torch.nn.Identity()
    
    elif op.Layer_type == 'permute':
        layer = PermuteLayer(dims=op.Layer_attributes['dims'])

    elif op.Layer_type == 'concatenate':
        layer = ConcatLayer()
     
    elif op.Layer_type in custom_objects:
        return custom_objects[op.Layer_type](**op.Layer_attributes)(x)

    else:
        raise ValueError('Layer_type "%s" is not understood' % op.Layer_type)

    return layer

