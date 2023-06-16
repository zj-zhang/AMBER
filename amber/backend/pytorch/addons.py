import torch
import abc
from ..amber_ops import AddOn
from typing import Optional


def get_addons(addon, base_layer):
    assert isinstance(addon, AddOn)
    if addon.AddOnType == 'L1WeightDecay':
        return L1WeightDecay(base_layer, weight_decay=addon.AddOn_attributes.get('weight_decay', 1e-3))
    elif addon.AddOnType == 'L2WeightDecay':
        return L2WeightDecay(base_layer, weight_decay=addon.AddOn_attributes.get('weight_decay', 1e-3))
    elif addon.AddOnType == 'BatchNorm':
        return BatchNorm(base_layer, dim=addon.AddOn_attributes.get('dim'))        
    else:
        raise ValueError("Unknown AddOn Type: {}".format(addon.AddOnType))


class BatchNorm(torch.nn.Module):
    def __init__(self, module, dim: Optional[int]=None):
        super().__init__()
        self.module = module
        if dim is None:
            self.dim = self._infer_shape()
        else:
            self.dim = dim
        if self.dim == 1:
            self.bn = torch.nn.BatchNorm1d(num_features=self.module.out_features)
        elif self.dim == 2:
            self.bn = torch.nn.BatchNorm2d(num_features=self.module.out_features)
        elif self.dim == 3:
            self.bn = torch.nn.BatchNorm3d(num_features=self.module.out_features)

    def _inter_shape(self):
        if isinstance(self.module, (torch.nn.Linear, torch.nn.Conv1d)):
            return 1
        elif isinstance(self.module, (torch.nn.Conv2d)):
            return 2
        elif isinstance(self.module, (torch.nn.Conv3d)):
            return 3
        else:
            raise TypeError("Cannot determine output dim for module: {}".format(self.module))
    
    def forward(self, *args, **kwargs):
        return self.bn(self.module(*args, **kwargs))


# from https://github.com/szymonmaszke/torchlayers/blob/master/torchlayers/regularization.py#L150
class WeightDecay(torch.nn.Module):
    def __init__(self, module, weight_decay, name: str = None):
        if weight_decay <= 0.0:
            raise ValueError(
                "Regularization's weight_decay should be greater than 0.0, got {}".format(
                    weight_decay
                )
            )

        super().__init__()
        self.module = module
        self.weight_decay = weight_decay
        self.name = name

        self.hook = self.module.register_full_backward_hook(self._weight_decay_hook)

    def remove(self):
        self.hook.remove()

    def _weight_decay_hook(self, *_):
        if self.name is None:
            for param in self.module.parameters():
                if param.grad is None or torch.all(param.grad == 0.0):
                    param.grad = self.regularize(param)
        else:
            for name, param in self.module.named_parameters():
                if self.name in name and (
                    param.grad is None or torch.all(param.grad == 0.0)
                ):
                    param.grad = self.regularize(param)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def extra_repr(self) -> str:
        representation = "weight_decay={}".format(self.weight_decay)
        if self.name is not None:
            representation += ", name={}".format(self.name)
        return representation

    @abc.abstractmethod
    def regularize(self, parameter):
        pass


class L2WeightDecay(WeightDecay):
    r"""Regularize module's parameters using L2 weight decay.

    Example::

        import torchlayers as tl

        # Regularize only weights of Linear module
        regularized_layer = tl.L2(tl.Linear(30), weight_decay=1e-5, name="weight")

    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L2` regularization use `remove()` method.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").

    """

    def regularize(self, parameter):
        return self.weight_decay * parameter.data


class L1WeightDecay(WeightDecay):
    """Regularize module's parameters using L1 weight decay.

    Example::

        import torchlayers as tl

        # Regularize all parameters of Linear module
        regularized_layer = tl.L1(tl.Linear(30), weight_decay=1e-5)

    .. note::
            Backward hook will be registered on `module`. If you wish
            to remove `L1` regularization use `remove()` method.

    Parameters
    ----------
    module : torch.nn.Module
        Module whose parameters will be regularized.
    weight_decay : float
        Strength of regularization (has to be greater than `0.0`).
    name : str, optional
        Name of parameter to be regularized (if any).
        Default: all parameters will be regularized (including "bias").

    """

    def regularize(self, parameter):
        return self.weight_decay * torch.sign(parameter.data)