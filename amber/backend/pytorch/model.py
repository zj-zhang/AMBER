import torch
from .cache import *

Model = torch.nn.Module


class Sequential(torch.nn.Sequential):
    def __init__(self, layers=None):
        layers = layers or []
        super().__init__(*layers)
    
    def add(layer):
        super().append(layer)


def get_metric(m):
    if callable(m):
        return m
    elif m.lower() == 'kl_div':
        return torch.nn.KLDivLoss()
    else:
        raise Exception("cannot understand metric type: %s" % m)

def get_loss(loss, y_true, y_pred):
    if type(loss) is str:
        loss = loss.lower()
        if loss == 'mse' or loss == 'mean_squared_error':
            pass
        elif loss == 'categorical_crossentropy':
            pass
        elif loss == 'binary_crossentropy':
            loss_ = torch.nn.BCELoss(reduction='none')(input=y_pred, target=y_true.float())
        elif loss == 'nllloss_with_logits':
            #loss_ = torch.nn.NLLLoss()(input=torch.nn.LogSoftmax(dim=-1)(y_pred), target=y_true.long())
            loss_ = torch.nn.CrossEntropyLoss(reduction='none')(input=torch.nn.LogSoftmax(dim=-1)(y_pred), target=y_true.long())
        else:
            raise Exception("cannot understand string loss: %s" % loss)
    elif type(loss) is callable:
        loss_ = loss(y_true, y_pred)
    else:
        raise TypeError("Expect loss argument to be str or callable, got %s" % type(loss))
    return loss_


def trainable_variables(scope=None):
    scope = scope or ''
    if isinstance(scope, torch.nn.Module):
        return scope.parameters()
    elif type(scope) is str:
        return [param for name, param in GLOBAL_DEFAULT_GRAPH.param_cache.items() if scope in name]


def get_optimizer(opt, parameters, opt_config=None):
    opt_config = opt_config or {'lr':0.01}
    if callable(opt):
        opt_ = opt
    elif type(opt) is str:
        if opt.lower() == 'adam':
            opt_ = torch.optim.Adam
        elif opt.lower() == 'sgd':
            opt_ = torch.optim.SGD
        else:
            raise Exception(f"unknown opt {opt}")
    else:
        raise Exception(f"unknown opt {opt}")
    return opt_(parameters,  **opt_config)


def get_train_op(loss, variables, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
