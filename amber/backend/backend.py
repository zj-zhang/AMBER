"""This file defines the unified tensor and autograd framework interface required by AMBER.
The principles of this interface:
* There should be as few interfaces as possible.
* The interface is used by AMBER system so it is more important to have
  clean definition rather than convenient usage.
* Default arguments should be avoided.
* Keyword or positional arguments should be avoided.
* Argument type should be easier to understand.
It is recommended the frameworks implement all the interfaces. However, it is
also OK to skip some. The generated backend module has an ``is_enabled`` function
that returns whether the interface is supported by the framework or not.
"""


def create_weight(name, shape, initializer=None, trainable=True, seed=None):
    pass

def create_bias(name, shape, trainable=True, initializer=None):
    pass

def get_train_op(loss, parameters, optimizer, **kwargs):
    pass

def get_layer():
    pass

def get_loss(loss, y_true, y_pred):
    pass

def get_metric(m):
    pass