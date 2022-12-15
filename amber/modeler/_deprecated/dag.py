"""represent neural network computation graph
as a directed-acyclic graph from a list of 
architecture selections

"""

import numpy as np
import warnings
from ..architect.modelSpace import ModelSpace

from .. import backend as F
from ..backend import Operation, ComputationNode, get_layer_shortname
from .supernet import EnasAnnModelBuilder, EnasCnnModelBuilder, EnasCNNwDataDescriptor
from .sparse_ffnn.keras_multiio import DenseAddOutputChild

def get_dag(arg):
    """Getter method for getting a DAG class from a string

    DAG refers to the underlying tensor computation graphs for child models. Whenever possible, we prefer to use Keras
    Model API to get the job done. For ENAS, the parameter-sharing scheme is implemented by tensorflow.

    Parameters
    ----------
    arg : str or callable
        return the DAG constructor corresponding to that identifier; if is callable, assume it's a DAG constructor
        already, do nothing and return it

    Returns
    -------
    callable
        A DAG constructor
    """
    if arg is None:
        return None
    elif type(arg) is str:
        if arg.lower() == 'dag':
            return DAG
        elif arg.lower() == 'inputblockdag':
            return InputBlockDAG
        elif arg.lower() == 'inputblockauxlossdag':
            return InputBlockAuxLossDAG
    elif callable(arg):
        return arg
    else:
        raise ValueError("Could not understand the DAG func:", arg)

