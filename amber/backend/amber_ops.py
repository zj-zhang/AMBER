"""amber_ops are universal amber operators; must be backend-agnostic
"""


def get_layer_shortname(layer):
    """Get the short name for a computational operation of a layer, useful in converting a Layer object to a string as
    ID or when plotting

    Parameters
    ----------
    layer : amber.architect.Operation
        The ``Operation`` object for any layer.

    Returns
    -------
    sn : str
        The unique short name for this operation

    TODO
    -----
    Consider refactoring ``layer`` to ``operation``
    """
    if layer.Layer_type == 'conv1d':
        sn = "conv_f%s_k%s_%s" % (layer.Layer_attributes['filters'], layer.Layer_attributes['kernel_size'],
                                  layer.Layer_attributes['activation'])
        if 'dilation' in layer.Layer_attributes:
            sn += '_d%i' % layer.Layer_attributes['dilation']
        if 'dilation_rate' in layer.Layer_attributes:
            sn += '_d%i' % layer.Layer_attributes['dilation_rate']

    elif layer.Layer_type == 'denovo':
        sn = "%s_f%s_k%s" % ('regconv2d',
                             layer.Layer_attributes['filters'],
                             layer.Layer_attributes['kernel_size'])
        # sn = "%s_f%s_k%s"%('denovo', layer.Layer_attributes['filters'], layer.Layer_attributes['kernel_size'])

    elif layer.Layer_type == 'dense':
        sn = "%s_u%s_%s" % (layer.Layer_type,
                            layer.Layer_attributes['units'],
                            layer.Layer_attributes['activation'])

    elif layer.Layer_type == 'maxpool1d' or layer.Layer_type == 'avgpool1d':
        sn = layer.Layer_type

    elif layer.Layer_type == 'flatten' or layer.Layer_type == 'identity' or layer.Layer_type == 'globalmaxpool1d' or layer.Layer_type == 'globalavgpool1d':
        sn = layer.Layer_type

    elif layer.Layer_type == 'sfc':
        sn = layer.Layer_type

    else:
        sn = str(layer)
    return sn


class Operation(object):
    """The Amber internal holder for a computational operation at any layer

    Parameters
    ----------
    Layer_type : str
        The string for the operation type; supports most commonly used ``tf.keras.layers`` types

    kwargs :
        Operation/layer specifications are parsed through keyword arguments

    Attributes
    ----------
    Layer_type : str
        The string for the operation type.

    Layer_attributes : dict
        The dictionary that holds key-value pairs for all specification for this layer.

    Notes
    ------
    Any attributes that are not specified in ``Layer_attributes`` will use the default value as defined in
    ``tf.keras.layers``. For example, if you do not specify ``activation`` in ``Layer_attributes``, it will use ``linear``.

    Examples
    --------
    For example, to create a 1D-convolutional operation with ReLU activation, kernel size=8, number of kernels=32::

        >>> from amber.architect import State
        >>> op = State("conv1d", filters=32, kernel_size=8, activation='relu')

    """

    def __init__(self, Layer_type, **kwargs):
        Layer_type = Layer_type.lower()
        # assert Layer_type in [
        #    'conv1d', 'maxpool1d', 'avgpool1d',
        #    'conv2d', 'maxpool2d', 'avgpool2d',
        #    'lstm',
        #    'dense', 'input', 'identity',
        #    'dropout', 'sparsek_vec', 'batchnorm',
        #    'flatten', 'globalavgpool1d', 'globalavgpool2d', 'globalmaxpool1d', 'globalmaxpool1d',
        #    'data', 'denovo', 'sfc',
        #    'concatenate'
        # ]

        self.Layer_type = Layer_type
        self.Layer_attributes = kwargs

    def __str__(self):
        return "{}:{}".format(self.Layer_type, self.Layer_attributes)

    def __eq__(self, other):
        return self.Layer_type == other.Layer_type and self.Layer_attributes == other.Layer_attributes

    def __hash__(self):
        unroll_attr = ((x, self.Layer_attributes[x])
                       for x in self.Layer_attributes)
        return hash((self.Layer_type, unroll_attr))



class ComputationNode:
    """Computation Node is an analog to :class:`tf.keras.layers.Layer` to make branching and multiple input/output
    feed-forward neural network (FFNN) models, represented by a directed-acyclic graph (DAG) in AMBER.

    The reason we need ComputationNode is that an :class:`amber.architect.Operation` focus on token-level computations,
    but does not represent the connectivity patterns well enough. When it comes to building DAG-represented FFNNs, we need
    more fine-grained control over the graph connectivities and validities.

    This is a helper that provides building blocks for :class:`amber.modeler.DAG` to use, and is not intended to be
    used by itself.

    See also :class:`amber.modeler.dag.DAG`.

    Parameters
    ----------
    operation: amber.architect.Operation
        defines the operation in current layer
    node_name: str
        name of the node
    merge_op: tf.keras.layers.merge, optional
        operation for merging multiple inputs
    """
    def __init__(self, operation, node_name, merge_op=Operation('Concatenate')):
        assert type(operation) is Operation, "Expect operation is of type amber.backend.Operation, got %s" % type(
            operation)
        self.operation = operation
        self.node_name = node_name
        self.merge_op = merge_op
        self.parent = []
        self.child = []
        self.operation_layer = None
        self.merge_layer = None
        self.is_built = False

    def build(self, F):
        """Build the keras layer with merge operations if applicable

        when building a node, its parents must all be built already.

        Parameters
        ----------
        F : amber.backend
        """
        if self.parent:
            if len(self.parent) > 1:
                self.merge_layer = F.get_layer(
                    x=[p.operation_layer for p in self.parent],
                    op=self.merge_op)
            else:
                self.merge_layer = self.parent[0].operation_layer
        self.operation.Layer_attributes['name'] = self.node_name
        self.operation_layer = F.get_layer(x=self.merge_layer, op=self.operation)
        self.is_built = True
