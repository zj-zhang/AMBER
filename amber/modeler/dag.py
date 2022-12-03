"""represent neural network computation graph
as a directed-acyclic graph from a list of 
architecture selections

"""

import numpy as np
import warnings
from ..architect.modelSpace import ModelSpace

# for general child
from .child import DenseAddOutputChild, EnasAnnModel, EnasCnnModel
from .. import backend as F
from ..backend import Operation, ComputationNode, get_layer_shortname
from .supernet import EnasAnnDAG, EnasConv1dDAG, EnasConv1DwDataDescrption

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
        elif arg.lower() == 'enasanndag':
            return EnasAnnDAG
        elif arg.lower() == 'enasconv1ddag':
            return EnasConv1dDAG
        elif arg == 'EnasConv1DwDataDescrption':
            return EnasConv1DwDataDescrption
    elif callable(arg):
        return arg
    else:
        raise ValueError("Could not understand the DAG func:", arg)


class DAG:
    """Construct a feed-forward neural network (FFNN) represented by a directed acyclic graph (DAG).

    While a simple, linear and sequential neural network model is also a DAG, here we are trying to build more flexible,
    generalizable branching models. In other words, the primary use is to construct a block-sparse FFNN, to create an
    specific inductive bias for a specific question; although one may use it in conv nets or other architectures stronger with
    inductive biases as well.

    Note that we are re-using the skip connection searching algorithms designed for building residual connections,
    but instead use it to build inter-layer connections without the "stem" connections in a ResNet. That is, the residual
    connections summed to the output of the currenct layer is now concatenated as input to the layer. By construction,
    it is possible that a node has no input, and these nodes will be removed in :func:`_remove_disconnected_nodes`.

    Parameters
    ----------
    arc_seq : list, or numpy.array
        a list of integers, each is a token for neural network architecture specific to a model space
    model_space : amber.architect.ModelSpace
        model space to sample model architectures from. Necessary for mapping token integers to operations.
    input_node : amber.modeler.ComputationNode, or list
        a list of input layers/nodes; in case of a single input node, use a single element list
    output_node : amber.modeler.ComputationNode, or list
        output node configuration
    with_skip_connection : bool
        if False, disable inter-layers connections (i.e. skip-layer connections). Default is True.
    with_input_blocks : bool
        if False, disable connecting partial inputs to intermediate layers. Default is True.

    Returns
    --------
    model : amber.backend.Model
        a constructed model using keras Model API
    """
    def __init__(self, arc_seq, model_space, input_node, output_node,
                 with_skip_connection=True,
                 with_input_blocks=True, *args, **kwargs):
        self.arc_seq = np.array(arc_seq)
        self.model_space = model_space
        assert isinstance(self.model_space, ModelSpace), \
            TypeError(f"model_space must be of Type amber.architect.ModelSpace; got {type(self.model_space)}")
        self.num_layers = len(self.model_space)
        self.input_node = input_node
        if isinstance(self.input_node, ComputationNode): self.input_node = [self.input_node]
        assert all([not x.is_built for x in self.input_node]), ValueError("input_node must not have been built")
        self.output_node = output_node
        assert isinstance(self.output_node, ComputationNode), \
            TypeError(f"output_node must be of Type amber.architect.Operation; got {type(self.output_node)}")
        assert not output_node.is_built, ValueError("output_node must not have been built")
        self.with_skip_connection = with_skip_connection
        self.with_input_blocks = with_input_blocks
        self.model = None
        self.nodes = []

    def _build_dag(self):
        if self.with_input_blocks:
            assert type(self.input_node) in (list, tuple), "If ``with_input_blocks=True" \
                                                           "``, ``input_node`` must be " \
                                                           "array-like. " \
                                                           "Current type of input_node is %s and" \
                                                           " with_input_blocks=%s" % (
                                                               type(self.input_node), self.with_input_blocks)
        assert type(self.output_node) is ComputationNode

        nodes = self._init_nodes()
        nodes = self._remove_disconnected_nodes(nodes)

        # build order is essential here
        node_list = self.input_node + nodes + [self.output_node]
        for node in node_list:
            node.build(F=F)

        self.model = F.Model(inputs=[x.operation_layer for x in self.input_node],
                           outputs=[self.output_node.operation_layer])
        self.nodes = nodes
        return self.model

    def _init_nodes(self):
        """first read through the architecture sequence to initialize the nodes"""
        arc_pointer = 0
        nodes = []
        for layer_id in range(self.num_layers):
            arc_id = self.arc_seq[arc_pointer]
            op = self.model_space[layer_id][arc_id]
            parent = []
            node_ = ComputationNode(op, node_name="L%i_%s" % (layer_id, get_layer_shortname(op)))
            if self.with_input_blocks:
                inp_bl = np.where(self.arc_seq[arc_pointer + 1: arc_pointer + 1 + len(self.input_node)] == 1)[0]
                for i in inp_bl:
                    parent.append(self.input_node[i])
                    self.input_node[i].child.append(node_)
            else:
                if layer_id == 0:
                    for n in self.input_node:
                        n.child.append(node_)
                        parent.append(n)
            if self.with_skip_connection and layer_id > 0:
                skip_con = np.where(
                    self.arc_seq[arc_pointer + 1 + len(self.input_node) * self.with_input_blocks: arc_pointer + 1 + len(
                        self.input_node) * self.with_input_blocks + layer_id] == 1)[0]
                # print(layer_id, skip_con)
                for i in skip_con:
                    parent.append(nodes[i])
                    nodes[i].child.append(node_)
            else:
                if layer_id > 0:
                    parent.append(nodes[-1])
                    nodes[-1].child.append(node_)
                # leave first layer without parent, so it
                # will be connected to default input node
            node_.parent = parent
            nodes.append(node_)
            arc_pointer += 1 + int(self.with_input_blocks) * len(self.input_node) + int(
                self.with_skip_connection) * layer_id
        # print('initial', nodes)
        return nodes

    def _remove_disconnected_nodes(self, nodes):
        """now need to deal with loose-ends: node with no parent, or no child
        """
        # CHANGE: add regularization to default input
        # such that all things in default_input would be deemed
        # un-important ZZ 2019.09.14
        # default_input_node = ComputationNode(State('Identity', name='default_input'), node_name="default_input")
        # default_input_node = ComputationNode(State('dropout', rate=0.999,  name='default_input'), node_name="default_input")
        # create an information bottleneck
        default_input_node = ComputationNode(Operation('dense', units=1, activation='linear', name='default_input'),
                                             node_name="default_input")
        # CHANGE: default input node cannot include every input node
        # otherwise will overwhelm the architecture. ZZ 2019.09.13
        # default_input_node.parent = self.input_node
        default_input_node.parent = [x for x in self.input_node if len(x.child) == 0]
        if default_input_node.parent:
            for x in self.input_node:
                if len(x.child) == 0:
                    x.child.append(default_input_node)
            has_default = True
        else:
            has_default = False
        is_default_intermediate = False
        # tmp_nodes: a tmp queue of connected/must-have nodes
        tmp_nodes = []
        for node in nodes:
            # filter out invalid parent nodes
            node.parent = [x for x in node.parent if x in tmp_nodes or x in self.input_node]
            # if no parent left, try to connect to default_input
            # otherwise, throw away as invalid
            if not node.parent:
                if has_default:
                    node.parent.append(default_input_node)
                    default_input_node.child.append(node)
                    is_default_intermediate = True
                else:
                    continue
            # if no child, connect to output
            if not node.child:
                self.output_node.parent.append(node)
                node.child.append(self.output_node)
            tmp_nodes.append(node)
        nodes = tmp_nodes
        # print('after filter', nodes)

        if has_default and not is_default_intermediate:
            default_input_node.child.append(self.output_node)
            self.output_node.parent.append(default_input_node)
            # for node in self.input_node:
            #    if not node.child:
            #        node.child.append(self.output_node)
            #        self.output_node.parent.append(node)
        if has_default:
            nodes = [default_input_node] + nodes
        return nodes


# these names are very confusing... FZZ 2022.5.8
class InputBlockDAG(DAG):
    """Add intermediate outputs to each level of network hidden layers. Based on DAG

    Compared to DAG, the difference is best illustrated by an example::

        |Input_A  Input_B   Input_C  Input_D      |
        |-------  -------   -------  -------      |
        |    |      |           |      |          |
        |   Hidden_AB          Hidden_CD          |
        |    /       |         |      \\           |
        |   /        Hidden_ABCD       \\          |
        | add_out1      |             add_out2    |
        |            Hidden_2                     |
        |               |                         |
        |             Output                      |

    In :class:`amber.modeler.dag.DAG`, *add_out1* and *add_out2* will NOT be added. The loss and out1 and out2 will be
    the same as output, but with a lower weight of 0.1.

    See also
    ----------
    :class:`amber.modeler.dag.DAG`: the base class.
    :class:`amber.modeler.dag.InputBlockAuxLossDAG`: add more auxillary outputs whenever two inputs meet.


    Returns
    -------
    model : amber.modeler.child.DenseAddOutputChild
        a subclass of keras Model API with multiple intermediate outputs predicting the same label
    """
    def __init__(self, add_output=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.with_input_blocks, "`InputBlockDAG` class only handles `with_input_blocks=True`"
        self.added_output_nodes = []
        self.add_output = add_output

    def _build_dag(self):
        assert type(self.input_node) in (list, tuple), "If ``with_input_blocks=True" \
                                                       "``, ``input_node`` must be " \
                                                       "array-like. " \
                                                       "Current type of input_node is %s and " \
                                                       "with_input_blocks=%s" % (
                                                           type(self.input_node), self.with_input_blocks)
        assert type(self.output_node) is ComputationNode

        nodes = self._init_nodes()
        nodes = self._remove_disconnected_nodes(nodes)

        # build order is essential here
        node_list = self.input_node + nodes + self.added_output_nodes + [self.output_node]
        for node in node_list:
            node.build(F=F)

        self.nodes = nodes
        self.model = DenseAddOutputChild(
            inputs=[x.operation_layer for x in self.input_node],
            outputs=[self.output_node.operation_layer] + [n.operation_layer for n in self.added_output_nodes],
            nodes=self.nodes
        )
        return self.model

    def _init_nodes(self):
        """first read through the architecture sequence to initialize the nodes,
        whenever a input block is connected, add an output tensor afterwards
        """
        arc_pointer = 0
        nodes = []
        for layer_id in range(self.num_layers):
            arc_id = self.arc_seq[arc_pointer]
            op = self.model_space[layer_id][arc_id]
            parent = []
            node_ = ComputationNode(op, node_name="L%i_%s" % (layer_id, get_layer_shortname(op)))
            inp_bl = np.where(self.arc_seq[arc_pointer + 1: arc_pointer + 1 + len(self.input_node)] == 1)[0]
            if any(inp_bl):
                for i in inp_bl:
                    parent.append(self.input_node[i])
                    self.input_node[i].child.append(node_)
                # do NOT add any additional outputs if this is
                #  the last layer..
                if self.add_output and layer_id != self.num_layers - 1:
                    if type(self.output_node) is list:
                        assert len(self.output_node) == 1
                        self.output_node = self.output_node[0]
                    new_out = ComputationNode(
                        operation=self.output_node.operation,
                        node_name="added_out_%i" % (len(self.added_output_nodes) + 1)
                    )
                    new_out.parent.append(node_)
                    self.added_output_nodes.append(new_out)
            if self.with_skip_connection and layer_id > 0:
                skip_con = np.where(
                    self.arc_seq[arc_pointer + 1 + len(self.input_node) * self.with_input_blocks: arc_pointer + 1 + len(
                        self.input_node) * self.with_input_blocks + layer_id] == 1)[0]
                # print(layer_id, skip_con)
                for i in skip_con:
                    parent.append(nodes[i])
                    nodes[i].child.append(node_)
            else:
                if layer_id > 0:
                    parent.append(nodes[-1])
                    nodes[-1].child.append(node_)
                # leave first layer without parent, so it
                # will be connected to default input node
            node_.parent = parent
            nodes.append(node_)
            arc_pointer += 1 + int(self.with_input_blocks) * len(self.input_node) + int(
                self.with_skip_connection) * layer_id
        # print('initial', nodes)
        return nodes


# these names are very confusing... FZZ 2022.5.8
class InputBlockAuxLossDAG(InputBlockDAG):
    """Add intermediate outputs whenever two input blocks first meet and merge.

    Compared to InputBlockDAG, the difference is best illustrated by an example::

        |Input_A  Input_B   Input_C  Input_D      |
        |-------  -------   -------  -------      |
        |    |      |           |      |          |
        |   Hidden_AB          Hidden_CD          |
        |    /       |         |      \\           |
        |   /        Hidden_ABCD       \\          |
        | add_out1      |    \\        add_out2    |
        |           Hidden_2  add_out3            |
        |               |                         |
        |             Output                      |

    In :class:`amber.modeler.dag.InputBlockDAG`, *add_out3* will NOT be added, since only immediate layers to input blocks
    (i.e. Hidden_AB and Hidden_CD) will be added output.

    See also
    ---------
    :class:`amber.modeler.dag.DAG`.
    :class:`amber.modeler.dag.InputBlockDAG`.

    Returns
    -------
    model : amber.modeler.child.DenseAddOutputChild
        a subclass of keras Model API with multiple intermediate outputs predicting the same label
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.add_output, "InputBlockAuxLossDAG must have `add_output=True`"
        # index is the List index in the output of `model.evaluate` method
        self.input_block_loss_mapping = {}

    def _build_dag(self):
        assert type(self.input_node) in (list, tuple), "If ``with_input_blocks=True" \
                                                       "``, ``input_node`` must be " \
                                                       "array-like. " \
                                                       "Current type of input_node is %s and with_input_blocks=%s" % (
                                                           type(self.input_node), self.with_input_blocks)
        assert type(self.output_node) is ComputationNode

        nodes = self._init_nodes()
        nodes = self._remove_disconnected_nodes(nodes)

        # build order is essential here
        node_list = self.input_node + nodes + self.added_output_nodes + [self.output_node]
        for node in node_list:
            node.build(F=F)

        self.nodes = nodes
        self.model = DenseAddOutputChild(
            inputs=[x.operation_layer for x in self.input_node],
            outputs=[self.output_node.operation_layer] + [n.operation_layer for n in self.added_output_nodes],
            nodes=self.nodes,
            # lcr: lowest common root
            block_loss_mapping=self.input_block_loss_mapping
        )
        return self.model

    def _aux_loss(self, nodes):
        input_blocks = [x.node_name for x in self.input_node]
        # inp_head keeps a queue of all leading branches for each input block
        inp_heads = {inp: None for inp in input_blocks}
        # inp_pair_root keeps the tree subroot for each pair of input blocks
        inp_pair_roots = {(b1, b2): 'None' for b1 in input_blocks for b2 in input_blocks}
        # root_leaves keeps all leaves for all nodes
        leaves_cardinality = {n.node_name: set([]) for n in nodes}
        for n in nodes:
            # get inputs to this layer and update heads
            _inputs = [x.node_name for x in n.parent if x.operation.Layer_type == 'input']
            inp_heads.update({x: n.node_name for x in _inputs})

            # get the set of parents nodes that are not input_blocks
            _ops = set([x.node_name for x in n.parent if x.operation.Layer_type != 'input'])

            # update leave cardinality
            for leaf in _inputs + [l for x in _ops for l in leaves_cardinality[x]]:
                leaves_cardinality[n.node_name].add(leaf)

            # update heads if connected to this layer
            inp_heads.update({x: n.node_name for x in input_blocks if inp_heads[x] in _ops})

            # update inp_pair_roots if new inp_heads met each other
            for b1 in input_blocks:
                for b2 in input_blocks:
                    if inp_pair_roots[(b1, b2)] != 'None':
                        continue
                    head1 = inp_heads[b1]
                    head2 = inp_heads[b2]
                    if head1 == head2 == n.node_name:
                        inp_pair_roots[(b1, b2)] = n.node_name

        aux_loss_nodes = []
        layer2loss = {}
        node_index = {node.node_name: node for node in nodes}
        for t in sorted(set(inp_pair_roots.values())):
            if t == 'None' or node_index[t] == nodes[-1]:
                layer2loss[t] = None  # None means look at the final loss, no aux loss
                continue
            else:
                new_out = ComputationNode(operation=self.output_node.operation,
                                          node_name="add_out_%i" % (len(aux_loss_nodes) + 1))
                new_out.parent.append(node_index[t])
                aux_loss_nodes.append(new_out)
                layer2loss[t] = len(aux_loss_nodes) + 1
        self.added_output_nodes = aux_loss_nodes

        for b1, b2 in inp_pair_roots:
            self.input_block_loss_mapping[(b1, b2)] = layer2loss[inp_pair_roots[(b1, b2)]]
        return

    def _init_nodes(self):
        arc_pointer = 0
        nodes = []
        for layer_id in range(self.num_layers):
            arc_id = self.arc_seq[arc_pointer]
            op = self.model_space[layer_id][arc_id]
            parent = []
            node_ = ComputationNode(op, node_name="L%i_%s" % (layer_id, get_layer_shortname(op)))
            inp_bl = np.where(self.arc_seq[arc_pointer + 1: arc_pointer + 1 + len(self.input_node)] == 1)[0]
            if any(inp_bl):
                for i in inp_bl:
                    parent.append(self.input_node[i])
                    self.input_node[i].child.append(node_)
            if self.with_skip_connection and layer_id > 0:
                skip_con = np.where(
                    self.arc_seq[arc_pointer + 1 + len(self.input_node) * self.with_input_blocks: arc_pointer + 1 + len(
                        self.input_node) * self.with_input_blocks + layer_id] == 1)[0]
                # print(layer_id, skip_con)
                for i in skip_con:
                    parent.append(nodes[i])
                    nodes[i].child.append(node_)
            else:
                if layer_id > 0:
                    parent.append(nodes[-1])
                    nodes[-1].child.append(node_)
                # leave first layer without parent, so it
                # will be connected to default input node
            node_.parent = parent
            nodes.append(node_)
            arc_pointer += 1 + int(self.with_input_blocks) * len(self.input_node) + int(
                self.with_skip_connection) * layer_id
        # print('initial', nodes)
        self._aux_loss(nodes)
        return nodes


