"""supernet built using static backend libraries
that is, only TF1.X at this point
"""

import numpy as np
import warnings
from ..child import DenseAddOutputChild, EnasAnnModel, EnasCnnModel
from ... import backend as F
from ...backend import Operation, ComputationNode, get_layer_shortname


class EnasAnnDAG:
    """EnasAnnDAG is a DAG model builder for using the weight sharing method for child models.

    This class deals with the feed-forward neural network (FFNN). The weight sharing is between all Ws for
    different hidden units sizes - that is, a larger hidden size always includes the smaller ones.

    Parameters
    -----------
    model_space: amber.architect.ModelSpace
        model space to search architectures from
    input_node: amber.architect.Operation, or list
        one or more input layers, each is a block of input features
    output_node: amber.architect.Operation, or list
        one or more output layers, each is a block of output labels
    model_compile_dict: dict
        compile dict for child models
    drop_probs: float, or List
        dropout rate; if float, then constant for every layer
    session: amber.backend.Session
        tensorflow session that hosts the computation graph; should use the same session as controller for
        sampling architectures
    with_skip_connection: bool
        if False, disable inter-layer connections. Default is True.
    with_input_blocks: bool
        if False, disable connecting input layers to hidden layers. Default is True.
    with_output_blocks: bool
        if True, add another architecture representation vector, to connect intermediate layers to output
        blocks.
    controller: amber.architect.MultiIOController, or None
        connect a controller to enable architecture sampling; if None, can only train fixed architecture manually
        provided
    feature_model : amber.backend.Model, or None
        If specified, use the provided upstream model for pre-transformations of inputs, instead of taking
        the raw input features.
    feature_model_trainable: bool, or None
        Boolean of whether pass gradients to the feature model.
    child_train_op_kwargs: dict, or None
        Keyword arguments passed to :func:`model.fit`.
    name: str
        a string name for this instance
    """
    def __init__(self,
                 model_space,
                 input_node,
                 output_node,
                 model_compile_dict,
                 session,
                 drop_probs=0.1,
                 l1_reg=0.0,
                 l2_reg=0.0,
                 with_skip_connection=True,
                 with_input_blocks=True,
                 with_output_blocks=False,
                 controller=None,
                 feature_model=None,
                 feature_model_trainable=None,
                 child_train_op_kwargs=None,
                 name='EnasDAG'):

        #assert with_skip_connection == with_input_blocks == True, \
        #    "EnasAnnDAG must have with_input_blocks and with_skip_connection"
        self.model_space = model_space
        if not type(input_node) in (tuple, list):
            self.input_node = [input_node]
        else:
            self.input_node = input_node
        if not type(output_node) in (tuple, list):
            self.output_node = [output_node]
        else:
            self.output_node = output_node
        if session is None:
            self.session = F.Session()
        else:
            self.session = session
        self.model_compile_dict = model_compile_dict
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.with_skip_connection = with_skip_connection
        self.with_input_blocks = with_input_blocks
        self.with_output_blocks = with_output_blocks
        self.num_layers = len(model_space)
        self.name = name
        self.input_arc = None
        self.sample_w_masks = []
        self.feature_model = feature_model
        if self.feature_model is None:
            self.feature_model_trainable = False
        else:
            self.feature_model_trainable = feature_model_trainable or False
        self.child_train_op_kwargs = child_train_op_kwargs
        self.drop_probs = [drop_probs] * len(self.model_space) if type(drop_probs) is float else drop_probs
        assert len(self.drop_probs) == len(self.model_space)

        self._verify_args()
        self._create_params()
        if controller is None:
            self.controller = None
        else:
            self.controller = controller
            self._build_sample_arc()
        self._build_fixed_arc()
        F.init_all_params(self.session, var_scope=self.name)

        # for compatability with EnasConv1d
        self.train_fixed_arc = False

    def __call__(self, arc_seq, node_builder=None, *args, **kwargs):
        model = self._model(arc_seq)
        if node_builder is not None and arc_seq is not None:
            nb = node_builder(arc_seq)
            nodes = nb._init_nodes()
            nodes = nb._remove_disconnected_nodes(nodes)
            nodes = nb.input_node + nodes + [nb.output_node]
            model.nodes = nodes
        return model

    def set_controller(self, controller):
        assert self.controller is None, "already has inherent controller, disallowed; start a new " \
                                        "EnasAnnDAG instance if you want to connect another controller"
        self.controller = controller
        self._build_sample_arc()
        F.init_all_params(sess=self.session, var_scope="%s/sample" % self.name)

    def _verify_args(self):
        """verify vanilla ANN model space, input nodes, etc.,
         and configure internal attr. like masking steps"""
        # check the consistency of with_output_blocks and output_op
        if not self.with_output_blocks and len(self.output_node)>1:
            warnings.warn(
                "You specified `with_output_blocks=False`, but gave a List of output operations of length %i" %
                len(self.output_node), stacklevel=2)
        # model space
        assert len(set([tuple(self.model_space[i]) for i in
                        range(self.num_layers)])) == 1, "model_space for EnasDAG must be identical for all layers"
        layer_ = self.model_space[0]
        all_actv_fns = set([x.Layer_attributes['activation'] for x in layer_])
        assert len(all_actv_fns) == 1, "all operations must share the same activation function, got %s" % all_actv_fns
        self._actv_fn = all_actv_fns.pop()
        # assert self._actv_fn.lower() == "relu", "EnasDAG only supports ReLU activation function"
        self._weight_units = np.array([x.Layer_attributes['units'] for x in layer_], dtype=np.int32)
        self._weight_max_units = np.max(self._weight_units)

        # input nodes
        # _input_block_map: mapping from input_blocks indicators to feature indices
        self._input_block_map = np.zeros((len(self.input_node), 2), dtype=np.int32)  # n, [start, end]
        self.num_input_blocks = len(self.input_node)
        start_idx = 0
        for i in range(len(self.input_node)):
            n_feature = self.input_node[i].Layer_attributes['shape'][0]
            self._input_block_map[i] = [start_idx, start_idx + n_feature]
            start_idx += n_feature
        self._feature_max_size = start_idx

        # output node
        self._child_output_size = [n.Layer_attributes['units'] for n in self.output_node]
        self._child_output_func = [n.Layer_attributes['activation'] for n in self.output_node]
        self.num_output_blocks = len(self.output_node)
        self._output_block_map = np.array([
            [i * self._weight_max_units, (i + 1) * self._weight_max_units] for i in range(self.num_layers)],
            dtype=np.int32).reshape((self.num_layers, 2))

        # skip connections
        # _skip_conn_map: mapping from skip connection indicators to layer output indices
        self._skip_conn_map = {}
        start_map = np.array([[0, self._weight_max_units]], dtype=np.int32).reshape((1, 2))
        for i in range(1, self.num_layers):
            self._skip_conn_map[i] = start_map  # n, [start, end]
            start_map = np.concatenate([start_map,
                                        np.array([[i * self._weight_max_units, (i + 1) * self._weight_max_units]],
                                                 dtype=np.int32).reshape(1, 2)])

    def _create_params(self):
        self.w = []
        self.b = []
        input_max_size = self._input_block_map[-1][-1]
        with F.variable_scope(self.name):
            self.train_step = F.Variable(0, shape=(), dtype=F.int32, trainable=False, name="train_step")
            for layer_id in range(self.num_layers):
                with F.variable_scope("layer_{}".format(layer_id)):
                    self.w.append(F.create_parameter("Weight/w", shape=(
                    input_max_size + layer_id * self._weight_max_units, self._weight_max_units),
                                                            dtype=F.float32,
                                                            initializer='he_normal'))
                    self.b.append(F.create_parameter("Bias/b", shape=(self._weight_max_units,),
                                                            initializer='zeros',
                                                            dtype=F.float32))
            with F.variable_scope("stem_io"):
                self.child_model_input = F.placeholder(shape=(None, self._feature_max_size),
                                                                  dtype=F.float32,
                                                                  name="child_input")
                self.child_model_label = [F.placeholder(shape=(None, self._child_output_size[i]),
                                                                   dtype=F.float32,
                                                                   name="child_output_%i" % i)
                                          for i in range(len(self.output_node))]
                if self.feature_model is not None:
                    # data pipeline by Tensorflow.data.Dataset.Iterator
                    self.child_model_label_pipe = self.feature_model.y_it
                    self.child_model_input_pipe = self.feature_model.x_it
                # if also learn the connection of output_blocks, need to enlarge the output to allow
                # potential multi-inputs from different hidden layers
                if self.with_output_blocks:
                    self.w_out = [F.create_parameter(name="w_out_%i" % i, shape=(
                    self._weight_max_units * self.num_layers, self._child_output_size[i]),
                                                            initializer='he_normal')
                                  for i in range(len(self.output_node))]
                    self.b_out = [
                        F.create_parameter(name="b_out_%i" % i, shape=(self._child_output_size[i]),
                                                  initializer='zeros')
                        for i in range(len(self.output_node))]
                # otherwise, only need to connect to the last hidden layer
                else:
                    self.w_out = [F.create_parameter(name="w_out_%i" % i,
                                                    shape=(self._weight_max_units, self._child_output_size[i]),
                                                    initializer='he_normal')
                                  for i in range(len(self.output_node))]
                    self.b_out = [
                        F.create_parameter(name="b_out_%i" % i, shape=(self._child_output_size[i]), dtype=F.float32,
                                                  initializer='zeros')
                        for i in range(len(self.output_node))]

    def _build_sample_arc(self):
        """
        sample_output and sample_w_masks are the child model tensors that got built after getting a random sample
        from controller.sample_arc
        """
        with F.variable_scope("%s/sample" % self.name):
            self.connect_controller(self.controller)
            sample_output, sample_w_masks, sample_layer_outputs, sample_dropouts = self._build_dag(self.sample_arc)
            self.sample_model_output = sample_output
            self.sample_w_masks = sample_w_masks
            self.sample_layer_outputs = sample_layer_outputs
            ops = self._compile(w_masks=self.sample_w_masks, model_output=self.sample_model_output,
                                use_pipe=True,
                                # var_scope="%s/sample" % self.name
                                )
            self.sample_train_op = ops['train_op']
            self.sample_loss = ops['loss']
            self.sample_optimizer = ops['optimizer']
            self.sample_metrics = ops['metrics']
            self.sample_ops = ops
            self.sample_dropouts = sample_dropouts

    def _build_fixed_arc(self):
        """
        fixed_output and fixed_w_masks are the child model tensors built according to a fixed arc from user inputs
        """
        with F.variable_scope("%s/fixed" % self.name):
            self._create_input_ph()
            fixed_output, fixed_w_masks, fixed_layer_outputs, fixed_dropouts = self._build_dag(self.input_arc)
            self.fixed_model_output = fixed_output
            self.fixed_w_masks = fixed_w_masks
            self.fixed_layer_outputs = fixed_layer_outputs
            ops = self._compile(w_masks=self.fixed_w_masks, model_output=self.fixed_model_output,
                                use_pipe=False,
                                # var_scope = "%s/fixed" % self.name
                                )
            self.fixed_train_op = ops['train_op']
            self.fixed_loss = ops['loss']
            self.fixed_optimizer = ops['optimizer']
            self.fixed_metrics = ops['metrics']
            self.fixed_ops = ops
            self.fixed_dropouts = fixed_dropouts

    def _build_dag(self, arc_seq):
        """
        Shared DAG building process for both sampled arc and fixed arc
        Args:
            arc_seq:

        Returns:

        """
        w_masks = []
        layer_outputs = []
        start_idx = 0
        dropout_placeholders = []
        # if no addition of feature model, use dataset directly
        if self.feature_model is None:
            inputs = self.child_model_input
        # otherwise, need to connect to feature model output
        else:
            # TODO: for now, only use data_pipe for sample_arc
            if self.feature_model.pseudo_inputs_pipe is None or type(arc_seq) is list:
                # print('='*80)
                # print('used placeholder')
                inputs = self.feature_model.pseudo_inputs
            else:
                # print('='*80)
                # print('used data pipe')
                inputs = self.feature_model.pseudo_inputs_pipe
        for layer_id in range(self.num_layers):
            w = self.w[layer_id]
            b = self.b[layer_id]
            # column masking for output units
            num_units = F.embedding_lookup(self._weight_units, arc_seq[start_idx])
            col_mask = F.cast(F.less(F.range(0, limit=self._weight_max_units, delta=1), num_units), F.int32)
            start_idx += 1

            # input masking for with_input_blocks
            if self.with_input_blocks:
                inp_mask = arc_seq[start_idx: start_idx + self.num_input_blocks]
                inp_mask = F.boolean_mask(self._input_block_map, F.squeeze(inp_mask))
                new_range = F.range(0, limit=self._feature_max_size, dtype=F.int32)
                inp_mask = F.map_fn(lambda x: F.cast(F.logical_and(x[0] <= new_range, new_range < x[1]), dtype=F.int32),
                                     inp_mask)
                inp_mask = F.reduce_sum(inp_mask, axis=0)
                start_idx += self.num_input_blocks * self.with_input_blocks
            else:
                # get all inputs if layer_id=0, else mask all
                inp_mask = F.ones(shape=(self._feature_max_size), dtype=F.int32) if layer_id == 0 else \
                           F.zeros(shape=(self._feature_max_size), dtype=F.int32)

            # hidden layer masking for with_skip_connection
            if self.with_skip_connection:
                if layer_id > 0:
                    layer_mask = arc_seq[
                                 start_idx: start_idx + layer_id]
                    layer_mask = F.boolean_mask(self._skip_conn_map[layer_id], layer_mask)
                    new_range2 = F.range(0, limit=layer_id * self._weight_max_units, delta=1, dtype=F.int32)
                    layer_mask = F.map_fn(
                        lambda t: F.cast(F.logical_and(t[0] <= new_range2, new_range2 < t[1]), dtype=F.int32),
                        layer_mask)
                    layer_mask = F.reduce_sum(layer_mask, axis=0)
                    start_idx += layer_id * self.with_skip_connection
                else:
                    layer_mask = []
                row_mask = F.concat([inp_mask, layer_mask], axis=0)
            else:
                if layer_id > 0:
                    # keep last/closest layer, mask all others
                    layer_masks = [F.zeros(shape=(self._weight_max_units*(layer_id-1)), dtype=F.int32), F.ones(shape=(self._weight_max_units), dtype=F.int32)]
                else:
                    layer_masks = []

                row_mask = F.concat([inp_mask] + layer_masks, axis=0)
            w_mask = F.matmul(F.expand_dims(row_mask, -1), F.expand_dims(col_mask, 0))

            # get the backend layer
            w = F.where(F.cast(w_mask, F.bool), x=w, y=F.fill(F.shape(w), 0.))
            b = F.where(F.cast(col_mask, F.bool), x=b, y=F.fill(F.shape(b), 0.))
            x, drop_rate = self._layer(w, b, inputs, layer_id)
            dropout_placeholders.append(drop_rate)
            layer_outputs.append(x)
            inputs = F.concat([inputs, x], axis=1)
            w_masks.append((w, b))

        if self.with_output_blocks:
            model_output = []
            output_arcs = arc_seq[start_idx::]
            if type(output_arcs) is list:
                output_arcs_len = len(output_arcs)
            else:  # is tensor
                output_arcs_len = int(output_arcs.shape[0])
            assert output_arcs_len == self.num_output_blocks * self.num_layers, "model builder was specified to build output" \
                                                                                " connections, but the input architecture did" \
                                                                                "n't match output info; expected arc of length" \
                                                                                "=%i, received %i" % (
                                                                                start_idx + self.num_output_blocks * self.num_layers,
                                                                                len(arc_seq) if type(
                                                                                    arc_seq) is list else arc_seq.shape[
                                                                                    0].value)
            layer_outputs_ = F.concat(layer_outputs, axis=1)  # shape: num_samples, max_units x num_layers
            for i in range(self.num_output_blocks):
                # output_mask is a row_mask
                output_mask = F.boolean_mask(self._output_block_map,
                                              output_arcs[i * self.num_layers: (i + 1) * self.num_layers])
                new_range = F.range(0, limit=self.num_layers * self._weight_max_units, delta=1, dtype=F.int32)
                output_mask = F.map_fn(
                    lambda t: F.cast(F.logical_and(t[0] <= new_range, new_range < t[1]), dtype=F.int32),
                    output_mask)
                output_mask = F.reduce_sum(output_mask, axis=0)
                output_mask = F.matmul(F.expand_dims(output_mask, -1),
                                        F.ones((1, self._child_output_size[i]), dtype=F.int32))
                w = F.where(F.cast(output_mask, F.bool), x=self.w_out[i], y=F.fill(F.shape(self.w_out[i]), 0.))
                model_output.append(
                    F.get_layer(
                        x=F.matmul(layer_outputs_, w) + self.b_out[i], 
                        op=Operation("Activation", activation=self._child_output_func[i])
                    ))
                w_masks.append((w, self.b_out[i]))
        else:
            model_output = [
                F.get_layer(
                    x=F.matmul(x, self.w_out[i]) + self.b_out[i],
                    op=Operation("Activation", activation=self._child_output_func[i]))
                            for i in range(len(self.output_node))]
        return model_output, w_masks, layer_outputs, dropout_placeholders

    def _layer(self, w, b, inputs, layer_id, use_dropout=True):
        layer = F.get_layer(
            x=F.matmul(inputs, w) + b,
            op=Operation("Activation", activation=self._actv_fn))
        layer = F.get_layer(x=layer, op=Operation('Dropout', rate=self.drop_probs[layer_id], training=use_dropout))
        return layer, None

    def _model(self, arc):
        if self.feature_model is None:
            child_model_input = self.child_model_input
        else:
            if self.feature_model.pseudo_inputs_pipe is None or arc is not None:
                child_model_input = self.feature_model.x_inputs
            else:
                child_model_input = self.child_model_input_pipe
        if arc is None:
            model = EnasAnnModel(inputs=child_model_input, outputs=self.sample_model_output,
                                 arc_seq=arc,
                                 dag=self,
                                 session=self.session)
        else:
            model = EnasAnnModel(inputs=child_model_input, outputs=self.fixed_model_output,
                                 arc_seq=arc,
                                 dag=self,
                                 session=self.session)
        return model

    def _create_input_ph(self):
        ops_each_layer = 1
        total_arc_len = sum([ops_each_layer + self._input_block_map.shape[0] * self.with_input_blocks] + [
            ops_each_layer + self._input_block_map.shape[0] * self.with_input_blocks + i * self.with_skip_connection
            for i in range(1, self.num_layers)])
        if self.with_output_blocks:
            total_arc_len += self.num_output_blocks * self.num_layers
        self.input_ph_ = [F.placeholder(shape=(), dtype=F.int32, name='arc_{}'.format(i))
                          for i in range(total_arc_len)]
        self.input_arc = self.input_ph_
        return

    def _compile(self, w_masks, model_output, use_pipe=True, var_scope=None):
        """
        Compile loss and train_op here so all child models will share the same, instead of creating new ones every time
        """
        loss = self.model_compile_dict['loss']
        optimizer = self.model_compile_dict['optimizer']
        metrics = self.model_compile_dict['metrics'] if 'metrics' in self.model_compile_dict else None
        var_scope = var_scope or self.name

        with F.variable_scope(var_scope):
            if self.feature_model is None or self.feature_model.pseudo_inputs_pipe is None:
                labels = self.child_model_label
            else:
                # TODO: for now, only use data_pipe for sample_arc
                if use_pipe:
                    labels = self.child_model_label_pipe
                else:
                    labels = self.child_model_label

            # TODO: process loss_weights
            loss_weights = self.model_compile_dict['loss_weights'] if 'loss_weights' in self.model_compile_dict else None
            if type(loss) is str:
                loss_ = [F.get_loss(loss, labels[i], model_output[i]) for i in range(len(model_output))]
                # loss_ should originally all have the batch_size dim, then reduce_mean to 1-unit sample
                # if the loss is homogeneous, then take the mean
                loss_ = F.reduce_mean(loss_)
            elif type(loss) is list:
                loss_ = []
                for i in range(len(loss)):
                    loss_.append(F.get_loss(
                        loss[i],
                        labels[i],
                        model_output[i]
                    ))
                # if the loss are provided by a list, assume its heterogeneous, and return the sum of
                # each individual loss components
                loss_ = F.reduce_sum(loss_)
            elif callable(loss):
                loss_ = F.reduce_sum([loss(labels[i], model_output[i]) for i in range(len(model_output))])
            else:
                raise Exception("expect loss to be str, list, dict or callable; got %s" % loss)
            trainable_var = [var for var in F.trainable_variables(scope=var_scope)]
            if self.feature_model_trainable:
                feature_model_trainable_var = [var for var in F.trainable_variables(scope=self.feature_model.name)]
                assert len(feature_model_trainable_var) > 0, "You asked me to train featureModel but there is no trainable " \
                                                             "variables in featureModel"
                trainable_var += feature_model_trainable_var
            regularization_penalty = 0.
            if self.l1_reg > 0:
                l1_regularization_penalty = self.l1_reg * F.reduce_mean([F.reduce_mean(F.abs(var[0])) for var in w_masks])
                loss_ += l1_regularization_penalty
            else:
                l1_regularization_penalty = 0.

            if self.l2_reg > 0:
                l2_regularization_penalty = self.l2_reg * F.reduce_mean([F.reduce_mean(F.pow(var[0], 2)) for var in w_masks ])
                loss_ += l2_regularization_penalty
            else:
                l2_regularization_penalty = 0.

            regularization_penalty += l1_regularization_penalty + l2_regularization_penalty

            # default settings used from enas
            if self.child_train_op_kwargs is None:
                # more sensible default values
                train_op, lr, optimizer_ = F.get_train_op(
                    loss=loss_,
                    variables=trainable_var,
                    optimizer=optimizer)
            # user specific settings; useful when training the final searched arc
            else:
                train_op, lr, optimizer_ = F.get_train_op(
                    loss=loss_,
                    variables=trainable_var,
                    optimizer=optimizer,
                    **self.child_train_op_kwargs)
            if metrics is None:
                metrics = []
            else:
                metrics = [F.get_metric(x) for x in metrics]
            # TODO: this needs fixing to be more generic;
            # TODO: for example, the squeeze op is not usable for
            # TODO: other metrics such as Acc
            metrics_ = [f(F.squeeze(self.child_model_label[i]), F.squeeze(model_output[i]))
                        for i in range(len(model_output)) for f in metrics]
            ops = {'train_op': train_op,
                   'lr': lr,
                   'optimizer': optimizer,
                   'loss': loss_,
                   'metrics': metrics_,
                   'reg_cost': regularization_penalty
                   }
            return ops

    def connect_controller(self, controller):
        self.sample_arc = controller.sample_arc
        self.controller = controller
        return


class EnasConv1dDAG:
    def __init__(self,
                 model_space,
                 input_node,
                 output_node,
                 model_compile_dict,
                 session,
                 with_skip_connection=True,
                 batch_size=128,
                 keep_prob=0.9,
                 l1_reg=0.0,
                 l2_reg=0.0,
                 reduction_factor=4,
                 controller=None,
                 child_train_op_kwargs=None,
                 stem_config=None,
                 data_format='NWC',
                 train_fixed_arc=False,
                 fixed_arc=None,
                 name='EnasDAG',
                 **kwargs):
        """EnasCnnDAG is a DAG model builder for using the weight sharing framework.

        This class deals with the Convolutional neural network.

        Parameters
        ----------
        model_space: amber.architect.ModelSpace
            model space to search network architectures from
        input_node: amber.architect.Operation, or list
            defines the input shapes and activations
        output_node: amber.architect.Operation, or list
            defines the output shapes and activations
        model_compile_dict: dict
            compile dict for child models
        session: amber.backend.Session
            session for building enas DAG
        train_fixed_arc: bool
            boolean indicator for whether is the final stage; if is True, must provide `fixed_arc` and not connect
            to a controller
        fixed_arc: list-like
            the architecture for final stage training
        name: str
            a name identifier for this instance
        """
        assert type(input_node) in (Operation, F.TensorType) or len(
            input_node) == 1, "EnasCnnDAG currently does not accept List type of inputs"
        assert type(output_node) in (Operation, F.TensorType) or len(
            output_node) == 1, "EnasCnnDAG currently does not accept List type of outputs"
        self.input_node = input_node
        self.output_node = output_node
        self.num_layers = len(model_space)
        self.model_space = model_space
        self.model_compile_dict = model_compile_dict
        self.session = session
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.with_skip_connection = with_skip_connection
        self.controller = None
        if controller is not None:
            self.set_controller(controller)
        self.child_train_op_kwargs = child_train_op_kwargs
        self.stem_config = stem_config or {}

        self.name = name
        self.batch_size = batch_size
        self.batch_dim = None
        self.reduction_factor = reduction_factor
        self.keep_prob = keep_prob
        self.data_format = data_format
        self.out_filters = None
        self.branches = []
        self.is_initialized = False

        self.add_conv1_under_pool = kwargs.get("add_conv1_under_pool", True)

        self.train_fixed_arc = train_fixed_arc
        self.fixed_arc = fixed_arc
        if self.train_fixed_arc:
            assert self.fixed_arc is not None, "if train_fixed_arc=True, must provide the architectures in `fixed_arc`"
            assert controller is None, "if train_fixed_arc=True, must not provide controller"
            self.skip_max_depth = None

        self._verify_args()
        self.vars = []
        if controller is None:
            self.controller = None
            print("this EnasDAG instance did not connect a controller; pleaes make sure you are only training a fixed "
                  "architecture.")
        else:
            self.controller = controller
            self._build_sample_arc()
        self._build_fixed_arc()
        F.init_all_params(sess=self.session)

    def _verify_args(self):
        out_filters = []
        pool_layers = []
        for layer_id in range(len(self.model_space)):
            layer = self.model_space[layer_id]
            this_out_filters = [l.Layer_attributes['filters'] for l in layer]
            assert len(
                set(this_out_filters)) == 1, "EnasConv1dDAG only supports one identical number of filters per layer," \
                                             "but found %i different number of filters in layer %s" % \
                                             (len(set(this_out_filters)), layer)
            if len(out_filters) and this_out_filters[0] != out_filters[-1]:
                pool_layers.append(layer_id - 1)

            out_filters.append(this_out_filters[0])
        self.out_filters = out_filters
        self.pool_layers = pool_layers

        # if train fixed arc, avoid building unused skip connections
        # and verify the input fixed_arc
        if self.train_fixed_arc:
            assert self.fixed_arc is not None
            skip_max_depth = {}
            start_idx = 0
            for layer_id in range(len(self.model_space)):
                skip_max_depth[layer_id] = layer_id
                operation = self.fixed_arc[start_idx]
                total_choices = len(self.model_space[layer_id])
                assert 0 <= operation < total_choices, "Invalid operation selection: layer_id=%i, " \
                                                       "operation=%i, model space len=%i" % (
                                                       layer_id, operation, total_choices)
                if layer_id > 0:
                    skip_binary = self.fixed_arc[(start_idx + 1):(start_idx + 1 + layer_id)]
                    skip = [i for i in range(layer_id) if skip_binary[i] == 1]
                    for d in skip:
                        skip_max_depth[d] = layer_id

                start_idx += 1 + layer_id
            print('-' * 80)
            print(skip_max_depth)
            self.skip_max_depth = skip_max_depth

        if type(self.input_node) is list:
            self.input_node = self.input_node[0]
        self.input_ph = F.placeholder(shape=[self.batch_dim] + list(self.input_node.Layer_attributes['shape']),
                                       name='child_input_placeholder',
                                       dtype=F.float32)
        if type(self.output_node) is list:
            self.output_node = self.output_node[0]
        self.label_ph = F.placeholder(shape=(self.batch_dim, self.output_node.Layer_attributes['units']),
                                       dtype=F.float32,
                                       name='child_output_placeholder')

    def __call__(self, arc_seq=None, **kwargs):
        return self._model(arc_seq, **kwargs)

    def _model(self, arc, **kwargs):
        if self.train_fixed_arc:
            assert arc == self.fixed_arc or arc is None, "This DAG instance is built to train fixed arc, hence you " \
                                                         "can only provide arc=None or arc=self.fixed_arc; check the " \
                                                         "initialization of this instances "
        if arc is None:
            if self.train_fixed_arc:
                model = EnasCnnModel(inputs=self.fixed_model_input,
                                     outputs=self.fixed_model_output,
                                     labels=self.fixed_model_label,
                                     arc_seq=arc,
                                     dag=self,
                                     session=self.session,
                                     name=self.name)

            else:
                model = EnasCnnModel(inputs=self.sample_model_input,
                                     outputs=self.sample_model_output,
                                     labels=self.sample_model_label,
                                     arc_seq=arc,
                                     dag=self,
                                     session=self.session,
                                     name=self.name)
        else:
            model = EnasCnnModel(inputs=self.fixed_model_input,
                                 outputs=self.fixed_model_output,
                                 labels=self.fixed_model_label,
                                 arc_seq=arc,
                                 dag=self,
                                 session=self.session,
                                 name=self.name)
        return model

    def set_controller(self, controller):
        assert self.controller is None, "already has inherent controller, disallowed; start a new " \
                                        "EnasCnnDAG instance if you want to connect another controller"
        self.controller = controller
        self.sample_arc = controller.sample_arc

    def _build_sample_arc(self, input_tensor=None, label_tensor=None, **kwargs):
        """
        Notes:
            I left `input_tensor` and `label_tensor` so that in the future some pipeline
            tensors can be connected to the model, instead of the placeholders as is now.

        Args:
            input_tensor:
            label_tensor:
            **kwargs:

        Returns:

        """
        var_scope = self.name
        is_training = kwargs.pop('is_training', True)
        with F.variable_scope(var_scope):
            input_tensor = self.input_ph if input_tensor is None else input_tensor
            label_tensor = self.label_ph if label_tensor is None else label_tensor
            model_out, dropout_placeholders = self._build_dag(arc_seq=self.sample_arc, input_tensor=input_tensor,
                                                              is_training=is_training,)
                                                              #reuse=reuse)
            self.sample_model_output = model_out
            self.sample_model_input = input_tensor
            self.sample_model_label = label_tensor
            self.sample_dropouts = dropout_placeholders
            ops = self._compile(model_output=[self.sample_model_output], labels=[label_tensor],
                                is_training=is_training,
                                var_scope=var_scope)
            self.sample_train_op = ops['train_op']
            self.sample_loss = ops['loss']
            self.sample_optimizer = ops['optimizer']
            self.sample_metrics = ops['metrics']
            self.sample_ops = ops
        # if not self.is_initialized:
        F.init_all_params(sess=self.session, var_scope=var_scope)

    def _build_fixed_arc(self, input_tensor=None, label_tensor=None, **kwargs):
        """
        Notes:
            I left `input_tensor` and `label_tensor` so that in the future some pipeline
            tensors can be connected to the model, instead of the placeholders as is now.

        Args:
            input_tensor:
            label_tensor:
            **kwargs:

        Returns:

        """
        var_scope = self.name
        if self.train_fixed_arc:
            is_training = True
        else:
            is_training = kwargs.get('is_training', False)
        with F.variable_scope(var_scope):
            input_tensor = self.input_ph if input_tensor is None else input_tensor
            label_tensor = self.label_ph if label_tensor is None else label_tensor
            self._create_input_ph()
            if self.train_fixed_arc:
                model_out, dropout_placeholders = self._build_dag(arc_seq=self.fixed_arc, input_tensor=input_tensor,
                                                                  is_training=is_training,)
                                                                  #reuse=reuse)
            else:
                model_out, dropout_placeholders = self._build_dag(arc_seq=self.input_arc, input_tensor=input_tensor,
                                                                  is_training=is_training,)
                                                                  #reuse=reuse)
            self.fixed_model_output = model_out
            self.fixed_model_input = input_tensor
            self.fixed_model_label = label_tensor
            self.fixed_dropouts = dropout_placeholders
            ops = self._compile(model_output=[self.fixed_model_output], labels=[label_tensor],
                                is_training=is_training,
                                var_scope=var_scope)
            self.fixed_train_op = ops['train_op']
            self.fixed_loss = ops['loss']
            self.fixed_optimizer = ops['optimizer']
            self.fixed_metrics = ops['metrics']
            self.fixed_ops = ops
        # if not self.is_initialized:
        F.init_all_params(sess=self.session, var_scope=var_scope)

    def _create_input_ph(self):
        ops_each_layer = 1
        total_arc_len = sum([ops_each_layer + i
                             for i in range(self.num_layers)])
        input_ph_ = [F.placeholder(shape=(), dtype=F.int32, name='arc_{}'.format(i))
                     for i in range(total_arc_len)]
        self.input_arc = input_ph_
        return

    def _compile(self, model_output, labels=None, is_training=True, var_scope=None):
        loss = self.model_compile_dict['loss']
        optimizer = self.model_compile_dict['optimizer']
        metrics = self.model_compile_dict.pop('metrics', None)
        var_scope = var_scope or self.name
        labels = self.label_ph if labels is None else labels
        with F.variable_scope('compile'):
            if type(loss) is str:
                loss_ = [F.get_loss(loss, labels[i], model_output[i]) for i in range(len(model_output))]
                # loss_ should originally all have the batch_size dim, then reduce_mean to 1-unit sample
                # if the loss is homogeneous, then take the mean
                loss_ = F.reduce_mean(loss_)
            elif type(loss) is list:
                loss_ = []
                for i in range(len(loss)):
                    loss_.append(F.get_loss(
                        loss[i],
                        labels[i],
                        model_output[i]
                    ))
                # if the loss are provided by a list, assume its heterogeneous, and return the sum of
                # each individual loss components
                loss_ = F.reduce_sum(loss_)
            elif callable(loss):
                loss_ = F.reduce_sum([loss(labels[i], model_output[i]) for i in range(len(model_output))])
            else:
                raise Exception("expect loss to be str, list, dict or callable; got %s" % loss)
            trainable_var = [var for var in F.trainable_variables(scope=var_scope)]
            regularization_penalty = 0.
            if self.l1_reg > 0:
                l1_regularization_penalty = self.l1_reg * F.reduce_mean([F.reduce_mean(F.abs(var[0])) for var in trainable_var if
                                                                                    F.get_param_name(var).split('/')[-1] == 'w:0'])
                loss_ += l1_regularization_penalty
            else:
                l1_regularization_penalty = 0.

            if self.l2_reg > 0:
                l2_regularization_penalty = self.l2_reg * F.reduce_mean([F.reduce_mean(F.pow(var[0], 2)) for var in trainable_var if F.get_param_name(var).split('/')[-1] == 'w:0'])
                loss_ += l2_regularization_penalty
            else:
                l2_regularization_penalty = 0.

            regularization_penalty += l1_regularization_penalty + l2_regularization_penalty

            if is_training:
                # default settings used from enas
                if self.child_train_op_kwargs is None:
                    # more sensible default values
                    train_op, lr, optimizer_ = F.get_train_op( 
                        loss=loss_,
                        variables=trainable_var,
                        optimizer=optimizer,
                        train_step=self.train_step)
                # user specific settings; useful when training the final searched arc
                else:
                    train_op, lr, optimizer_ = F.get_train_op(  # get_train_ops(
                        loss=loss_,
                        variables=trainable_var,
                        optimizer=optimizer,
                        train_step=self.train_step,
                        **self.child_train_op_kwargs)
            else:
                train_op, lr, optimizer_ = None, None, None
            if metrics is None:
                metrics = []
            else:
                metrics = [F.get_metric(x) for x in metrics]
            # TODO: this needs fixing to be more generic;
            # TODO: for example, the squeeze op is not usable for
            # TODO: other metrics such as Acc
            metrics_ = [f(labels[i], model_output[i])
                        for i in range(len(model_output)) for f in metrics]
            ops = {'train_op': train_op,
                   'lr': lr,
                   'optimizer': optimizer_,
                   'loss': loss_,
                   'metrics': metrics_,
                   'reg_cost': regularization_penalty
                   }
        return ops

    def _build_dag(self, arc_seq, input_tensor=None, is_training=True):
        self.layers = []
        self.train_step = F.Variable(0, shape=(), dtype=F.int32, trainable=False, name="train_step")
        dropout_placeholders = []
        with F.variable_scope('child_model'):
            out_filters = self.out_filters
            # input = self.input_node
            input = self.input_ph if input_tensor is None else input_tensor
            layers = []
            has_stem_conv = self.stem_config.get('has_stem_conv', True)
            if has_stem_conv:
                with F.variable_scope("stem_conv"):
                    stem_kernel_size = self.stem_config.get('stem_kernel_size', 8)
                    stem_filters = out_filters[0]
                    x = F.get_layer(
                        x=input, 
                        op=F.Operation('Conv1D', filters=stem_filters, kernel_size=stem_kernel_size, strides=1, padding="SAME"))
                    x = F.get_layer(x=x, op=F.Operation('BatchNormalization', training=is_training))
                    layers.append(x)
                    self.layers.append(x)
            else:
                layers = [input]

            start_idx = 0
            for layer_id in range(self.num_layers):
                with F.variable_scope("layer_{0}".format(layer_id)):
                    x = self._layer(arc_seq, layer_id, layers, start_idx, out_filters[layer_id], is_training)
                    x = F.get_layer(x=x, op=Operation('Dropout', rate=1-self.keep_prob, training=is_training))
                    if layer_id == 0:
                        layers = [x]
                    else:
                        layers.append(x)
                    self.layers.append(x)
                    if (self.with_skip_connection is True) and (layer_id in self.pool_layers):
                        with F.variable_scope("pool_at_{0}".format(layer_id)):
                            pooled_layers = []
                            for i, layer in enumerate(layers):
                                if self.train_fixed_arc and self.skip_max_depth[i] < layer_id:
                                    print("Not building pool_at_%i/from_%i because its max-depth=%i" % (
                                    layer_id, i, self.skip_max_depth[i]))
                                    x = layer
                                else:
                                    with F.variable_scope("from_{0}".format(i)):
                                        x = self._refactorized_channels_for_skipcon(
                                            layer, out_filters[layer_id + 1], is_training)
                                pooled_layers.append(x)
                            layers = pooled_layers
                start_idx += 1 + layer_id*int(self.with_skip_connection)

            flatten_op = self.stem_config.get('flatten_op', 'flatten')
            if flatten_op == 'global_avg_pool' or flatten_op == 'gap':
                keras_data_format = 'channels_last' if self.data_format.endswith('C') else "channels_first"
                x = F.get_layer(x=x, op=Operation('GlobalAveragePooling1D', data_format=keras_data_format))
            elif flatten_op == 'flatten':
                keras_data_format = 'channels_last' if self.data_format.endswith('C') else "channels_first"
                x = F.get_layer(x=x, op=Operation('flatten', data_format=keras_data_format))
            else:
                raise Exception("cannot understand flatten_op: %s" % flatten_op)
            self.layers.append(x)
            x = F.get_layer(x=x, op=Operation('Dropout', rate=1-self.keep_prob, training=is_training))
            with F.variable_scope("fc"):
                fc_units = self.stem_config.get('fc_units', 1000)
                x = F.get_layer(x=x, op=F.Operation('Dense', units=fc_units, activation='relu'))
                x = F.get_layer(x=x, op=F.Operation('Dropout', rate=1-self.keep_prob, training=is_training))
                w_out = F.create_parameter("w_out", [fc_units, self.output_node.Layer_attributes['units']])
                b_out = F.create_parameter("b_out", shape=[self.output_node.Layer_attributes['units']], initializer='zeros')
                model_output = F.get_layer(
                    x=F.matmul(x, w_out) + b_out,
                    op=Operation("Activation", activation=self.output_node.Layer_attributes['activation'])
                )
        return model_output, dropout_placeholders

    def _refactorized_channels_for_skipcon(self, layer, out_filters, is_training):
        """for dealing with mismatch-dimensions in skip connections: use a linear transformation"""
        channel_num = self.data_format.index('C')
        #inp_c = F.get_shape(layer)[channel_num]
        actual_data_format = 'channels_first' if channel_num == 1 else 'channels_last'

        with F.variable_scope("path1_conv"):
            x = F.get_layer(x=x, op=Operation('Conv1D', filters=out_filters, kernel_size=1, strides=1, padding="same"))
            x = F.get_layer(x=x, op=Operation('MaxPooling1D', pool_size=self.reduction_factor, strides=self.reduction_factor, padding="same", data_format=actual_data_format))
        return x

    def _layer(self, arc_seq, layer_id, prev_layers, start_idx, out_filters, is_training):
        inputs = prev_layers[-1]
        if self.data_format == "NWC":
            inp_w = F.get_shape(inputs)[1]
            inp_c = F.get_shape(inputs)[2]

        elif self.data_format == "NCW":
            inp_c = F.get_shape(inputs)[1]
            inp_w = F.get_shape(inputs)[2]
        else:
            raise Exception("cannot understand data format: %s" % self.data_format)
        count = arc_seq[start_idx]
        branches = {}
        strides = []
        for i in range(len(self.model_space[layer_id])):
            if self.train_fixed_arc and i != count:
                continue

            with F.variable_scope("branch_%i" % i):
                if self.model_space[layer_id][i].Layer_type == 'conv1d':
                    # print('%i, conv1d'%layer_id)
                    y = self._conv_branch(inputs, layer_attr=self.model_space[layer_id][i].Layer_attributes,
                                          is_training=is_training)
                    branches[F.equal(count, i)] = y
                elif self.model_space[layer_id][i].Layer_type == 'maxpool1d':
                    # print('%i, maxpool1d' % layer_id)
                    y = self._pool_branch(inputs, "max",
                                          layer_attr=self.model_space[layer_id][i].Layer_attributes,
                                          is_training=is_training)
                    branches[F.equal(count, i)] = y
                    strides.append(self.model_space[layer_id][i].Layer_attributes['strides'])
                elif self.model_space[layer_id][i].Layer_type == 'avgpool1d':
                    # print('%i, avgpool1d' % layer_id)
                    y = self._pool_branch(inputs, "avg",
                                          layer_attr=self.model_space[layer_id][i].Layer_attributes,
                                          is_training=is_training)
                    branches[F.equal(count, i)] = y
                    strides.append(self.model_space[layer_id][i].Layer_attributes['strides'])
                elif self.model_space[layer_id][i].Layer_type == 'identity':
                    y = self._identity_branch(inputs)
                    branches[F.equal(count, i)] = y
                else:
                    raise Exception("Unknown layer: %s" % self.model_space[layer_id][i])

        self.branches.append(branches)
        if len(strides) > 0:
            assert len(set(strides)) == 1, "If you set strides!=1 (i.e. a reduction layer), then all candidate operations must have the same strides to keep the shape identical; got %s" % strides
            inp_w = int(np.ceil(inp_w / strides[0]))
        if self.train_fixed_arc:
            ks = list(branches.keys())
            assert len(ks) == 1
            out = branches[ks[0]]()
        else:
            out = F.case(
                branches,
                default=lambda: F.zeros(shape=[self.batch_size, inp_w, out_filters], dtype=F.float32),
                exclusive=True)
        if self.data_format == "NWC":
            out.set_shape([None, inp_w, out_filters])
        elif self.data_format == "NCW":
            out.set_shape([None, out_filters, inp_w])
        if self.with_skip_connection is True and layer_id > 0:
            skip_start = start_idx + 1
            skip = arc_seq[skip_start: skip_start + layer_id]
            with F.variable_scope("skip"):
                # might be the cause of oom.. zz 2020.1.6
                res_layers = []
                for i in range(layer_id):
                    if self.train_fixed_arc:
                        res_layers = [prev_layers[i] for i in range(len(skip)) if skip[i] == 1]
                    else:
                        res_layers.append(F.cond(F.equal(skip[i], 1),
                                                  lambda: prev_layers[i],
                                                  lambda: F.stop_gradient(F.zeros_like(prev_layers[i]))
                                                  )
                                        )
                res_layers.append(out)
                out = F.reduce_sum(res_layers, axis=0)
                out = F.get_layer(x=out, op=Operation('BatchNormalization', training=is_training))
        return out

    def _conv_branch(self, inputs, layer_attr, is_training):
        kernel_size = layer_attr['kernel_size']
        activation_fn = layer_attr['activation']
        dilation = layer_attr['dilation'] if 'dilation' in layer_attr else 1
        filters = layer_attr['filters']
        x = F.get_layer(
            x=inputs, 
            op=F.Operation('Conv1D', filters=filters, kernel_size=kernel_size, strides=1, padding="SAME", dilation_rate=dilation))
        x = F.get_layer(x=x, op=F.Operation('BatchNormalization', training=is_training))
        b = F.create_parameter("b", shape=[1], initializer='zeros')
        x = F.get_layer(
            x=x+b,
            op=Operation("Activation", activation=activation_fn))
        return lambda: x

    def _pool_branch(self, inputs, avg_or_max, layer_attr, is_training):
        pool_size = layer_attr['pool_size']
        strides = layer_attr['strides']
        filters = layer_attr['filters']
        if self.data_format == "NWC":
            actual_data_format = "channels_last"
        elif self.data_format == "NCW":
            actual_data_format = "channels_first"
        else:
            raise Exception("Unknown data format: %s" % self.data_format)

        if self.add_conv1_under_pool:
            with F.variable_scope("conv_1"):
                x = F.get_layer(
                    x=inputs, 
                    op=F.Operation('Conv1D', filters=filters, kernel_size=1, strides=1, padding="same"))
                x = F.get_layer(x=x, op=F.Operation('BatchNormalization', training=is_training))
                x = F.get_layer(x=x, op=F.Operation('Activation', activation="relu"))
        else:
            x = inputs
        with F.variable_scope("pool"):
            if avg_or_max == "avg":
                x = F.get_layer(
                    x=x, 
                    op=F.Operation("AveragePooling1D", pool_size=pool_size, strides=strides, padding="same",data_format=actual_data_format))
            elif avg_or_max == "max":
                x = F.get_layer(
                    x=x, 
                    op=F.Operation("MaxPooling1D", pool_size=pool_size, strides=strides, padding="same",data_format=actual_data_format))
            else:
                raise ValueError("Unknown pool {}".format(avg_or_max))
        return lambda: x

    def _identity_branch(self, inputs):
        return lambda: inputs


# Initial Date: 2020.5.17
class EnasConv1DwDataDescrption(EnasConv1dDAG):
    """This is a modeler that specifiied for convolution network with data description features
    """
    def __init__(self, data_description, *args, **kwargs):
        self.data_description = data_description
        super().__init__(*args, **kwargs)
        if len(self.data_description.shape) < 2:
            self.data_description = np.expand_dims(self.data_description, axis=0)

    def _model(self, arc, **kwargs):
        """
        Overwrite the parent `_model` method to feed the description to controller when sampling architectures
        :param arc:
        :param kwargs:
        :return:
        """
        if self.train_fixed_arc:
            assert arc == self.fixed_arc or arc is None, "This DAG instance is built to train fixed arc, hence you " \
                                                         "can only provide arc=None or arc=self.fixed_arc; check the " \
                                                         "initialization of this instances "
        if arc is None:
            if self.train_fixed_arc:
                model = EnasCnnModel(inputs=self.fixed_model_input,
                                     outputs=self.fixed_model_output,
                                     labels=self.fixed_model_label,
                                     arc_seq=arc,
                                     dag=self,
                                     session=self.session,
                                     name=self.name)

            else:
                assert len(self.data_description) == 1, \
                    ValueError(f"data_descriptor to EnasCnnModel must be a single row vector, shape {self.data_description.shape}")
                model = EnasCnnModel(inputs=self.sample_model_input,
                                     outputs=self.sample_model_output,
                                     labels=self.sample_model_label,
                                     arc_seq=arc,
                                     dag=self,
                                     session=self.session,
                                     name=self.name,
                                     sample_dag_feed_dict={
                                         self.controller.data_descriptive_feature: self.data_description}
                                     )
        else:
            model = EnasCnnModel(inputs=self.fixed_model_input,
                                 outputs=self.fixed_model_output,
                                 labels=self.fixed_model_label,
                                 arc_seq=arc,
                                 dag=self,
                                 session=self.session,
                                 name=self.name)
        return model

