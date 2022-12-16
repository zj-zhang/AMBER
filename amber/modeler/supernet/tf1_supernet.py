"""supernet built using static backend libraries
that is, only TF1.X at this point
"""

import numpy as np
import warnings
import h5py
import datetime
from amber.architect.commonOps import batchify
from tqdm import trange, tqdm
from .base import BaseEnasConv1dDAG
from ... import backend as F
from ...backend import Operation


class EnasAnnModelBuilder:
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
                 inputs_op,
                 output_op,
                 model_compile_dict,
                 session,
                 drop_probs=0.1,
                 l1_reg=0.0,
                 l2_reg=0.0,
                 with_skip_connection=True,
                 with_input_blocks=False,
                 with_output_blocks=False,
                 controller=None,
                 feature_model=None,
                 feature_model_trainable=None,
                 child_train_op_kwargs=None,
                 name='EnasDAG',
                 **kwargs
                 ):

        #assert with_skip_connection == with_input_blocks == True, \
        #    "EnasAnnDAG must have with_input_blocks and with_skip_connection"
        input_node = inputs_op
        output_node = output_op
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
            self.session = F.get_session()
        else:
            self.session = session
        self.model_compile_dict = model_compile_dict
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.with_skip_connection = with_skip_connection
        self.with_input_blocks = with_input_blocks
        if self.with_input_blocks is True:
            assert len(self.input_node) > 1, "Must have more than one input if with_input_blocks=True"
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

    def __call__(self, arc_seq=None, *args, **kwargs):
        model = self._model(arc_seq)
        model.compile(**self.model_compile_dict)
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
        self._child_output_func = [n.Layer_attributes.get('activation', 'linear') for n in self.output_node]
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
            inputs = self.feature_model.pseudo_inputs
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
                # if layer_id > 0, keep last/closest layer, mask all others
                layer_masks = [F.zeros(shape=(self._weight_max_units*(layer_id-1)), dtype=F.int32), F.ones(shape=(self._weight_max_units), dtype=F.int32)] if layer_id>0 else []
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
                    F.get_math_func(self._child_output_func[i])(
                        F.matmul(layer_outputs_, w) + self.b_out[i])
                    )
                w_masks.append((w, self.b_out[i]))
        # no output block
        else:
            model_output = [
                F.get_math_func(self._child_output_func[i])(F.matmul(x, self.w_out[i]) + self.b_out[i])
                for i in range(len(self.output_node))
                ]
        return model_output, w_masks, layer_outputs, dropout_placeholders

    def _layer(self, w, b, inputs, layer_id, use_dropout=True):
        layer = F.get_math_func(self._actv_fn)(F.matmul(inputs, w) + b)
        if use_dropout:
            layer = F.get_math_func('dropout')(layer, rate=self.drop_probs[layer_id])
        return layer, None

    def _model(self, arc):
        if self.feature_model is None:
            child_model_input = self.child_model_input
        else:
            child_model_input = self.feature_model.x_inputs
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
                feature_model_trainable_var = self.feature_model.trainable_var
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
            #if self.child_train_op_kwargs is None:
            assert self.child_train_op_kwargs is None, "legacy child_train_op_kwargs to tf1_supernet.EnasAnnModelBuilder is no longer supported"
            train_op, lr, optimizer_ = F.get_train_op(
                loss=loss_,
                variables=trainable_var,
                optimizer=optimizer)
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


class EnasCnnModelBuilder(BaseEnasConv1dDAG):
    def __init__(self,
                 model_space,
                 inputs_op,
                 output_op,
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
        super().__init__(model_space=model_space, inputs_op=inputs_op, output_op=output_op, model_compile_dict=model_compile_dict,
            session=session, train_fixed_arc=train_fixed_arc, fixed_arc=fixed_arc, name=name,
            with_skip_connection=with_skip_connection, batch_size=batch_size, keep_prob=keep_prob,
            l1_reg=l1_reg, l2_reg=l2_reg,
            reduction_factor=reduction_factor,
            controller=controller,
            stem_config=stem_config,
            data_format=data_format,
            **kwargs
        )
        F.init_all_params(sess=self.session)

    def _verify_args(self):
        super()._verify_args()
        self.input_ph = F.placeholder(shape=[self.batch_dim] + list(self.input_node.Layer_attributes['shape']),
                                       name='child_input_placeholder',
                                       dtype=F.float32)
        if type(self.output_node) is list:
            self.output_node = self.output_node[0]
        self.label_ph = F.placeholder(shape=(self.batch_dim, self.output_node.Layer_attributes['units']),
                                       dtype=F.float32,
                                       name='child_output_placeholder')

    def __call__(self, arc_seq=None, **kwargs):
        model = self._model(arc_seq, **kwargs)
        model.compile(**self.model_compile_dict)
        return model
    
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

    def _build_sample_arc(self, input_tensor=None, label_tensor=None, **kwargs):
        """
        Notes:
            I left `input_tensor` and `label_tensor` so that in the future some pipeline
            tensors can be connected to the model, instead of the placeholders as is now.
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
            regularizable_vars = [var for var in trainable_var if F.get_param_name(var).split('/')[-1] in ('kernel:0', 'w:0', 'w_out:0' 'weight:0')]
            if self.l1_reg > 0:
                assert len(regularizable_vars) > 0, "no parameter to apply regularization"
                l1_regularization_penalty = self.l1_reg * F.reduce_mean([F.reduce_mean(F.abs(var)) for var in regularizable_vars] )
                loss_ += l1_regularization_penalty
            else:
                l1_regularization_penalty = 0.

            if self.l2_reg > 0:
                assert len(regularizable_vars) > 0, "no parameter to apply regularization"
                l2_regularization_penalty = self.l2_reg * F.reduce_mean([F.reduce_mean(F.pow(var, 2)) for var in regularizable_vars])
                loss_ += l2_regularization_penalty
            else:
                l2_regularization_penalty = 0.

            regularization_penalty += l1_regularization_penalty + l2_regularization_penalty

            if is_training:
                assert self.child_train_op_kwargs is None, "legacy child_train_op_kwargs is not supported"
                train_op, lr, optimizer_ = F.get_train_op( 
                    loss=loss_,
                    variables=trainable_var,
                    optimizer=optimizer,
                    train_step=self.train_step)
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
                    op=Operation("Activation", activation=self.output_node.Layer_attributes.get('activation', 'linear'))
                )
        return model_output, dropout_placeholders

    def _refactorized_channels_for_skipcon(self, layer, out_filters, is_training):
        """for dealing with mismatch-dimensions in skip connections: use a linear transformation"""
        channel_num = self.data_format.index('C')
        #inp_c = F.get_shape(layer)[channel_num]
        actual_data_format = 'channels_first' if channel_num == 1 else 'channels_last'

        with F.variable_scope("path1_conv"):
            x = F.get_layer(x=layer, op=Operation('Conv1D', filters=out_filters, kernel_size=1, strides=1, padding="same"))
            x = F.get_layer(x=x, op=Operation('MaxPooling1D', pool_size=self.reduction_factor, strides=self.reduction_factor, padding="same", data_format=actual_data_format))
        return x

    def _layer(self, arc_seq, layer_id, prev_layers, start_idx, out_filters, is_training):
        inputs = prev_layers[-1]
        if self.data_format == "NWC":
            inp_w = F.get_shape(inputs)[1]
            #inp_c = F.get_shape(inputs)[2]

        elif self.data_format == "NCW":
            #inp_c = F.get_shape(inputs)[1]
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
        x = F.get_layer(
            x=x,
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

        with F.variable_scope("conv_1"):
            x = F.get_layer(
                x=inputs, 
                op=F.Operation('Conv1D', filters=filters, kernel_size=1, strides=1, padding="same"))
            x = F.get_layer(x=x, op=F.Operation('BatchNormalization', training=is_training))
            x = F.get_layer(x=x, op=F.Operation('Activation', activation="relu"))
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
class EnasCNNwDataDescriptor(EnasCnnModelBuilder):
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


"""
because TF1 supernet model builders are fairly tedious with low-level APIs,
we add another class to wrap it up with high-level functional methods, such
as `.fit`, `.predict`, `.evaluate`.
"""

class EnasAnnModel:
    def __init__(self, inputs, outputs, arc_seq, dag, session, dropouts=None, name='EnasModel'):
        """
        Parameters
        ----------
        inputs: amber.backend.Tensor
            input tensors/placeholders
        outputs: amber.backend.Tensor
            output tensors
        session: amber.backend.Session
            tensorflow Session for use
        name: str
            name for amber.backend.variable_scope; default is "EnasDAG"
        """
        assert type(inputs) in (F.TensorType, list), "get unexpected inputs types: %s" % type(inputs)
        assert type(outputs) in (F.TensorType, list), "get unexpected outputs types: %s" % type(outputs)
        self.arc_seq = arc_seq
        self.dag = dag
        self.inputs = [inputs] if type(inputs) is F.TensorType else inputs
        self.outputs = [outputs] if type(outputs) is F.TensorType else outputs
        labels = dag.child_model_label
        self.labels = [labels] if type(labels) is F.TensorType else labels
        self.session = session
        self.dropouts = dropouts
        self.dropout_placeholders = None
        self.name = name
        self.trainable_var = [var for var in F.trainable_variables(scope=self.name) ]
        self.train_op = None
        self.optimizer = None
        self.optimizer_ = None
        self.lr = None
        self.grad_norm = None
        self.loss = None
        self.metrics = None
        self.weights = None
        self.loss_weights = None
        self.is_compiled = False
        self.nodes = None
        self.use_pipe = None
        self.reinitialize_train_pipe = None

        # for Keras
        self.stop_training = False

    def compile(self, optimizer, loss=None, metrics=None, loss_weights=None):
        assert not self.is_compiled, "already compiled"
        if self.arc_seq is None:
            assert self.dag.train_fixed_arc is False, "You specified EnasAnnModelBuilder to train_fixed_arc=True, but didn't give arc_seq to the child model instance"
            self.train_op = self.dag.sample_train_op
            self.optimizer = self.dag.sample_optimizer
            self.loss = self.dag.sample_loss
            self.metrics = self.dag.sample_metrics
            self.weights = self.dag.sample_w_masks
            # self.loss_weights = self.dag.loss_weights
            self.dropout_placeholders = self.dag.sample_dropouts
        else:
            self.train_op = self.dag.fixed_train_op
            self.optimizer = self.dag.fixed_optimizer
            self.loss = self.dag.fixed_loss
            self.metrics = self.dag.fixed_metrics
            self.weights = self.dag.fixed_w_masks
            # self.loss_weights = self.dag.loss_weights
            self.dropout_placeholders = self.dag.fixed_dropouts

        if self.dropouts:
            assert len(self.dropout_placeholders) == len(self.dropouts), "provided dropout probs of len %i does not " \
                                                                         "match number of layers: %i" % (
                                                                         len(self.dropout_placeholders),
                                                                         len(self.dropouts))

        if self.arc_seq is None and self.dag.feature_model is not None and self.dag.feature_model.pseudo_inputs_pipe is not None:
            self.use_pipe = True
            self.reinitialize_train_pipe = True
        else:
            self.use_pipe = False

        self.is_compiled = True
        return

    def _make_feed_dict(self, x=None, y=None, is_training_phase=False):
        assert x is None or type(x) is list, "x arg for _make_feed_dict must be List"
        assert y is None or type(y) is list, "y arg for _make_feed_dict must be List"
        if self.arc_seq is None:
            feed_dict = {}
        else:
            assert len(self.arc_seq) == len(self.dag.input_arc), f"got {len(self.arc_seq)} length arc_seq as input, but expect {len(self.dag.input_arc)} length input_arc"
            feed_dict = {self.dag.input_arc[i]: self.arc_seq[i] for i in range(len(self.arc_seq))}
        if x is not None:
            for i in range(len(self.inputs)):
                if len(x[i].shape) > 1:
                    feed_dict.update({self.inputs[i]: x[i]})
                else:
                    feed_dict.update({self.inputs[i]: x[i][np.newaxis, :]})
        if y is not None:
            for i in range(len(self.outputs)):
                if len(y[i].shape) > 1:
                    feed_dict.update({self.labels[i]: y[i]})
                else:
                    feed_dict.update({self.labels[i]: np.expand_dims(y[i], -1)})
        if is_training_phase and self.dropouts:
            feed_dict.update({self.dropout_placeholders[i]: self.dropouts[i]
                              for i in range(len(self.dropouts))})
        return feed_dict

    def fit_generator(self,
                      generator,
                      steps_per_epoch=None,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      shuffle=True):
        if workers > 1:
            from keras.utils.data_utils import GeneratorEnqueuer
            enqueuer = GeneratorEnqueuer(
                generator,
                use_multiprocessing=use_multiprocessing)

            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            output_generator = generator

        callback_list = F.get_callback('CallbackList')(callbacks=callbacks)
        callback_list.set_model(self)
        callback_list.on_train_begin()
        if steps_per_epoch is None: steps_per_epoch = len(generator)
        assert steps_per_epoch is not None
        hist = {'loss': [], 'val_loss': []}
        for epoch in range(epochs):
            seen = 0
            epoch_logs = {'loss': 0, 'val_loss': 0}
            t = trange(steps_per_epoch) if verbose == 1 else range(steps_per_epoch)
            for _ in t:
                generator_output = next(output_generator)
                x, y = generator_output
                if x is None or len(x) == 0:
                    # Handle data tensors support when no input given
                    # step-size = 1 for data tensors
                    batch_size = 1
                elif isinstance(x, list):
                    batch_size = x[0].shape[0]
                # elif isinstance(x, dict):
                #     batch_size = list(x.values())[0].shape[0]
                else:
                    batch_size = x.shape[0]
                    x = [x]
                    y = [y]
                batch_loss, batch_metrics = self.train_on_batch(x, y)
                epoch_logs['loss'] += batch_loss * batch_size
                seen += batch_size

            for k in epoch_logs:
                epoch_logs[k] /= seen
            hist['loss'].append(epoch_logs['loss'])

            if validation_data:
                val_loss_and_metrics = self.evaluate(validation_data[0], validation_data[1])
                hist['val_loss'].append(val_loss_and_metrics[0])
                epoch_logs.update({'val_loss': val_loss_and_metrics[0]})

            callback_list.on_epoch_end(epoch, epoch_logs)

            if self.stop_training:
                break
        if workers > 1:
            enqueuer.stop()
        return hist

    def train_on_batch(self, x, y):
        feed_dict = self._make_feed_dict(x, y, is_training_phase=True)
        _, batch_loss, batch_metrics = self.session.run([self.train_op, self.loss, self.metrics], feed_dict=feed_dict)
        return batch_loss, batch_metrics

    def fit(self, x, y, batch_size=None, nsteps=None, epochs=1, verbose=1, callbacks=None, validation_data=None):
        hist = {'loss': [], 'val_loss': []}
        total_len = len(y[0]) if type(y) is list else len(y)
        batch_size = batch_size or 32
        if nsteps is None:
            nsteps = total_len // batch_size
        callback_list = F.get_callback('CallbackList')(callbacks=callbacks)
        callback_list.set_model(self)
        callback_list.on_train_begin()
        assert epochs > 0
        g = batchify(x, y, batch_size)
        for epoch in range(epochs):
            t = trange(nsteps) if verbose == 1 else range(nsteps)
            metrics_val = []
            curr_loss = None
            for it in t:
                try:
                    x_, y_ = next(g)
                except StopIteration:
                    g = batchify(x, y, batch_size)
                    x_, y_ = next(g)
                feed_dict = self._make_feed_dict(x_, y_, is_training_phase=True)
                _, batch_loss, batch_metrics = self.session.run([self.train_op, self.loss, self.metrics],
                                                                feed_dict=feed_dict)
                if len(metrics_val):
                    metrics_val = list(map(lambda x: x[0] * 0.95 + x[1] * 0.05, zip(metrics_val, batch_metrics)))
                else:
                    metrics_val = batch_metrics
                curr_loss = batch_loss if curr_loss is None else curr_loss * 0.95 + batch_loss * 0.05
                if verbose == 1:
                    t.set_postfix(loss="%.4f" % curr_loss)
                if verbose == 2 and it % 1000 == 0:
                    print("%s %i/%i, loss=%.5f" % (datetime.datetime.now().strftime("%H:%M:%S"), it, nsteps, curr_loss), flush=True)

            hist['loss'].append(curr_loss)
            logs = {'loss': curr_loss}
            if validation_data:
                val_loss_and_metrics = self.evaluate(validation_data[0], validation_data[1])
                hist['val_loss'].append(val_loss_and_metrics[0])
                logs.update({'val_loss': val_loss_and_metrics[0]})

            if verbose:
                if validation_data:
                    print("Epoch %i, loss=%.3f, metrics=%s; val=%s" % (
                    epoch, curr_loss, metrics_val, val_loss_and_metrics))
                else:
                    print("Epoch %i, loss=%.3f, metrics=%s" % (epoch, curr_loss, metrics_val))

            callback_list.on_epoch_end(epoch=epoch, logs=logs)
            if self.stop_training:
                break
        return hist

    def predict(self, x, batch_size=None):
        if type(x) is not list: x = [x]
        if batch_size is None:
            batch_size = min(1000, len(x[0]))
        y_pred_ = []
        for x_ in batchify(x, None, batch_size=batch_size, shuffle=False, drop_remainder=False):
            feed_dict = self._make_feed_dict(x_)
            y_pred = self.session.run(self.outputs, feed_dict)
            y_pred_.append(y_pred)
        y_pred = [np.concatenate(t, axis=0) for t in zip(*y_pred_)]
        if len(y_pred) > 1:
            y_pred = [y for y in y_pred]
        else:
            y_pred = y_pred[0]
        return y_pred

    def evaluate(self, x, y, batch_size=None, verbose=0):
        if batch_size is None:
            batch_size = min(100, x.shape[0])
        loss_and_metrics = []
        seen = 0
        if verbose:
            gen = tqdm(batchify(x, y, batch_size=batch_size, shuffle=False))
        else:
            gen = batchify(x, y, batch_size=batch_size, shuffle=False)
        for x_, y_ in gen:
            feed_dict = self._make_feed_dict(x_, y_)
            loss, metrics = self.session.run([self.loss, self.metrics], feed_dict=feed_dict)
            this_batch_size = x_[0].shape[0]
            if not len(loss_and_metrics):
                loss_and_metrics = [loss * this_batch_size] + [x * this_batch_size for x in metrics]
            else:
                tmp = [loss] + metrics
                loss_and_metrics = [loss_and_metrics[i] + this_batch_size * tmp[i] for i in range(len(tmp))]
            seen += this_batch_size
        loss_and_metrics = [x / seen for x in loss_and_metrics]
        return loss_and_metrics

    def save(self, *args, **kwargs):
        """
        TODO
        -----
            save model architectures
        """
        warnings.warn("Not implemented yet; rolling back to `save_weights`")
        self.save_weights(*args, **kwargs)
        return

    def save_weights(self, filepath, **kwargs):
        weights = self.get_weights()
        with h5py.File(filepath, 'w') as hf:
            for i, d in enumerate(weights):
                hf.create_dataset(name='layer_{:d}/weight'.format(i), data=d[0])
                hf.create_dataset(name='layer_{:d}/bias'.format(i), data=d[1])

    def load_weights(self, filepath, **kwargs):
        total_layers = len(self.weights)
        weights = []
        with h5py.File(filepath, 'r') as hf:
            for i in range(total_layers):
                w_key = "layer_{:d}/weight".format(i)
                b_key = "layer_{:d}/bias".format(i)
                weights.append((hf.get(w_key).value, hf.get(b_key).value))
        self.set_weights(weights)

    def get_weights(self, **kwargs):
        weights = self.session.run(self.weights, feed_dict=self._make_feed_dict())
        return weights

    def set_weights(self, weights, **kwargs):
        assign_ops = []
        weights_vars = list(zip(self.dag.w, self.dag.b)) + list(zip(self.dag.w_out, self.dag.b_out))
        for i in range(len(weights)):
            # weight
            assign_ops.append(F.assign(weights_vars[i][0], weights[i][0]))
            # bias
            assign_ops.append(F.assign(weights_vars[i][1], weights[i][1]))
        self.session.run(assign_ops)


class EnasCnnModel:
    """
    TODO
    -----
    - re-write weights save/load
    - use the input/output/label tensors provided by EnasConv1dDAG; this should unify the
      fit method when using placeholder and Tensor pipelines - probably still need two separate
      methods though

    """

    def __init__(self, inputs, outputs, labels, arc_seq, dag, session, dropouts=None, use_pipe=None, name='EnasModel',
                 **kwargs):
        assert type(inputs) in (F.TensorType, list), "get unexpected inputs types: %s" % type(inputs)
        assert type(outputs) in (F.TensorType, list), "get unexpected outputs types: %s" % type(outputs)
        self.arc_seq = arc_seq
        self.dag = dag
        self.inputs = [inputs] if type(inputs) is F.TensorType else inputs
        self.outputs = [outputs] if type(outputs) is F.TensorType else outputs
        self.callbacks = None
        self.labels = [labels] if type(labels) is F.TensorType else labels
        self.session = session
        self.dropouts = dropouts
        self.dropout_placeholders = None
        self.name = name
        self.trainable_var = [var for var in F.trainable_variables(scope=self.name)]
        self.train_op = None
        self.optimizer = None
        self.optimizer_ = None
        self.lr = None
        self.grad_norm = None
        self.loss = None
        self.metrics = None
        self.weights = None
        self.loss_weights = None
        self.is_compiled = False

        self.use_pipe = False if use_pipe is None else use_pipe
        self.reinitialize_train_pipe = None

        # added 2020.5.17: add default feed-dict for sampled architecture, to account for data description NAS
        self.sample_dag_feed_dict = kwargs.pop("sample_dag_feed_dict", {})

        # for Keras
        self.stop_training = False
        self.metrics_name = []
        self.batch_size = self.dag.batch_size

    def compile(self, optimizer=None, loss=None, metrics=None, loss_weights=None):
        assert not self.is_compiled, "already compiled"
        if self.arc_seq is None:
            assert self.dag.train_fixed_arc is False, "You specified EnasCnnModelBuilder to train_fixed_arc=True, but didn't give arc_seq to the child model instance"
            self.train_op = self.dag.sample_train_op
            self.optimizer = self.dag.sample_optimizer
            self.loss = self.dag.sample_loss
            self.metrics = self.dag.sample_metrics
            # self.loss_weights = self.dag.loss_weights
            self.dropout_placeholders = self.dag.sample_dropouts
        else:
            self.train_op = self.dag.fixed_train_op
            self.optimizer = self.dag.fixed_optimizer
            self.loss = self.dag.fixed_loss
            self.metrics = self.dag.fixed_metrics
            # self.loss_weights = self.dag.loss_weights
            self.dropout_placeholders = self.dag.fixed_dropouts

        if self.dropouts:
            assert len(self.dropout_placeholders) == len(self.dropouts), "provided dropout probs of len %i does not " \
                                                                         "match number of layers: %i" % (
                                                                         len(self.dropout_placeholders),
                                                                         len(self.dropouts))
        if self.use_pipe:
            self.reinitialize_train_pipe = True
        if metrics:
            metrics = [metrics] if type(metrics) is not list else metrics
            self.metrics_name = [str(m) for m in metrics]
        self.weights = [v for v in F.trainable_variables(scope=self.name)]
        self.is_compiled = True

    def _make_feed_dict(self, x=None, y=None, is_training_phase=False):
        assert x is None or type(x) is list, "x arg for _make_feed_dict must be List, got %s" % type(x)
        assert y is None or type(y) is list, "x arg for _make_feed_dict must be List, got %s" % type(y)
        if self.arc_seq is None:
            feed_dict = self.sample_dag_feed_dict
        else:
            feed_dict = {self.dag.input_arc[i]: self.arc_seq[i] for i in range(len(self.arc_seq))}
        if x is not None:
            for i in range(len(self.inputs)):
                if len(x[i].shape) > 1:
                    feed_dict.update({self.inputs[i]: x[i]})
                else:
                    feed_dict.update({self.inputs[i]: x[i][np.newaxis, :]})
        if y is not None:
            for i in range(len(self.outputs)):
                if len(y[i].shape) > 1:
                    feed_dict.update({self.labels[i]: y[i]})
                else:
                    feed_dict.update({self.labels[i]: np.expand_dims(y[i], -1)})
        if is_training_phase and self.dropouts:
            feed_dict.update({self.dropout_placeholders[i]: self.dropouts[i]
                              for i in range(len(self.dropouts))})
        elif (not is_training_phase) and len(self.dropout_placeholders):
            feed_dict.update({self.dropout_placeholders[i]: 0.0
                              for i in range(len(self.dropout_placeholders))})

        return feed_dict

    def fit(self, x, y, batch_size=None, nsteps=None, epochs=1, verbose=1, callbacks=None, validation_data=None):
        assert self.is_compiled, "Must compile model first"
        assert epochs > 0
        x = x if type(x) is list else [x]
        y = y if type(y) is list else [y]
        if nsteps is None:
            total_len = len(y[0]) if type(y) is list else len(y)
            nsteps = total_len // batch_size
        # BaseLogger should always be the first metric since it computes the stats on epoch end
        base_logger = F.get_callback('BaseLogger')(stateful_metrics=["val_%s" % m for m in self.metrics_name] + ['val_loss', 'size'])
        base_logger_params = {'metrics': ['loss'] + self.metrics_name}
        if validation_data:
            base_logger_params['metrics'] += ['val_%s' % m for m in base_logger_params['metrics']]
        base_logger.set_params(base_logger_params)
        hist = F.get_callback('History')()
        if callbacks is None:
            callbacks = [base_logger] + [hist]
        elif type(callbacks) is list:
            callbacks = [base_logger] + callbacks + [hist]
        else:
            callbacks = [base_logger] + [callbacks] + [hist]
        callback_list = F.get_callback('CallbackList')(callbacks=callbacks)
        callback_list.set_model(self)
        callback_list.on_train_begin()
        self.callbacks = callback_list
        for epoch in range(epochs):
            g = batchify(x, y, batch_size) if batch_size else None
            t = trange(nsteps) if verbose == 1 else range(nsteps)
            callback_list.on_epoch_begin(epoch)
            for it in t:
                x_, y_ = next(g) if g else (None, None)
                batch_logs = self.train_on_batch(x_, y_)
                callback_list.on_batch_end(it, batch_logs)
                curr_loss = base_logger.totals['loss'] / base_logger.seen
                if verbose == 1:
                    t.set_postfix(loss="%.4f" % curr_loss)
                if verbose == 2:
                    if it % 1000 == 0:
                        print(
                            "%s %i/%i, loss=%.5f" %
                            (datetime.datetime.now().strftime("%H:%M:%S"), it, nsteps, curr_loss),
                            flush=True)

            if validation_data:
                val_logs = self.evaluate(validation_data[0], validation_data[1])
                base_logger.on_batch_end(None, val_logs)

            epoch_logs = {}
            callback_list.on_epoch_end(epoch=epoch, logs=epoch_logs)

            if verbose:
                if validation_data:
                    to_print = ['loss'] + self.metrics_name + ['val_loss'] + ['val_%s' % m for m in self.metrics_name]
                else:
                    to_print = ['loss'] + self.metrics_name
                prog = ", ".join(["%s=%.4f" % (name, hist.history[name][-1]) for name in to_print])
                print("Epoch %i, %s" % (epoch, prog), flush=True)

            if self.stop_training:
                break

        return hist.history

    def fit_generator(self):
        raise NotImplementedError("not implemented")

    def train_on_batch(self, x=None, y=None):
        assert self.is_compiled, "Must compile model first"
        feed_dict = self._make_feed_dict(x, y, is_training_phase=True)
        batch_size = x[0].shape[0] if x is not None else self.batch_size
        _, batch_loss, batch_metrics = self.session.run([self.train_op, self.loss, self.metrics], feed_dict=feed_dict)
        logs = {self.metrics_name[i]: batch_metrics[i] for i in range(len(self.metrics_name))}
        logs.update({'loss': batch_loss, 'size': batch_size})
        return logs

    def evaluate(self, x, y, batch_size=None, verbose=0):
        assert self.is_compiled, "Must compile model first"
        batch_size = batch_size or self.batch_size
        loss_and_metrics = []
        seen = 0
        gen = tqdm(batchify(x, y, batch_size=batch_size, shuffle=False, drop_remainder=False)) if verbose else \
            batchify(x, y, batch_size=batch_size, shuffle=False, drop_remainder=False)
        for x_, y_ in gen:
            feed_dict = self._make_feed_dict(x_, y_)
            loss, metrics = self.session.run([self.loss, self.metrics], feed_dict=feed_dict)
            this_batch_size = x_[0].shape[0]
            if not len(loss_and_metrics):
                loss_and_metrics = [loss * this_batch_size] + [x * this_batch_size for x in metrics]
            else:
                tmp = [loss] + metrics
                loss_and_metrics = [loss_and_metrics[i] + this_batch_size * tmp[i] for i in range(len(tmp))]
            seen += this_batch_size
        loss_and_metrics = [x / seen for x in loss_and_metrics]
        # return loss_and_metrics
        logs = {'val_loss': loss_and_metrics.pop(0)}
        logs.update({'val_%s' % self.metrics_name[i]: loss_and_metrics[i] for i in range(len(self.metrics_name))})
        return logs

    def predict(self, x, batch_size=None, verbose=0):
        assert self.is_compiled, "Must compile model first"
        if type(x) is not list: x = [x]
        batch_size = batch_size or self.batch_size
        y_pred_ = []
        if verbose:
            gen = tqdm(batchify(x, None, batch_size=batch_size, shuffle=False, drop_remainder=False))
        else:
            gen = batchify(x, None, batch_size=batch_size, shuffle=False, drop_remainder=False)
        for x_ in gen:
            feed_dict = self._make_feed_dict(x_)
            y_pred = self.session.run(self.outputs, feed_dict)
            y_pred_.append(y_pred)
        y_pred = [np.concatenate(t, axis=0) for t in zip(*y_pred_)]
        if len(y_pred) > 1:
            y_pred = [y for y in y_pred]
        else:
            y_pred = y_pred[0]
        return y_pred

    def save(self, *args, **kwargs):
        """
        TODO
        ------
        save model architectures
        """
        warnings.warn("Not implemented yet; rolling back to `save_weights`", stacklevel=2)
        self.save_weights(*args, **kwargs)
        return

    def save_weights(self, filepath, **kwargs):
        weights = self.get_weights()
        with h5py.File(filepath, 'w') as hf:
            for i, d in enumerate(weights):
                hf.create_dataset(name=self.weights[i].name, data=d)

    def load_weights(self, filepath, **kwargs):
        weights = []
        with h5py.File(filepath, 'r') as hf:
            for i in range(len(self.weights)):
                key = self.weights[i].name
                weights.append(hf.get(key).value)
        self.set_weights(weights)

    def get_weights(self, **kwargs):
        weights = self.session.run(self.weights, feed_dict=self._make_feed_dict())
        return weights

    def set_weights(self, weights, **kwargs):
        assign_ops = []
        for i in range(len(self.weights)):
            assign_ops.append(F.assign(self.weights[i], weights[i]))
        self.session.run(assign_ops)
