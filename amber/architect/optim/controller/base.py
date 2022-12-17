from .... import backend as F
import os
import numpy as np
from ...base import BaseSearcher


def lstm(x, prev_c, prev_h, w):
    ifog = F.matmul(F.concat([x, prev_h], axis=1), w)
    i, f, o, g = F.split(ifog, 4, axis=1)
    i = F.sigmoid(i)
    f = F.sigmoid(f)
    o = F.sigmoid(o)
    g = F.tanh(g)
    next_c = i * g + f * prev_c
    next_h = o * F.tanh(next_c)
    return next_c, next_h


def stack_lstm(x, prev_c, prev_h, w):
    next_c, next_h = [], []
    for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
        inputs = x if layer_id == 0 else next_h[-1]
        curr_c, curr_h = lstm(inputs, _c, _h, _w)
        next_c.append(curr_c)
        next_h.append(curr_h)
    return next_c, next_h


def proximal_policy_optimization_loss(curr_prediction, curr_onehot, old_prediction, old_onehotpred, rewards, advantage, clip_val, beta=None):
    rewards_ = F.squeeze(rewards, axis=1)
    advantage_ = F.squeeze(advantage, axis=1)

    entropy = 0
    r = 1
    for t, (p, onehot, old_p, old_onehot) in \
            enumerate(zip(curr_prediction, curr_onehot, old_prediction, old_onehotpred)):
        ll_t = F.log(F.reduce_sum(old_onehot * p))
        ll_0 = F.log(F.reduce_sum(old_onehot * old_p))
        r_t = F.exp(ll_t - ll_0)
        r = r * r_t
        # approx entropy
        entropy += -F.reduce_mean(F.log(F.reduce_sum(onehot * p, axis=1)))

    surr_obj = F.reduce_mean(F.abs(1 / (rewards_ + 1e-8)) *
                              F.minimum(r * advantage_,
                                         F.clip_by_value(r,
                                                          clip_value_min=1 - clip_val,
                                                          clip_value_max=1 + clip_val) * advantage_)
                              )
    if beta:
        # maximize surr_obj for learning and entropy for regularization
        return - surr_obj + beta * (- entropy)
    else:
        return - surr_obj


class BaseController(BaseSearcher):
    """Base class for controllers
    """

    def __init__(self, model_space=None, buffer=None, *args, **kwargs):
        super().__init__()
        self.model_space = model_space
        self.buffer = None
        #self.decoder = get_decoder(arc_decoder)

    def __str__(self):
        return "AMBER Controller for architecture search"

    def _create_weight(self):
        """Private method for creating tensors; called at initialization"""
        with F.variable_scope("create_weights"):
            with F.variable_scope("lstm", reuse=False):
                self.w_lstm = []
                for layer_id in range(self.lstm_num_layers):
                    with F.variable_scope("lstm_layer_{}".format(layer_id)):
                        w = F.create_parameter(
                            "w", [2 * self.lstm_size, 4 * self.lstm_size],
                            initializer='uniform'
                            )
                        self.w_lstm.append(w)

            # g_emb: initial controller hidden state tensor; to be learned
            self.g_emb = F.create_parameter("g_emb", [1, self.lstm_size], initializer='uniform')

            # XXX: all torch.nn.Parameters not directly assigned as a class.attribute
            # XXX: are not registered nor recognized by self.parameters()
            # w_emb: embedding for computational operations
            self.w_emb = {"start": []}

            with F.variable_scope("emb"):
                for layer_id in range(self.num_layers):
                    with F.variable_scope("layer_{}".format(layer_id)):
                        if self.share_embedding:
                            if layer_id not in self.share_embedding:
                                self.w_emb["start"].append(F.create_parameter(
                                    "w_start", [self.num_choices_per_layer[layer_id], self.lstm_size], initializer='uniform'))
                            else:
                                shared_id = self.share_embedding[layer_id]
                                assert shared_id < layer_id, \
                                    "You turned on `share_embedding`, but specified the layer %i " \
                                    "to be shared with layer %i, which is not built yet" % (layer_id, shared_id)
                                self.w_emb["start"].append(self.w_emb["start"][shared_id])

                        else:
                            self.w_emb["start"].append(F.create_parameter(
                                "w_start", [self.num_choices_per_layer[layer_id], self.lstm_size], initializer='uniform'))

            # w_soft: dictionary of tensors for transforming RNN hiddenstates to softmax classifier
            self.w_soft = {"start": []}
            with F.variable_scope("softmax"):
                for layer_id in range(self.num_layers):
                    if self.share_embedding:
                        if layer_id not in self.share_embedding:
                            with F.variable_scope("layer_{}".format(layer_id)):
                                self.w_soft["start"].append(F.create_parameter(
                                    name="w_start", shape=[self.lstm_size, self.num_choices_per_layer[layer_id]], initializer='uniform'))
                        else:
                            shared_id = self.share_embedding[layer_id]
                            assert shared_id < layer_id, \
                                "You turned on `share_embedding`, but specified the layer %i " \
                                "to be shared with layer %i, which is not built yet" % (layer_id, shared_id)
                            self.w_soft["start"].append(self.w_soft['start'][shared_id])
                    else:
                        with F.variable_scope("layer_{}".format(layer_id)):
                            self.w_soft["start"].append(F.create_parameter(
                                "w_start", [self.lstm_size, self.num_choices_per_layer[layer_id]], initializer='uniform'))

            #  w_attn_1/2, v_attn: for sampling skip connections
            if self.with_skip_connection:
                with F.variable_scope("attention"):
                    self.w_attn_1 = F.create_parameter("w_1", [self.lstm_size, self.lstm_size],initializer='uniform')
                    self.w_attn_2 = F.create_parameter("w_2", [self.lstm_size, self.lstm_size],initializer='uniform')
                    self.v_attn = F.create_parameter("v", [self.lstm_size, 1],initializer='uniform')
            else:
                self.w_attn_1 = None
                self.w_attn_2 = None
                self.v_attn = None
    
    def _squeeze_logits(self, logit):
        """modify logits to be more smooth, if necessary"""
        if self.temperature is not None:
            logit /= self.temperature
        if self.tanh_constant is not None:
            logit = self.tanh_constant * F.tanh(logit)
        return logit
    
    def _stepwise_loss(self, logits, tokens, batch_size=None):
        # sparse NLL/CCE: logits are weights, labels are integers
        log_prob = F.get_loss('NLLLoss_with_logits', y_true=tokens, y_pred=logits, reduction='none')
        log_prob = F.reshape(log_prob, [batch_size, -1])
        log_prob = F.reduce_sum(log_prob, axis=-1)
        entropy = F.stop_gradient(F.reduce_sum(log_prob * F.exp(-log_prob)))
        return log_prob, entropy

    def forward(self, input_arc=None):
        is_training = False if input_arc is None else True
        ops_each_layer = 1
        hidden_states = []
        anchors = []
        anchors_w_1 = []
        entropys = []
        probs_ = []
        log_probs = []
        skip_count = []
        skip_penaltys = []
        if is_training:
            batch_size = F.shape(input_arc[0])[0]
        else:
            batch_size = 1
        prev_c = [F.zeros([batch_size, self.lstm_size], F.float32) for _ in
                  range(self.lstm_num_layers)]
        prev_h = [F.zeros([batch_size, self.lstm_size], F.float32) for _ in
                  range(self.lstm_num_layers)]
        skip_targets = F.Variable([1.0 - self.skip_target, self.skip_target], trainable=False, shape=(2,),
                                   dtype=F.float32)
        # only expand `g_emb` if necessary
        g_emb_nrow = self.g_emb.shape[0] if type(self.g_emb.shape[0]) in (int, type(None)) \
            else self.g_emb.shape[0].value
        if self.g_emb.shape[0] is not None and g_emb_nrow == 1:
            inputs = F.matmul(F.ones((batch_size, 1)), self.g_emb)
        else:
            inputs = self.g_emb

        arc_pointer = 0
        skip_conn_record = []
        arc_seq = []

        for layer_id in range(self.num_layers):
            # STEP 1: for each layer, sample operations first
            # by un-rolling RNN
            next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c, next_h
            hidden_states.append(prev_h)

            logit = F.matmul(next_h[-1], self.w_soft["start"][layer_id])  # out_filter x 1
            logit = self._squeeze_logits(logit)
            probs_.append(F.softmax(logit))

            if is_training:
                start = input_arc[arc_pointer]
            else:
                start = F.multinomial(logit, 1)
            start = F.reshape(start, [batch_size])
            start = F.cast(start, F.int32)
            arc_seq.append(start)

            # sparse NLL/CCE: logits are weights, labels are integers
            log_prob, entropy = self._stepwise_loss(tokens=start, logits=logit, batch_size=batch_size)
            log_probs.append(log_prob)
            entropys.append(entropy)
            # inputs: get a row slice of [out_filter[i], lstm_size]
            inputs = F.embedding_lookup(self.w_emb["start"][layer_id], start)
            # END STEP 1

            # STEP 2: sample the connections, unless the first layer
            # the number `skip` of each layer grows as layer_id grows
            if self.with_skip_connection:
                if layer_id > 0:
                    next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                    prev_c, prev_h = next_c, next_h
                    hidden_states.append(prev_h)

                    #query = F.concat(anchors_w_1, axis=0)  # layer_id x lstm_size
                    query = F.transpose(F.stack(anchors_w_1), [1, 0, 2])  # batch_size x layer_id x lstm_size
                    # w_attn_2: lstm_size x lstm_size
                    # P(Layer j is an input to layer i) = sigmoid(v^T %*% tanh(W_prev ∗ h_j + W_curr ∗ h_i))
                    query = F.tanh(
                        query + F.expand_dims(F.matmul(next_h[-1], self.w_attn_2),
                                               axis=1))  # query: layer_id x lstm_size
                    query = F.reshape(query, (batch_size * layer_id, self.lstm_size))
                    query = F.matmul(query, self.v_attn)  # query: batch_size*layer_id x 1

                    if self.skip_connection_unique_connection:
                        mask = F.stop_gradient(F.reduce_sum(F.stack(skip_conn_record), axis=0))
                        mask = F.slice(mask, begin=[0, 0], size=[batch_size, layer_id])
                        mask1 = F.greater(mask, 0)
                        query = F.where(mask1, y=query, x=F.fill(F.shape(query), -10000.))
                    
                    logit = F.concat([-query, query], axis=1)  # logit: layer_id x 2
                    logit = self._squeeze_logits(logit)
                    probs_.append(F.reshape(F.softmax(logit), [batch_size, layer_id, 2]))

                    if is_training:
                        skip = input_arc[(arc_pointer + ops_each_layer): (arc_pointer + ops_each_layer + layer_id)]
                    else:
                        skip = F.multinomial(logit, 1)  # layer_id x 1 of booleans
                    skip = F.reshape(F.transpose(skip), [batch_size * layer_id])
                    skip = F.cast(skip, F.int32)
                    arc_seq.append(skip)
                    skip_conn_record.append(
                        F.concat([F.cast(skip, F.float32), F.zeros(self.num_layers - layer_id)], axis=0))

                    skip_prob = F.sigmoid(logit)
                    kl = skip_prob * F.log(skip_prob / skip_targets)  # (batch_size*layer_id, 2)
                    kl = F.reduce_sum(kl, axis=1)    # (batch_size*layer_id,)
                    kl = F.reshape(kl, [batch_size, -1])  # (batch_size, layer_id)
                    skip_penaltys.append(kl)

                    log_prob, entropy = self._stepwise_loss(tokens=skip, logits=logit, batch_size=batch_size)
                    log_probs.append(log_prob)
                    entropys.append(entropy)

                    skip = F.cast(skip, F.float32)
                    skip = F.reshape(skip, [batch_size, 1, layer_id])
                    skip_count.append(F.reduce_sum(skip, axis=2))
                    
                    anchors_ = F.stack(anchors)
                    anchors_ = F.transpose(anchors_, [1, 0, 2])  # batch_size, layer_id, lstm_size
                    inputs = F.matmul(skip, anchors_)  # batch_size, 1, lstm_size
                    inputs = F.squeeze(inputs, axis=1)
                    inputs /= (1.0 + F.reduce_sum(skip, axis=2))  # batch_size, lstm_size
                
                else:
                    skip_conn_record.append(F.zeros((batch_size, self.num_layers)))

                anchors.append(next_h[-1])
                # next_h: 1 x lstm_size
                # anchors_w_1: 1 x lstm_size
                anchors_w_1.append(F.matmul(next_h[-1], self.w_attn_1))
                # added Sep.28.2019; removed as inputs for next layer, Nov.21.2019
                # next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                # prev_c, prev_h = next_c, next_h
                # hidden_states.append(prev_h)
            arc_pointer += ops_each_layer + layer_id * self.with_skip_connection
            # END STEP 2
        return arc_seq, probs_, log_probs, hidden_states, entropys, skip_count, skip_penaltys
    
    def store(self, state=None, prob=None, action=None, reward=None, *args, **kwargs):
        """Store all necessary information and rewards for a given architecture

        This is the receiving method for controller to interact with manager by storing the rewards for a given architecture.
        The architecture and its probabilities can be generated by ``get_action()`` method.

        Parameters
        ----------
        state : list
            The state for which the action and probabilities are drawn.

        prob : list of ndarray
            A list of probabilities for each operation and skip connections.

        action : list
            A list of architecture tokens ordered as::

                [categorical_operation_0,
                categorical_operation_1, binary_skip_0,
                categorical_operation_2, binary_skip_0, binary_skip_1,
                ...]

        reward : float
            Reward for this architecture, as evaluated by ``amber.architect.manager``

        Returns
        -------
        None

        """
        state = [[[0]]] if state is None else state
        self.buffer.store(state=state, prob=prob, action=action, reward=reward, **kwargs)
        return

    @staticmethod
    def remove_files(files, working_dir='.'):
        """Static method for removing files

        Parameters
        ----------
        files : list of str
            files to be removed

        working_dir : str
            filepath to working directory

        Returns
        -------
        None
        """
        for file in files:
            file = os.path.join(working_dir, file)
            if os.path.exists(file):
                os.remove(file)
    
    @property
    def search_space_size(self):
        input_blocks, output_blocks, num_layers, num_choices_per_layer = self.input_blocks, self.output_blocks, len(self.model_space), np.mean([len(layer) for layer in self.model_space])
        s = np.log10(num_choices_per_layer) * num_layers
        s += np.log10(2) * (num_layers-1)*num_layers/2
        s += np.log10(input_blocks) * num_layers + np.log10(output_blocks) * num_layers
        return s

    @staticmethod
    def convert_arc_to_onehot(controller):
        """Convert a categorical architecture sequence to a one-hot encoded architecture sequence

        Parameters
        ----------
        controller : amber.architect.controller
            An instance of controller

        Returns
        -------
        onehot_list : list
            a one-hot encoded architecture sequence
        """
        with_skip_connection = controller.with_skip_connection
        if hasattr(controller, 'with_input_blocks'):
            with_input_blocks = controller.with_input_blocks
            num_input_blocks = controller.num_input_blocks
        else:
            with_input_blocks = False
            num_input_blocks = 1
        arc_seq = controller.input_arc
        model_space = controller.model_space
        onehot_list = []
        arc_pointer = 0
        for layer_id in range(len(model_space)):
            onehot_list.append(F.squeeze(F.one_hot(tensor=arc_seq[arc_pointer], num_classes=len(model_space[layer_id])), axis=1))
            if with_input_blocks:
                inp_blocks_idx = arc_pointer + 1, arc_pointer + 1 + num_input_blocks * with_input_blocks
                tmp = []
                for i in range(inp_blocks_idx[0], inp_blocks_idx[1]):
                    tmp.append(F.squeeze(F.one_hot(arc_seq[i], 2), axis=1))
                onehot_list.append(F.transpose(F.stack(tmp), [1, 0, 2]))
            if layer_id > 0 and with_skip_connection:
                skip_con_idx = arc_pointer + 1 + num_input_blocks * with_input_blocks, \
                               arc_pointer + 1 + num_input_blocks * with_input_blocks + layer_id * with_skip_connection
                tmp = []
                for i in range(skip_con_idx[0], skip_con_idx[1]):
                    tmp.append(F.squeeze(F.one_hot(arc_seq[i], 2), axis=1))
                onehot_list.append(F.transpose(F.stack(tmp), [1, 0, 2]))
            arc_pointer += 1 + num_input_blocks * with_input_blocks + layer_id * with_skip_connection
        return onehot_list

    def train(self, *args, **kwargs):
        pass

    def get_action(self, *args, **kwargs):
        pass
