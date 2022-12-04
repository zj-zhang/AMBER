"""
General controller for searching computational operation per layer, and residual connection
"""

# Author       : zzjfrank
# Last Update  : Aug. 16, 2020

import os
import sys
import numpy as np
import h5py
from ..... import backend as F
from ..base import BaseController
from amber.architect.buffer import get_buffer
from amber.architect.commonOps import get_kl_divergence_n_entropy
from amber.architect.commonOps import proximal_policy_optimization_loss
from amber.architect.commonOps import stack_lstm


class GeneralController(BaseController):
    """
    GeneralController for neural architecture search

    This class searches for two steps:
        - computational operations for each layer
        - skip connections for each layer from all previous layers [optional]

    It is a modified version of enas: https://github.com/melodyguan/enas . Notable modifications include: dissection of
    sampling and training processes to enable better understanding of controller behaviors, buffering and logging;
    loss function can be optimized by either REINFORCE or PPO.

    TODO
    ----------
    Refactor the rest of the attributes to private.


    Parameters
    ----------
    model_space : amber.architect.ModelSpace
        A ModelSpace object constructed to perform architecture search for.

    with_skip_connection : bool
        If false, will not search residual connections and only search for computation operations per layer. Default is
        True.

    share_embedding : dict
        a Dictionary defining which child-net layers will share the softmax and embedding weights during Controller
        training and sampling. For example, ``{1:0, 2:0}`` means layer 1 and 2 will share the embedding with layer 0.

    use_ppo_loss : bool
        If true, use PPO loss for optimization instead of REINFORCE. Default is False.

    kl_threshold : float
        If KL-divergence between the sampling probabilities of updated controller parameters and that of original
        parameters exceeds kl_threshold within a single controller training step, triggers early-stopping to halt the
        controller training. Default is 0.05.

    buffer_size : int
        amber.architect.Buffer stores only the sampled architectures from the last ``buffer_size`` number of from previous
        controller steps, where each step has a number of sampled architectures as specified in ``amber.architect.ControllerTrainEnv``.

    batch_size : int
        How many architectures in a batch to train the controller

    session : F.Session
        The session where the controller tensors is placed

    train_pi_iter : int
        The number of epochs/iterations to train controller policy in one controller step.

    lstm_size : int
        The size of hidden units for stacked LSTM, i.e. controller RNN.

    lstm_num_layers : int
        The number of stacked layers for stacked LSTM, i.e. controller RNN.

    lstm_keep_prob : float
        keep_prob = 1 - dropout probability for stacked LSTM.

    tanh_constant : float
        If not None, the logits for each multivariate classification will be transformed by ``F.tanh`` then multiplied by
        tanh_constant. This can avoid over-confident controllers asserting probability=1 or 0 caused by logit going to +/- inf.
        Default is None.

    temperature : float
        The temperature is a scale factor to logits. Higher temperature will flatten the probabilities among different
        classes, while lower temperature will freeze them. Default is None, i.e. 1.

    optim_algo : str
        Optimizer for controller RNN. Can choose from ["adam", "sgd", "rmsprop"]. Default is "adam".

    skip_target : float
        The expected proportion of skip connections, i.e. the proportion of 1's in the skip/extra
        connections in the output `arc_seq`

    skip_weight : float
        The weight for skip connection kl-divergence from the expected `skip_target`

    name : str
        The name for this Controller instance; all ``F.Tensors`` will be placed under this VariableScope. This name
        determines which tensors will be initialized when a new Controller instance is created.


    Attributes
    ----------
    weights : list of F.Variable
        The list of all trainable ``F.Variable`` in this controller

    model_space : amber.architect.ModelSpace
        The model space which the controller will be searching from.

    buffer : amber.architect.Buffer
        The Buffer object stores the history architectures, computes the rewards, and gets feed dict for training.

    session : F.Session
        The reference to the session that hosts this controller instance.

    """

    def __init__(self, model_space, buffer_type='ordinal', with_skip_connection=True, share_embedding=None,
                 use_ppo_loss=False, kl_threshold=0.05, skip_connection_unique_connection=False, buffer_size=15,
                 batch_size=5, session=None, train_pi_iter=20, lstm_size=32, lstm_num_layers=2, lstm_keep_prob=1.0,
                 tanh_constant=None, temperature=None, optim_algo="adam", skip_target=0.8, skip_weight=None,
                 rescale_advantage_by_reward=False, name="controller", verbose=0, **kwargs):
        super().__init__(**kwargs)

        self.model_space = model_space
        # -----
        # FOR LEGACY ATTRIBUTES
        self.state_space = model_space
        # -----
        self.share_embedding = share_embedding
        self.with_skip_connection = with_skip_connection
        self.num_layers = len(model_space)
        self.num_choices_per_layer = [len(model_space[i]) for i in range(self.num_layers)]
        self.skip_connection_unique_connection = skip_connection_unique_connection

        buffer_fn = get_buffer(buffer_type)
        self.buffer = buffer_fn(max_size=buffer_size,
                                #ewa_beta=max(1 - 1. / buffer_size, 0.9),
                                discount_factor=0.,
                                is_squeeze_dim=True,
                                rescale_advantage_by_reward=rescale_advantage_by_reward)
        self.batch_size = batch_size
        self.verbose = verbose

        # need to use the same session throughout one App; ZZ 2020.3.2
        assert session is not None
        self.session = session if session else F.Session()
        self.train_pi_iter = train_pi_iter
        self.use_ppo_loss = use_ppo_loss
        self.kl_threshold = kl_threshold

        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_keep_prob = lstm_keep_prob
        self.tanh_constant = tanh_constant
        self.temperature = temperature

        self.skip_target = skip_target
        self.skip_weight = skip_weight
        if self.skip_weight is not None:
            assert self.with_skip_connection, "If skip_weight is not None, must have with_skip_connection=True"

        self.optim_algo = optim_algo
        self.name = name
        self.loss = 0

        with F.device_scope("/cpu:0"):
            with F.variable_scope(self.name):
                self._create_weight()
                self._build_sampler()
                self._build_trainer()
                self._build_train_op()
        # initialize variables in this scope
        self.weights = [var for var in F.trainable_variables(scope=self.name)]
        F.init_all_params(sess=self.session)

    def __str__(self):
        s = "GeneralController '%s' for %s" % (self.name, self.model_space)
        return s

    def _build_sampler(self):
        """Build the sampler ops and the log_prob ops.

        For sampler, the architecture sequence is randomly sampled, and only sample one architecture at each call to
        fill in self.sample_arc
        """
        anchors = []
        anchors_w_1 = []

        arc_seq = []
        hidden_states = []
        entropys = []
        probs_ = []
        log_probs = []
        skip_count = []
        skip_penaltys = []

        prev_c = [F.zeros([1, self.lstm_size], F.float32) for _ in
                  range(self.lstm_num_layers)]
        prev_h = [F.zeros([1, self.lstm_size], F.float32) for _ in
                  range(self.lstm_num_layers)]

        inputs = self.g_emb
        skip_targets = F.Variable([1.0 - self.skip_target, self.skip_target], trainable=False, shape=(2,),
                                   dtype=F.float32)
        skip_conn_record = []

        for layer_id in range(self.num_layers):
            # STEP 1: for each layer, sample operations first
            next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c, next_h
            hidden_states.append(prev_h)

            logit = F.matmul(next_h[-1], self.w_soft["start"][layer_id])  # out_filter x 1
            if self.temperature is not None:
                logit /= self.temperature
            if self.tanh_constant is not None:
                logit = self.tanh_constant * F.tanh(logit)
            probs_.append(F.softmax(logit))
            start = F.multinomial(logit, 1)
            start = F.cast(start, F.int32)
            start = F.reshape(start, [1])
            arc_seq.append(start)
            #log_prob = F.nn.sparse_softmax_cross_entropy_with_logits(
            #    logits=logit, labels=start)
            log_prob = F.get_loss('NLLLoss_with_logits', y_true=start, y_pred=logit)
            log_probs.append(log_prob)
            entropy = F.stop_gradient(log_prob * F.exp(-log_prob))
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

                    query = F.concat(anchors_w_1, axis=0)  # layer_id x lstm_size
                    # w_attn_2: lstm_size x lstm_size
                    query = F.tanh(query + F.matmul(next_h[-1], self.w_attn_2))  # query: layer_id x lstm_size
                    # P(Layer j is an input to layer i) = sigmoid(v^T %*% tanh(W_prev ∗ h_j + W_curr ∗ h_i))
                    query = F.matmul(query, self.v_attn)  # query: layer_id x 1
                    if self.skip_connection_unique_connection:
                        mask = F.stop_gradient(F.reduce_sum(F.stack(skip_conn_record), axis=0))
                        mask = F.slice(mask, begin=[0], size=[layer_id])
                        mask1 = F.greater(mask, 0)
                        query = F.where(mask1, y=query, x=F.fill(F.shape(query), -10000.))
                    logit = F.concat([-query, query], axis=1)  # logit: layer_id x 2
                    if self.temperature is not None:
                        logit /= self.temperature
                    if self.tanh_constant is not None:
                        logit = self.tanh_constant * F.tanh(logit)

                    probs_.append(F.expand_dims(F.softmax(logit), axis=0))
                    skip = F.multinomial(logit, 1)  # layer_id x 1 of booleans
                    skip = F.cast(skip, F.int32)
                    skip = F.reshape(skip, [layer_id])
                    arc_seq.append(skip)
                    skip_conn_record.append(
                        F.concat([F.cast(skip, F.float32), F.zeros(self.num_layers - layer_id)], axis=0))

                    skip_prob = F.sigmoid(logit)
                    kl = skip_prob * F.log(skip_prob / skip_targets)
                    kl = F.reduce_sum(kl)
                    skip_penaltys.append(kl)

                    #log_prob = F.nn.sparse_softmax_cross_entropy_with_logits(
                    #    logits=logit, labels=skip)
                    log_prob = F.get_loss('NLLLoss_with_logits', y_true=skip, y_pred=logit)
                    log_probs.append(F.reshape(F.reduce_sum(log_prob), [-1]))

                    entropy = F.stop_gradient(
                        F.reshape(F.reduce_sum(log_prob * F.exp(-log_prob)), [-1]))
                    entropys.append(entropy)

                    skip = F.cast(skip, F.float32)
                    skip = F.reshape(skip, [1, layer_id])
                    skip_count.append(F.reduce_sum(skip))
                    inputs = F.matmul(skip, F.concat(anchors, axis=0))
                    inputs /= (1.0 + F.reduce_sum(skip))
                else:
                    skip_conn_record.append(F.zeros(self.num_layers, 1))

                anchors.append(next_h[-1])
                # next_h: 1 x lstm_size
                # anchors_w_1: 1 x lstm_size
                anchors_w_1.append(F.matmul(next_h[-1], self.w_attn_1))
                # added Sep.28.2019; removed as inputs for next layer, Nov.21.2019
                # next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                # prev_c, prev_h = next_c, next_h
                # hidden_states.append(prev_h)

            # END STEP 2

        # for DEBUG use
        self.anchors = anchors
        self.anchors_w_1 = anchors_w_1
        self.sample_hidden_states = hidden_states

        # for class attr.
        arc_seq = F.concat(arc_seq, axis=0)
        self.sample_arc = F.reshape(arc_seq, [-1])
        entropys = F.stack(entropys)
        self.sample_entropy = F.reduce_sum(entropys)
        log_probs = F.stack(log_probs)
        self.sample_log_prob = F.reduce_sum(log_probs)
        skip_count = F.stack(skip_count)
        self.skip_count = F.reduce_sum(skip_count)
        skip_penaltys = F.stack(skip_penaltys)
        self.skip_penaltys = F.reduce_mean(skip_penaltys)
        self.sample_probs = probs_

    def _build_trainer(self):
        """"Build the trainer ops and the log_prob ops.

        For trainer, the input architectures are ``F.placeholder`` to receive previous architectures from buffer.
        It also supports batch computation.
        """
        anchors = []
        anchors_w_1 = []
        probs_ = []

        ops_each_layer = 1
        total_arc_len = sum(
            [ops_each_layer] +   # first layer
            [ops_each_layer + i * self.with_skip_connection for i in range(1, self.num_layers)]  # rest layers
        )
        self.total_arc_len = total_arc_len
        self.input_arc = [F.placeholder(shape=(None, 1), dtype=F.int32, name='arc_{}'.format(i))
                          for i in range(total_arc_len)]

        batch_size = F.shape(self.input_arc[0])[0]
        entropys = []
        log_probs = []
        skip_count = []
        skip_penaltys = []

        prev_c = [F.zeros([batch_size, self.lstm_size], F.float32) for _ in
                  range(self.lstm_num_layers)]
        prev_h = [F.zeros([batch_size, self.lstm_size], F.float32) for _ in
                  range(self.lstm_num_layers)]
        # only expand `g_emb` if necessary
        g_emb_nrow = self.g_emb.shape[0] if type(self.g_emb.shape[0]) in (int, type(None)) \
            else self.g_emb.shape[0].value
        if self.g_emb.shape[0] is not None and g_emb_nrow == 1:
            inputs = F.matmul(F.ones((batch_size, 1)), self.g_emb)
        else:
            inputs = self.g_emb
        skip_targets = F.Variable([1.0 - self.skip_target, self.skip_target], shape=(2,), trainable=False,
                                   dtype=F.float32)

        arc_pointer = 0
        skip_conn_record = []
        hidden_states = []
        for layer_id in range(self.num_layers):

            # STEP 1: compute log-prob for operations
            next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
            prev_c, prev_h = next_c, next_h
            hidden_states.append(prev_h)

            logit = F.matmul(next_h[-1], self.w_soft["start"][layer_id])  # batch_size x num_choices_layer_i
            if self.temperature is not None:
                logit /= self.temperature
            if self.tanh_constant is not None:
                logit = self.tanh_constant * F.tanh(logit)
            start = self.input_arc[arc_pointer]
            start = F.reshape(start, [batch_size])
            probs_.append(F.softmax(logit))

            #log_prob1 = F.nn.sparse_softmax_cross_entropy_with_logits(
            #    logits=logit, labels=start)
            log_prob1 = F.get_loss('NLLLoss_with_logits', y_true=start, y_pred=logit)
            log_probs.append(log_prob1)
            entropy = F.stop_gradient(log_prob1 * F.exp(-log_prob1))
            entropys.append(entropy)
            # inputs: get a row slice of [out_filter[i], lstm_size]
            # inputs = F.embedding_lookup(self.w_emb["start"][branch_id], start)
            inputs = F.embedding_lookup(self.w_emb["start"][layer_id], start)
            # END STEP 1

            # STEP 2: compute log-prob for skip connections, unless the first layer
            if self.with_skip_connection:
                if layer_id > 0:

                    next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                    prev_c, prev_h = next_c, next_h
                    hidden_states.append(prev_h)

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
                        mask = F.reshape(mask, (batch_size * layer_id, 1))
                        mask1 = F.greater(mask, 0)
                        query = F.where(mask1, y=query, x=F.fill(F.shape(query), -10000.))

                    logit = F.concat([-query, query], axis=1)  # logit: batch_size*layer_id x 2
                    if self.temperature is not None:
                        logit /= self.temperature
                    if self.tanh_constant is not None:
                        logit = self.tanh_constant * F.tanh(logit)

                    probs_.append(F.reshape(F.softmax(logit), [batch_size, layer_id, 2]))

                    skip = self.input_arc[(arc_pointer + ops_each_layer): (arc_pointer + ops_each_layer + layer_id)]
                    # print(layer_id, (arc_pointer+2), (arc_pointer+2 + layer_id), skip)
                    skip = F.reshape(F.transpose(skip), [batch_size * layer_id])
                    skip = F.cast(skip, F.int32)

                    skip_prob = F.sigmoid(logit)
                    kl = skip_prob * F.log(skip_prob / skip_targets)  # (batch_size*layer_id, 2)
                    kl = F.reduce_sum(kl, axis=1)    # (batch_size*layer_id,)
                    kl = F.reshape(kl, [batch_size, -1])  # (batch_size, layer_id)
                    skip_penaltys.append(kl)

                    #log_prob3 = F.nn.sparse_softmax_cross_entropy_with_logits(
                    #    logits=logit, labels=skip)
                    log_prob3 = F.get_loss('NLLLoss_with_logits', y_true=skip, y_pred=logit)
                    log_prob3 = F.reshape(log_prob3, [batch_size, -1])
                    log_probs.append(F.reduce_sum(log_prob3, axis=1))

                    entropy = F.stop_gradient(
                        F.reduce_sum(log_prob3 * F.exp(-log_prob3), axis=1))
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

                # next_h: batch_size x lstm_size
                anchors.append(next_h[-1])
                # anchors_w_1: batch_size x lstm_size
                anchors_w_1.append(F.matmul(next_h[-1], self.w_attn_1))
                # added 9.28.2019; removed 11.21.2019
                # next_c, next_h = stack_lstm(inputs, prev_c, prev_h, self.w_lstm)
                # prev_c, prev_h = next_c, next_h

            arc_pointer += ops_each_layer + layer_id * self.with_skip_connection
        # END STEP 2

        # for DEBUG use
        self.train_hidden_states = hidden_states

        # for class attributes
        self.entropys = F.stack(entropys)
        self.onehot_probs = probs_
        log_probs = F.stack(log_probs)
        self.onehot_log_prob = F.reduce_sum(log_probs, axis=0)
        skip_count = F.stack(skip_count)
        self.onehot_skip_count = F.reduce_sum(skip_count, axis=0)
        skip_penaltys_flat = [F.reduce_mean(x, axis=1) for x in skip_penaltys] # from (num_layer-1, batch_size, layer_id) to (num_layer-1, batch_size); layer_id makes each tensor of varying lengths in the list
        self.onehot_skip_penaltys = F.reduce_mean(skip_penaltys_flat, axis=0)  # (batch_size,)

    def _build_train_op(self):
        """build train_op based on either REINFORCE or PPO
        """
        self.advantage = F.placeholder(shape=(None, 1), dtype=F.float32, name="advantage")
        self.reward = F.placeholder(shape=(None, 1), dtype=F.float32, name="reward")

        normalize = F.cast(self.num_layers * (self.num_layers - 1) / 2, F.float32)
        self.skip_rate = F.cast(self.skip_count, F.float32) / normalize

        self.input_arc_onehot = self.convert_arc_to_onehot(self)
        self.old_probs = [F.placeholder(shape=self.onehot_probs[i].shape, dtype=F.float32, name="old_prob_%i" % i) for
                          i in range(len(self.onehot_probs))]
        if self.skip_weight is not None:
            self.loss += self.skip_weight * F.reduce_mean(self.onehot_skip_penaltys)
        if self.use_ppo_loss:
            self.loss += proximal_policy_optimization_loss(
                curr_prediction=self.onehot_probs,
                curr_onehot=self.input_arc_onehot,
                old_prediction=self.old_probs,
                old_onehotpred=self.input_arc_onehot,
                rewards=self.reward,
                advantage=self.advantage,
                clip_val=0.2)
        else:
            self.loss += F.reshape(F.tensordot(self.onehot_log_prob, self.advantage, axes=1), [])

        self.kl_div, self.ent = get_kl_divergence_n_entropy(curr_prediction=self.onehot_probs,
                                                            old_prediction=self.old_probs,
                                                            curr_onehot=self.input_arc_onehot,
                                                            old_onehotpred=self.input_arc_onehot)
        self.train_step = F.Variable(
            0, shape=(), dtype=F.int32, trainable=False, name="train_step")
        tf_variables = [var
                        for var in F.trainable_variables(scope=self.name)]

        self.train_op, self.lr, self.optimizer = F.get_train_op(
            loss=self.loss,
            variables=tf_variables,
            optimizer=self.optim_algo
        )

    def get_action(self, *args, **kwargs):
        """Get a sampled architecture/action and its corresponding probabilities give current controller policy parameters.

        The generated architecture is the out-going information from controller to manager. which in turn will feedback
        the reward signal for storage and training by the controller.

        Parameters
        ----------
        None

        Returns
        ----------
        onehots : list
            The sampled architecture sequence. In particular, the architecture sequence is ordered as::

                [categorical_operation_0,
                categorical_operation_1, binary_skip_0,
                categorical_operation_2, binary_skip_0, binary_skip_1,
                ...]


        probs : list of ndarray
            The probabilities associated with each sampled operation and residual connection. Shapes will vary depending
            on each layer's specification in ModelSpace for operation, and the layer number for residual connections.
        """
        probs, onehots = self.session.run([self.sample_probs, self.sample_arc])
        return onehots, probs

    def train(self, episode, working_dir):
        """Train the controller policy parameters for one step.

        Parameters
        ----------
        episode : int
            Total number of epochs to train the controller. Each epoch will iterate over all architectures stored in buffer.
        working_dir : str
            Filepath to working directory to store (possible) intermediate results

        Returns
        -------
        aloss : float
            Average controller loss for this train step

        Notes
        -----
        Consider renaming this method to ``train_step()`` to better reflect its function, and avoid confusion with the
        training function in environment ``ControllerTrainEnv.train()``
        """
        try:
            self.buffer.finish_path(self.model_space, episode, working_dir)
        except Exception as e:
            print("cannot finish path in buffer because: %s" % e)
            sys.exit(1)
        aloss = 0
        g_t = 0

        for epoch in range(self.train_pi_iter):
            t = 0
            kl_sum = 0
            ent_sum = 0
            # get data from buffer
            for s_batch, p_batch, a_batch, ad_batch, nr_batch in self.buffer.get_data(self.batch_size):
                feed_dict = {self.input_arc[i]: a_batch[:, [i]]
                             for i in range(a_batch.shape[1])}
                feed_dict.update({self.advantage: ad_batch})
                feed_dict.update({self.old_probs[i]: p_batch[i]
                                  for i in range(len(self.old_probs))})
                feed_dict.update({self.reward: nr_batch})

                self.session.run(self.train_op, feed_dict=feed_dict)
                curr_loss, curr_kl, curr_ent = self.session.run([self.loss, self.kl_div, self.ent], feed_dict=feed_dict)
                aloss += curr_loss
                kl_sum += curr_kl
                ent_sum += curr_ent
                t += 1
                g_t += 1

                if kl_sum / t > self.kl_threshold and epoch > 0 and self.verbose > 0:
                    print("     Early stopping at step {} as KL(old || new) = ".format(g_t), kl_sum / t)
                    return aloss / g_t

            if epoch % max(1, (self.train_pi_iter // 5)) == 0 and self.verbose > 0:
                print("     Epoch: {} Actor Loss: {} KL(old || new): {} Entropy(new) = {}".format(
                    epoch, aloss / g_t,
                    kl_sum / t,
                    ent_sum / t)
                )

        return aloss / g_t

    def save_weights(self, filepath, **kwargs):
        """Save current controller weights to a hdf5 file

        Parameters
        ----------
        filepath : str
            file path to save the weights

        Returns
        -------
        None
        """
        weights = self.get_weights()
        with h5py.File(filepath, "w") as hf:
            for i, d in enumerate(weights):
                hf.create_dataset(name=self.weights[i].name, data=d)

    def load_weights(self, filepath, **kwargs):
        """Load the controller weights from a hdf5 file

        Parameters
        ----------
        filepath : str
            file path to saved weights

        Returns
        -------
        None
        """
        weights = []
        with h5py.File(filepath, 'r') as hf:
            for i in range(len(self.weights)):
                key = self.weights[i].name
                weights.append(hf.get(key).value)
        self.set_weights(weights)

    def get_weights(self, **kwargs):
        """Get the current controller weights in a numpy array

        Parameters
        ----------
        None

        Returns
        -------
        weights : list
            A list of numpy array for each weights in controller
        """
        weights = self.session.run(self.weights)
        return weights

    def set_weights(self, weights, **kwargs):
        """Set the current controller weights

        Parameters
        ----------
        weights : list of numpy.ndarray
            A list of numpy array for each weights in controller

        Returns
        -------
        None
        """
        assign_ops = []
        for i in range(len(self.weights)):
            assign_ops.append(F.assign(self.weights[i], weights[i]))
        self.session.run(assign_ops)

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
            # print("layer_type ",arc_pointer)
            onehot_list.append(F.squeeze(F.one_hot(tensor=arc_seq[arc_pointer], num_classes=len(model_space[layer_id])), axis=1))
            if with_input_blocks:
                inp_blocks_idx = arc_pointer + 1, arc_pointer + 1 + num_input_blocks * with_input_blocks
                tmp = []
                for i in range(inp_blocks_idx[0], inp_blocks_idx[1]):
                    # print("input block ",i)
                    tmp.append(F.squeeze(F.one_hot(arc_seq[i], 2), axis=1))
                onehot_list.append(F.transpose(F.stack(tmp), [1, 0, 2]))
            if layer_id > 0 and with_skip_connection:
                skip_con_idx = arc_pointer + 1 + num_input_blocks * with_input_blocks, \
                               arc_pointer + 1 + num_input_blocks * with_input_blocks + layer_id * with_skip_connection
                tmp = []
                for i in range(skip_con_idx[0], skip_con_idx[1]):
                    # print("skip con ",i)
                    tmp.append(F.squeeze(F.one_hot(arc_seq[i], 2), axis=1))
                onehot_list.append(F.transpose(F.stack(tmp), [1, 0, 2]))
            arc_pointer += 1 + num_input_blocks * with_input_blocks + layer_id * with_skip_connection
        return onehot_list
