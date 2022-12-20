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
from ..base import BaseController, stack_lstm, proximal_policy_optimization_loss
from amber.architect.buffer import get_buffer
from amber.architect.commonOps import get_kl_divergence_n_entropy


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
        super().__init__(model_space=model_space, **kwargs)

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
        arc_seq, probs_, log_probs, hidden_states, entropys, skip_count, skip_penaltys = self.forward(input_arc=None)
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
        self.sample_probs = probs_

    def _build_trainer(self):
        ops_each_layer = 1
        total_arc_len = sum(
            [ops_each_layer] +   # first layer
            [ops_each_layer + i * self.with_skip_connection for i in range(1, self.num_layers)]  # rest layers
        )
        self.total_arc_len = total_arc_len
        self.input_arc = [F.placeholder(shape=(None, 1), dtype=F.int32, name='arc_{}'.format(i))
                          for i in range(total_arc_len)]
        arc_seq, probs_, log_probs, hidden_states, entropys, skip_count, skip_penaltys = self.forward(input_arc=self.input_arc)
        self.train_hidden_states = hidden_states

        # for class attributes
        self.entropys = F.stack(entropys)
        self.onehot_probs = probs_
        log_probs = F.stack(log_probs)
        self.onehot_log_prob = F.reshape(F.reduce_sum(log_probs, axis=0), [-1]) # (batch_size,)
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
            # onehot_log_prob: 1xbatch_size; advantage: batch_sizex1
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

    def train(self):
        """Train the controller policy parameters for one step.

        Returns
        -------
        aloss : float
            Average controller loss for this train step

        Notes
        -----
        Consider renaming this method to ``train_step()`` to better reflect its function, and avoid confusion with the
        training function in environment ``ControllerTrainEnv.train()``
        """
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
