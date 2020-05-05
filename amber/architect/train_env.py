# -*- coding: UTF-8 -*-

import csv
import datetime
import logging
import shutil
import warnings

from keras.optimizers import *

from .buffer import parse_action_str, parse_action_str_squeezed
from ._operation_controller import *
from ..utils.io import save_action_weights, save_stats
from ..plots import plot_stats2, plot_environment_entropy, plot_controller_performance, \
    plot_action_weights, plot_wiring_weights


def setup_logger(working_dir='.', verbose_level=logging.INFO):
    # setup logger
    logger = logging.getLogger('AMBER')
    logger.setLevel(verbose_level)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(working_dir, 'log.AMBER.txt'))
    fh.setLevel(verbose_level)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(verbose_level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -\n %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_controller_states(model):
    return [K.get_value(s) for s, _ in model.state_updates]


def set_controller_states(model, states):
    for (d, _), s in zip(model.state_updates, states):
        K.set_value(d, s)


def get_controller_history(fn='train_history.csv'):
    with open(fn, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            trial = row[0]
    return int(trial)


def compute_entropy(prob_states):
    ent = 0
    for prob in prob_states:
        for p in prob:
            p = np.array(p).flatten()
            i = np.where(p > 0)[0]
            t = np.sum(-p[i] * np.log2(p[i]))
            ent += t
        # ent += np.sum([-p * np.log2(p) if p>0 else 0 for p in prob])
    return ent


class ControllerTrainEnvironment:
    """ControllerNetEnvironment: employs `controller` model and `manager` to mange data and reward,
    creates a reinforcement learning environment
    """

    def __init__(self,
                 controller,
                 manager,
                 max_episode=100,
                 max_step_per_ep=2,
                 logger=None,
                 resume_prev_run=False,
                 should_plot=True,
                 initial_buffering_queue=15,
                 working_dir='.', entropy_converge_epsilon=0.01,
                 squeezed_action=True,
                 with_input_blocks=True,
                 with_skip_connection=True,
                 save_controller=False,
                 continuous_run=False,
                 verbose=0,
                 **kwargs):
        """ Note if either `with_input_blocks` or `with_skip_connection`, a list of integers will be used to represent the
            sequential model architecture and wiring, instead of a list of BioNAS.Controller.States
        """
        self.controller = controller
        self.manager = manager
        self.max_episode = max_episode
        self.max_step_per_ep = max_step_per_ep
        self.start_ep = 0
        self.should_plot = should_plot
        self.working_dir = working_dir
        self.total_reward = 0
        self.entropy_record = []
        self.entropy_converge_epsilon = entropy_converge_epsilon
        self.squeezed_action = squeezed_action
        self.with_input_blocks = with_input_blocks
        self.with_skip_connection = with_skip_connection
        self.save_controller = save_controller
        self.initial_buffering_queue = min(initial_buffering_queue, controller.buffer.max_size)
        self.continuous_run = continuous_run

        # FOR DEPRECATED USE
        try:
            self.last_actionState_size = len(self.controller.state_space[-1])
        except Exception as e:
            warnings.warn("DEPRECATED Exception in ControllerTrainEnv: %s" % e)
            self.last_actionState_size = 1

        if resume_prev_run:
            self.restore()
        else:
            self.clean()
        self.resume_prev_run = resume_prev_run
        self.logger = logger if logger else setup_logger(working_dir)
        if os.path.realpath(manager.working_dir) != os.path.realpath(self.working_dir):
            warnings.warn("manager working dir and environment working dir are different.")

    def __str__(self):
        s = 'ControllerTrainEnv for %i max steps, %i child mod. each step' % (self.max_episode, self.max_step_per_ep)
        return s

    def restore(self):
        if self.save_controller:
            controller_states = np.load(os.path.join(self.working_dir, 'controller_states.npy'))
            set_controller_states(self.controller.model, controller_states)
            self.controller.model.load_weights(os.path.join(self.working_dir, 'controller_weights.h5'))
            self.start_ep = get_controller_history(os.path.join(self.working_dir, 'train_history.csv'))
        else:
            raise Exception("Did not turn on option `save_controller`")

    def clean(self):
        bak_weights_dir = os.path.join(self.working_dir,
                                       'weights_bak_%s' % datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if os.path.isdir(os.path.join(self.working_dir, 'weights')):
            shutil.move(os.path.join(self.working_dir, 'weights'), bak_weights_dir)

        movable_files = [
            'buffers.txt',
            'log.controller.txt',
            'train_history.csv',
            'train_history.png',
            'entropy.png',
            'nas_training_stats.png',
            'controller_states.npy',
            'controller_weights.h5',
            'controller_hidden_states.png'
        ]
        if not self.continuous_run:
            movable_files += ['nas_training_stats.json', 'weight_data.json']
        movable_files += [x for x in os.listdir(self.working_dir) if
                          x.startswith("weight_at_layer_") and x.endswith(".png")]
        movable_files += [x for x in os.listdir(self.working_dir) if
                          x.startswith("inp_at_layer_") and x.endswith(".png")]
        movable_files += [x for x in os.listdir(self.working_dir) if
                          x.startswith("skip_at_layer_") and x.endswith(".png")]
        for file in movable_files:
            file = os.path.join(self.working_dir, file)
            if os.path.exists(file):
                shutil.move(file, bak_weights_dir)
        os.makedirs(os.path.join(self.working_dir, 'weights'))
        self.controller.remove_files(movable_files, self.working_dir)

    def reset(self):
        # x = np.random.normal(size=(1, 1, self.last_actionState_size))
        # return x
        x = np.random.uniform(0, 5, (1, 1, self.last_actionState_size))
        x = np.exp(x) / np.sum(np.exp(x))
        return x

    def step(self, action_prob):
        # returns a state given an action (prob list)
        # fix discrepancy between operation_controller and general_controller. 20190912 ZZ
        try:
            next_state = np.array(action_prob[-1]).reshape(1, 1, self.last_actionState_size)
        except ValueError:
            next_state = self.reset()
        return next_state

    def train(self):
        """Performs training for controller
        """
        LOGGER = self.logger

        action_probs_record = []

        loss_and_metrics_list = []
        global_step = self.start_ep * self.max_step_per_ep
        if self.resume_prev_run:
            f = open(os.path.join(self.working_dir, 'train_history.csv'), mode='a+')
        else:
            f = open(os.path.join(self.working_dir, 'train_history.csv'), mode='w')
        writer = csv.writer(f)
        for ep in range(self.start_ep, self.max_episode):
            try:
                # reset env
                state = self.reset()
                ep_reward = 0
                loss_and_metrics_ep = {'knowledge': 0, 'acc': 0, 'loss': 0}
                if 'metrics' in self.manager.model_compile_dict:
                    loss_and_metrics_ep.update({x: 0 for x in self.manager.model_compile_dict['metrics']})

                ep_probs = []

                for step in range(self.max_step_per_ep):
                    # value = self.controller.get_value(state)
                    actions, probs = self.controller.get_action(state)  # get an action for the previous state
                    self.entropy_record.append(compute_entropy(probs))
                    next_state = self.step(probs)
                    # next_value = self.controller.get_value(next_state)
                    ep_probs.append(probs)
                    # LOGGER.debug the action probabilities
                    if self.squeezed_action:
                        action_list = parse_action_str_squeezed(actions, self.controller.state_space)
                    else:
                        action_list = parse_action_str(actions, self.controller.state_space)
                    LOGGER.debug("Predicted actions : {}".format([str(x) for x in action_list]))

                    # build a model, train and get reward and accuracy from the network manager
                    if self.with_input_blocks or self.with_skip_connection:
                        # if use model arc, parse the raw ops/connection indices
                        reward, loss_and_metrics = self.manager.get_rewards(
                            global_step, actions)
                    else:
                        # otherwise, parse the BioNAS.Controller.State class
                        reward, loss_and_metrics = self.manager.get_rewards(
                            global_step, action_list)
                        # global_step, actions)
                    LOGGER.debug("Rewards : " + str(reward) + " Metrics : " + str(loss_and_metrics))

                    ep_reward += reward
                    for x in loss_and_metrics.keys():
                        loss_and_metrics_ep[x] += loss_and_metrics[x]

                    # actions and states are equivalent, save the state and reward
                    self.controller.store(state, probs, actions, reward)

                    # write the results of this trial into a file
                    data = [global_step, [loss_and_metrics[x] for x in sorted(loss_and_metrics.keys())],
                            reward]
                    if self.squeezed_action:
                        data.extend(actions)
                    else:
                        data.extend(action_list)
                    writer.writerow(data)
                    f.flush()

                    # update trial
                    global_step += 1
                    state = next_state

                loss_and_metrics_list.append({x: (v / self.max_step_per_ep) for x, v in loss_and_metrics_ep.items()})
                # average probs over trajectory
                ep_p = [sum(p) / len(p) for p in zip(*ep_probs)]
                action_probs_record.append(ep_p)
                if ep >= self.initial_buffering_queue - 1:
                    # train the controller on the saved state and the discounted rewards
                    loss = self.controller.train(ep, self.working_dir)
                    self.total_reward += np.sum(np.array(self.controller.buffer.lt_adbuffer[-1]).flatten())
                    LOGGER.debug("Total reward : " + str(self.total_reward))
                    LOGGER.debug("END episode %d: Controller loss : %0.6f" % (ep, loss))
                    LOGGER.debug("-" * 10)
                else:
                    LOGGER.debug("END episode %d: Buffering" % (ep))
                    LOGGER.debug("-" * 10)
                    # self.controller.buffer.finish_path(self.controller.state_space, ep, self.working_dir)

                # save the controller states and weights
                if self.save_controller:
                    np.save(os.path.join(self.working_dir, 'controller_states.npy'),
                            get_controller_states(self.controller.model))
                    self.controller.model.save_weights(os.path.join(self.working_dir, 'controller_weights.h5'))

                # TODO: add early-stopping
                # check the entropy record and stop training if no progress was made
                # (less than entropy_converge_epsilon)
                # if ep >= self.max_episode//3 and \
                # np.std(self.entropy_record[-(self.max_step_per_ep):])<self.entropy_converge_epsilon:
                #    LOGGER.info("Controller converged at episode %i"%ep)
                #    break
            except KeyboardInterrupt:
                LOGGER.info("User disrupted training")
                break

        LOGGER.debug("Total Reward : %s" % self.total_reward)

        f.close()
        plot_controller_performance(os.path.join(self.working_dir, 'train_history.csv'),
                                    metrics_dict={k: v for k, v in
                                                  zip(sorted(loss_and_metrics.keys()), range(len(loss_and_metrics)))},
                                    save_fn=os.path.join(self.working_dir, 'train_history.png'), N_sma=10)
        plot_environment_entropy(self.entropy_record,
                                 os.path.join(self.working_dir, 'entropy.png'))
        save_kwargs = {}
        if self.with_input_blocks:
            save_kwargs['input_nodes'] = self.manager.model_fn.inputs_op
        save_action_weights(action_probs_record, self.controller.state_space, self.working_dir,
                            with_input_blocks=self.with_input_blocks, with_skip_connection=self.with_skip_connection,
                            **save_kwargs)
        save_stats(loss_and_metrics_list, self.working_dir)

        if self.should_plot:
            plot_action_weights(self.working_dir)
            plot_wiring_weights(self.working_dir, self.with_input_blocks, self.with_skip_connection)
            plot_stats2(self.working_dir)

        # return converged config idx
        act_idx = []
        for p in ep_p:
            act_idx.append(np.argmax(p))
        return act_idx


class EnasTrainEnv(ControllerTrainEnvironment):
    """
    Params:
        time_budget: defaults to 72 hours
    """

    def __init__(self, *args, **kwargs):
        self.time_budget = kwargs.pop('time_budget', "72:00:00")
        self.child_train_steps = kwargs.pop('child_train_steps', None)
        self.child_warm_up_epochs = kwargs.pop('child_warm_up_epochs', 0)
        self.save_controller_every = kwargs.pop('save_controller_every', None)
        super().__init__(*args, **kwargs)
        self.initial_buffering_queue = 0
        if self.manager.model_fn.controller is None:
            self.manager.model_fn.set_controller(self.controller)
        if self.time_budget is None:
            pass
        elif type(self.time_budget) is str:
            print("time budget set to: %s" % self.time_budget)
            self.time_budget = sum(x * int(t) for x, t in zip([3600, 60, 1], self.time_budget.split(":")))
        else:
            raise Exception("time budget should be in format HH:mm:ss; cannot understand : %s" % (self.time_budget))
        self.action_probs_record = None

    def train(self):
        LOGGER = self.logger

        action_probs_record = []
        loss_and_metrics_list = []
        state = self.reset()  # nuisance param
        controller_step = self.start_ep * self.max_step_per_ep
        if self.resume_prev_run:
            f = open(os.path.join(self.working_dir, 'train_history.csv'), mode='a+')
        else:
            f = open(os.path.join(self.working_dir, 'train_history.csv'), mode='w')
        writer = csv.writer(f)
        starttime = datetime.datetime.now()
        if self.child_warm_up_epochs > 0:
            LOGGER.info("warm-up for child model: %i epochs" % self.child_warm_up_epochs)
            # warmup_nsteps = self.child_train_steps*self.child_warm_up_epochs if self.child_train_steps else None
            warmup_nsteps = None
            for i in range(1, self.child_warm_up_epochs + 1):
                LOGGER.info("warm-up : %i epoch" % i)
                self.manager.get_rewards(trial=-i, model_arc=None, nsteps=warmup_nsteps)
        for child_step in range(self.start_ep, self.max_episode):
            try:
                ep_reward = 0
                loss_and_metrics_ep = {'knowledge': 0, 'acc': 0, 'loss': 0}
                if 'metrics' in self.manager.model_compile_dict:
                    loss_and_metrics_ep.update({x: 0 for x in self.manager.model_compile_dict['metrics']})

                ep_probs = []

                # train child parameters w
                self.manager.get_rewards(child_step, None, nsteps=self.child_train_steps)

                # train controller parameters theta
                for step in range(self.max_step_per_ep):
                    arc_seq, probs = self.controller.get_action()
                    self.entropy_record.append(compute_entropy(probs))
                    ep_probs.append(probs)
                    # LOGGER.debug the action probabilities
                    action_list = parse_action_str_squeezed(arc_seq, self.controller.state_space)
                    LOGGER.debug("Predicted actions : {}".format([str(x) for x in action_list]))

                    # build a model, train and get reward and accuracy from the network manager
                    reward, loss_and_metrics = self.manager.get_rewards(
                        controller_step, arc_seq, nsteps=self.child_train_steps)
                    LOGGER.debug("Rewards : " + str(reward) + " Metrics : " + str(loss_and_metrics))

                    ep_reward += reward
                    for x in loss_and_metrics.keys():
                        loss_and_metrics_ep[x] += loss_and_metrics[x]

                    # save the arc_seq and reward
                    self.controller.store(state, probs, arc_seq, reward)

                    # write the results of this trial into a file
                    data = [controller_step, [loss_and_metrics[x] for x in sorted(loss_and_metrics.keys())],
                            reward]
                    if self.squeezed_action:
                        data.extend(arc_seq)
                    else:
                        data.extend(action_list)
                    writer.writerow(data)
                    f.flush()

                    # update trial
                    controller_step += 1

                loss_and_metrics_list.append({x: (v / self.max_step_per_ep) for x, v in loss_and_metrics_ep.items()})
                # average probs over trajectory
                ep_p = [sum(p) / len(p) for p in zip(*ep_probs)]
                action_probs_record.append(ep_p)
                if child_step >= self.initial_buffering_queue - 1:
                    # train the controller on the saved state and the discounted rewards
                    loss = self.controller.train(child_step, self.working_dir)
                    self.total_reward += np.sum(np.array(self.controller.buffer.lt_adbuffer[-1]).flatten())
                    LOGGER.info("Total reward : " + str(self.total_reward))
                    LOGGER.info("END episode %d: Controller loss : %0.6f" % (child_step, loss))
                    LOGGER.info("-" * 10)
                else:
                    LOGGER.info("END episode %d: Buffering" % (child_step))
                    LOGGER.info("-" * 10)

                if self.save_controller_every is not None and child_step % self.save_controller_every == 0:
                    self.controller.save_weights(
                        os.path.join(self.working_dir, "controller_weights-epoch-%i.h5" % child_step))

            except KeyboardInterrupt:
                LOGGER.info("User disrupted training")
                break
            consumed_time = (datetime.datetime.now() - starttime).total_seconds()
            LOGGER.info("used time: %.2f %%" % (consumed_time / self.time_budget * 100))
            if consumed_time >= self.time_budget:
                LOGGER.info("training ceased because run out of time budget")
                break

        LOGGER.debug("Total Reward : %s" % self.total_reward)

        f.close()
        plot_controller_performance(os.path.join(self.working_dir, 'train_history.csv'),
                                    metrics_dict={k: v for k, v in
                                                  zip(sorted(loss_and_metrics.keys()), range(len(loss_and_metrics)))},
                                    save_fn=os.path.join(self.working_dir, 'train_history.png'), N_sma=10)
        plot_environment_entropy(self.entropy_record,
                                 os.path.join(self.working_dir, 'entropy.png'))
        save_kwargs = {}
        if self.with_input_blocks:
            save_kwargs['input_nodes'] = self.manager.model_fn.inputs_op
        self.action_probs_record = action_probs_record
        save_action_weights(action_probs_record, self.controller.state_space, self.working_dir,
                            with_input_blocks=self.with_input_blocks, with_skip_connection=self.with_skip_connection,
                            **save_kwargs)
        self.action_probs_record = loss_and_metrics_list
        save_stats(loss_and_metrics_list, self.working_dir)

        if self.should_plot:
            plot_action_weights(self.working_dir)
            plot_wiring_weights(self.working_dir, self.with_input_blocks, self.with_skip_connection)
            plot_stats2(self.working_dir)

        # return converged config idx
        act_idx = []
        for p in ep_p:
            act_idx.append(np.argmax(p))
        return act_idx
