# -*- coding: UTF-8 -*-

"""```Buffer``` class for holding reinforcement-learning histories"""

import os
import numpy as np


def get_buffer(arg):
    if arg is None:
        return None
    elif type(arg) is str:
        if arg.lower() == 'ordinal':
            return Buffer
        elif arg.lower() == 'replay':
            return ReplayBuffer
        elif arg.lower() == "multimanager":
            return MultiManagerBuffer
    elif callable(arg):
        return arg
    else:
        raise ValueError("Could not understand the Buffer func:", arg)


def parse_action_str(action_onehot, state_space):
    return [state_space[i][int(j)] for i in range(len(state_space)) for j in range(len(action_onehot[i][0])) if
            action_onehot[i][0][int(j)] == 1]


def parse_action_str_squeezed(action_onehot, state_space):
    return [state_space[i][action_onehot[i]] for i in range(len(state_space))]


class Buffer(object):
    def __init__(self, max_size,
                 discount_factor=0.,
                 ewa_beta=0.95,
                 is_squeeze_dim=False,
                 rescale_advantage_by_reward=False,
                 clip_advantage=10.):
        """
        ewa_beta: the average is approx. over the past 1/(1-ewa_beta)
        is_squeeze_dim: if True, controller samples a sequence of tokens, instead of one-hot arrays
        """
        self.max_size = max_size
        self.ewa_beta = ewa_beta
        self.discount_factor = discount_factor
        self.is_squeeze_dim = is_squeeze_dim
        self.rescale_advantage_by_reward = rescale_advantage_by_reward
        self.clip_advantage = clip_advantage
        self.r_bias = None

        # short term buffer storing single trajectory
        self.state_buffer = []
        self.action_buffer = []
        self.prob_buffer = []
        self.reward_buffer = []

        # long_term buffer
        self.lt_sbuffer = []     # state
        self.lt_abuffer = []     # action
        self.lt_pbuffer = []     # prob
        self.lt_adbuffer = []    # advantage
        self.lt_nrbuffer = []    # lt buffer for non discounted reward
        self.lt_rmbuffer = []    # reward mean buffer

    def store(self, state=None, prob=None, action=None, reward=None, *args, **kwargs):
        if state is not None:
            self.state_buffer.append(state)
        if prob is not None:
            self.prob_buffer.append(prob)
        if action is not None:
            self.action_buffer.append(action)
        if reward is not None:
            self.reward_buffer.append(reward)

    def discount_rewards(self):
        """
        Example
        ----------
        behavior of discounted reward, given `reward_buffer=[1,2,3]`:

        >>> if buffer.discount_factor=0.1:
        >>>     ([array([[1.23]]), array([[2.3]]), array([[3.]])], [1, 2, 3])
        >>> if buffer.discount_factor=0.9:
        >>>     ([array([[5.23]]), array([[4.7]]), array([[3.]])], [1, 2, 3])

        """
        discounted_rewards = []
        running_add = 0
        for i, r in enumerate(reversed(self.reward_buffer)):
            # if rewards[t] != 0:   # with this condition, a zero reward_t can get a non-zero
            #    running_add = 0   # discounted reward from its predecessors
            running_add = running_add * self.discount_factor + np.array([r])
            discounted_rewards.append(np.reshape(running_add, (running_add.shape[0], 1)))

        discounted_rewards.reverse()
        return discounted_rewards, self.reward_buffer

    def finish_path(self, state_space, global_ep, working_dir, *args, **kwargs):
        """dump buffers to file and return processed buffer
        return processed [state, ob_prob, ob_onehot, reward, advantage]
        """

        dcreward, reward = self.discount_rewards()
        # advantage = self.get_advantage()

        # get data from buffer
        # TODO: complete remove `state_buffer` as it's useless; ZZ 2020.5.15
        if self.state_buffer:
            state = np.concatenate(self.state_buffer, axis=0)
        old_prob = [np.concatenate(p, axis=0) for p in zip(*self.prob_buffer)]
        if self.is_squeeze_dim:
            action_onehot = np.array(self.action_buffer)
        else:  # action squeezed into sequence
            action_onehot = [np.concatenate(onehot, axis=0) for onehot in zip(*self.action_buffer)]
        r = np.concatenate(dcreward, axis=0)
        if not self.lt_rmbuffer:
            self.r_bias = r.mean()
        else:
            self.r_bias = self.r_bias * self.ewa_beta + (1. - self.ewa_beta) * self.lt_rmbuffer[-1]

        nr = np.array(reward)[:, np.newaxis]

        if self.rescale_advantage_by_reward:
            ad = (r - self.r_bias) / np.abs(self.r_bias)
        else:
            ad = (r - self.r_bias)
        # if ad.shape[1] > 1:
        #   ad = ad / (ad.std() + 1e-8)

        if self.clip_advantage:
            ad = np.clip(ad, -np.abs(self.clip_advantage), np.abs(self.clip_advantage))

        if self.state_buffer:
            self.lt_sbuffer.append(state)
        self.lt_pbuffer.append(old_prob)
        self.lt_abuffer.append(action_onehot)
        # self.lt_rbuffer.append(r)
        self.lt_adbuffer.append(ad)
        self.lt_nrbuffer.append(nr)

        self.lt_rmbuffer.append(r.mean())

        with open(os.path.join(working_dir, 'buffers.txt'), mode='a+') as f:
            action_onehot = self.action_buffer[-1]
            if self.is_squeeze_dim:
                action_readable_str = ','.join([str(x) for x in parse_action_str_squeezed(action_onehot, state_space)])
            else:
                action_readable_str = ','.join([str(x) for x in parse_action_str(action_onehot, state_space)])
            f.write(
                "-" * 80 + "\n" +
                "Episode:%d\tReward:%.4f\tR_bias:%.4f\tAdvantage:%s\n" %
                (
                    global_ep,
                    self.reward_buffer[-1],
                    self.r_bias,
                    np.round(self.lt_adbuffer[-1].flatten(), 2),
                ) +
                "\tAction:%s\n\tProb:%s\n" % (
                    action_readable_str,
                    ' || '.join([str(np.round(x, 2)).replace("\n ", ";") for x in self.prob_buffer[-1]])
                )
            )
            print("Saved buffers to file `buffers.txt` !")

        if len(self.lt_pbuffer) > self.max_size:
            self.lt_sbuffer = self.lt_sbuffer[-self.max_size:]
            self.lt_pbuffer = self.lt_pbuffer[-self.max_size:]
            # self.lt_rbuffer = self.lt_rbuffer[-self.max_size:]
            self.lt_adbuffer = self.lt_adbuffer[-self.max_size:]
            self.lt_abuffer = self.lt_abuffer[-self.max_size:]
            self.lt_nrbuffer = self.lt_nrbuffer[-self.max_size:]
            self.lt_rmbuffer = self.lt_rmbuffer[-self.max_size:]

        self.state_buffer, self.prob_buffer, self.action_buffer, self.reward_buffer = [], [], [], []

    def get_data(self, bs, shuffle=True):
        """
        get a batched data
        size of buffer: (traj, traj_len, data_shape)
        """

        lt_sbuffer, lt_pbuffer, lt_abuffer, lt_adbuffer, lt_nrbuffer = self.lt_sbuffer, self.lt_pbuffer, \
                                                                       self.lt_abuffer, self.lt_adbuffer, self.lt_nrbuffer

        lt_sbuffer = np.concatenate(lt_sbuffer, axis=0)
        lt_pbuffer = [np.concatenate(p, axis=0) for p in zip(*lt_pbuffer)]
        if self.is_squeeze_dim:
            lt_abuffer = np.concatenate(lt_abuffer, axis=0)
        else:
            lt_abuffer = [np.concatenate(a, axis=0) for a in zip(*lt_abuffer)]
        lt_adbuffer = np.concatenate(lt_adbuffer, axis=0)
        lt_nrbuffer = np.concatenate(lt_nrbuffer, axis=0)

        if shuffle:
            slice_ = np.random.choice(lt_sbuffer.shape[0], size=lt_sbuffer.shape[0], replace=False)
            lt_sbuffer = lt_sbuffer[slice_]
            lt_pbuffer = [p[slice_] for p in lt_pbuffer]
            if self.is_squeeze_dim:
                lt_abuffer = lt_abuffer[slice_]
            else:
                lt_abuffer = [a[slice_] for a in lt_abuffer]
            lt_adbuffer = lt_adbuffer[slice_]
            lt_nrbuffer = lt_nrbuffer[slice_]

        for i in range(0, len(lt_sbuffer), bs):
            b = min(i + bs, len(lt_sbuffer))
            p_batch = [p[i:b, :] for p in lt_pbuffer]
            if self.is_squeeze_dim:
                a_batch = lt_abuffer[i:b]
            else:
                a_batch = [a[i:b, :] for a in lt_abuffer]
            yield lt_sbuffer[i:b, :, :], p_batch, a_batch, lt_adbuffer[i:b, :], lt_nrbuffer[i:b, :]


class ReplayBuffer(Buffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_advantage(self):
        r = np.concatenate(self.lt_nrbuffer, axis=0)
        if self.rescale_advantage_by_reward:
            ad = (r - self.r_bias) / np.abs(r)
        else:
            ad = (r - self.r_bias)

        if self.clip_advantage:
            ad = np.clip(ad, -np.abs(self.clip_advantage), np.abs(self.clip_advantage))
        return ad

    def get_data(self, bs, shuffle=True):
        lt_sbuffer, lt_pbuffer, lt_abuffer, lt_nrbuffer = self.lt_sbuffer, self.lt_pbuffer, \
                                                          self.lt_abuffer, self.lt_nrbuffer
        lt_adbuffer = self.get_advantage()
        lt_sbuffer = np.concatenate(lt_sbuffer, axis=0)
        lt_pbuffer = [np.concatenate(p, axis=0) for p in zip(*lt_pbuffer)]
        if self.is_squeeze_dim:
            lt_abuffer = np.concatenate(lt_abuffer, axis=0)
        else:
            lt_abuffer = [np.concatenate(a, axis=0) for a in zip(*lt_abuffer)]
        lt_nrbuffer = np.concatenate(lt_nrbuffer, axis=0)

        if shuffle:
            slice_ = np.random.choice(lt_sbuffer.shape[0], size=lt_sbuffer.shape[0], replace=False)
            lt_sbuffer = lt_sbuffer[slice_]
            lt_pbuffer = [p[slice_] for p in lt_pbuffer]
            if self.is_squeeze_dim:
                lt_abuffer = lt_abuffer[slice_]
            else:
                lt_abuffer = [a[slice_] for a in lt_abuffer]
            lt_adbuffer = lt_adbuffer[slice_]
            lt_nrbuffer = lt_nrbuffer[slice_]

        for i in range(0, len(lt_sbuffer), bs):
            b = min(i + bs, len(lt_sbuffer))
            p_batch = [p[i:b, :] for p in lt_pbuffer]
            if self.is_squeeze_dim:
                a_batch = lt_abuffer[i:b]
            else:
                a_batch = [a[i:b, :] for a in lt_abuffer]
            s_batch = lt_sbuffer[i:b, :, :]
            nr_batch = lt_nrbuffer[i:b, :]
            ad_batch = lt_adbuffer[i:b, :]
            yield s_batch, p_batch, a_batch, ad_batch, nr_batch


class MultiManagerBuffer:
    def __init__(self, max_size,
                 ewa_beta=0.95,
                 is_squeeze_dim=False,
                 rescale_advantage_by_reward=False,
                 clip_advantage=10.,
                 **kwargs
                 ):
        self.max_size = max_size
        self.ewa_beta = ewa_beta
        self.is_squeeze_dim = is_squeeze_dim
        self.rescale_advantage_by_reward = rescale_advantage_by_reward
        self.clip_advantage = clip_advantage
        self.r_bias = None

        # short term buffer storing single trajectory
        self.action_buffer = []
        self.prob_buffer = []
        self.reward_buffer = []

        # long_term buffer
        self.lt_action = []     # action
        self.lt_prob = []     # prob
        self.lt_adv = []    # advantage
        self.lt_reward = []    # lt buffer for non discounted reward
        self.lt_reward_mean = []    # reward mean buffer
        # unique to this class
        self.description_buffer = []   # short-term descriptive feature buffer
        self.lt_desc_buffer = []       # long-term descriptive feature buffer

    def store(self, prob, action, reward, description):
        self.prob_buffer.append(prob)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.description_buffer.append(description)

    def reset_short_term(self):
        self.prob_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.description_buffer = []

    def dump_buffer(self, model_space, global_ep, working_dir):
        with open(os.path.join(working_dir, 'buffers.txt'), mode='a+') as f:
            action_onehot = self.action_buffer[-1]
            if self.is_squeeze_dim:
                action_readable_str = ','.join([str(x) for x in parse_action_str_squeezed(action_onehot, model_space)])
            else:
                action_readable_str = ','.join([str(x) for x in parse_action_str(action_onehot, model_space)])
            f.write(
                "-" * 80 + "\n" +
                "Episode:%d\tReward:%.4f\tR_bias:%.4f\tAdvantage:%s\n" %
                (
                    global_ep,
                    self.reward_buffer[-1],
                    self.r_bias,
                    np.round(self.lt_adv[-1].flatten(), 2),
                ) +
                "\tAction:%s\n\tProb:%s\n" % (
                    action_readable_str,
                    ' || '.join([str(np.round(x, 2)).replace("\n ", ";") for x in self.prob_buffer[-1]])
                )
            )
            print("Saved buffers to file `buffers.txt` !")

    def finish_path(self, model_space, global_ep, working_dir, *args, **kwargs):
        old_prob = [np.concatenate(p, axis=0) for p in zip(*self.prob_buffer)]
        if self.is_squeeze_dim:
            action_onehot = np.array(self.action_buffer)
        else:  # action squeezed into sequence
            action_onehot = [np.concatenate(onehot, axis=0) for onehot in zip(*self.action_buffer)]

        reward = np.array(self.reward_buffer)
        if not self.lt_reward_mean:
            self.r_bias = reward.mean()
        else:
            self.r_bias = self.r_bias * self.ewa_beta + (1. - self.ewa_beta) * self.lt_reward_mean[-1]

        if self.rescale_advantage_by_reward:
            ad = (reward - self.r_bias) / np.abs(self.r_bias)
        else:
            ad = (reward - self.r_bias)

        if self.clip_advantage:
            ad = np.clip(ad, -np.abs(self.clip_advantage), np.abs(self.clip_advantage))

        self.lt_prob.append(old_prob)
        self.lt_action.append(action_onehot)
        self.lt_adv.append(ad)
        self.lt_reward.append(reward)
        self.lt_reward_mean.append(reward.mean())

        self.dump_buffer(model_space=model_space, global_ep=global_ep, working_dir=working_dir)

        # NOTE: this will only keep the `max_size` number of short-term buffers;
        # whereas each short-term buffer could have multiple samples
        if len(self.lt_prob) > self.max_size:
            self.lt_prob = self.lt_prob[-self.max_size:]
            self.lt_adv = self.lt_adv[-self.max_size:]
            self.lt_action = self.lt_action[-self.max_size:]
            self.lt_reward = self.lt_reward[-self.max_size:]
            self.lt_reward_mean = self.lt_reward_mean[-self.max_size:]

        description = np.concatenate(self.description_buffer, axis=0)
        self.lt_desc_buffer.append(description)
        if len(self.lt_desc_buffer) > self.max_size:
            self.lt_desc_buffer = self.lt_desc_buffer[-self.max_size:]
        self.reset_short_term()

    def get_data(self, bs, shuffle=True):
        lt_prob, lt_action, lt_adv, lt_reward = self.lt_prob, self.lt_action, self.lt_adv, self.lt_reward
        lt_desc = self.lt_desc_buffer

        lt_prob = [np.concatenate(p, axis=0) for p in zip(*lt_prob)]
        if self.is_squeeze_dim:
            lt_action = np.concatenate(lt_action, axis=0)
        else:
            lt_action = [np.concatenate(a, axis=0) for a in zip(*lt_action)]
        lt_adv = np.concatenate(lt_adv, axis=0)
        lt_reward = np.concatenate(lt_reward, axis=0)
        lt_desc = np.concatenate(lt_desc, axis=0)

        if shuffle:
            slice_ = np.random.choice(lt_adv.shape[0], size=lt_adv.shape[0], replace=False)
            lt_prob = [p[slice_] for p in lt_prob]
            if self.is_squeeze_dim:
                lt_action = lt_action[slice_]
            else:
                lt_action = [a[slice_] for a in lt_action]
            lt_adv = lt_adv[slice_]
            lt_reward = lt_reward[slice_]
            lt_desc = lt_desc[slice_]

        for i in range(0, len(lt_prob), bs):
            b = min(i + bs, len(lt_prob))
            p_batch = [p[i:b, :] for p in lt_prob]
            if self.is_squeeze_dim:
                a_batch = lt_action[i:b]
            else:
                a_batch = [a[i:b, :] for a in lt_action]
            batch_data = {
                "prob": p_batch,
                "action": a_batch,
                "advantage": np.expand_dims(lt_adv[i:b], axis=-1),
                "reward": np.expand_dims(lt_reward[i:b], axis=-1),
                "description": lt_desc[i:b]
            }
            yield batch_data

