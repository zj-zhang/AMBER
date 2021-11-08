"""
pmbga = Probabilistic Model-Building Genetic Algorithms

From Wiki:
Estimation of distribution algorithms (EDAs), sometimes called probabilistic model-building genetic algorithms (
PMBGAs),[1] are stochastic optimization methods that guide the search for the optimum by building and sampling
explicit probabilistic models of promising candidate solutions. Optimization is viewed as a series of incremental
updates of a probabilistic model, starting with the model encoding an uninformative prior over admissible solutions
and ending with the model that generates only the global optima.
"""

import os
import sys
import numpy as np
import scipy.stats as ss
import tensorflow as tf
from collections import defaultdict
from .buffer import Buffer
from .modelSpace import ModelSpace, Operation
from ..modeler.enasModeler import ModelBuilder

# sort out more deps
#from bpm_ml_blending.gen_kinetic_weight_mats import KineticModel
import yaml


class PopulationBuffer(Buffer):
    """Population buffer for working with genetic algorithms
    TODO: move to amber.architect.buffer
    """
    def finish_path(self, state_space, global_ep, working_dir, *args, **kwargs):
        # sort acts by reward
        act_reward_pairs = [(a, r) for a,r in zip(*[self.action_buffer, self.reward_buffer])]
        act_reward_pairs = sorted(act_reward_pairs, key=lambda x: x[1], reverse=True)
        # append to long-term
        if self.lt_abuffer is None:
            self.lt_abuffer = [[x[0] for x in act_reward_pairs]]
            self.lt_nrbuffer = [[x[1] for x in act_reward_pairs]]
        else:
            self.lt_abuffer.append([x[0] for x in act_reward_pairs])
            self.lt_nrbuffer.append([x[1] for x in act_reward_pairs])
        # update r-bias
        if self.r_bias is None:
            self.r_bias = np.mean(self.lt_nrbuffer[-1])
        else:
            self.r_bias = self.r_bias * self.ewa_beta + (1. - self.ewa_beta) * np.mean(self.lt_nrbuffer[-1])
        # write out
        with open(os.path.join(working_dir, 'buffers.txt'), mode='a+') as f:
            action_readable_str = ','.join([str(x) for x in self.action_buffer[-1]])
            f.write(
                "-" * 80 + "\n" +
                "Episode:%d\tReward:%.4f\tR_bias:%.4f\tAdvantage:NA\n" %
                (
                    global_ep,
                    self.reward_buffer[-1],
                    self.r_bias,
                ) +
                "\tAction:%s\n\tProb:NA\n" % (
                    action_readable_str,
                )
            )
        # remove tailing buffer to max_size
        if len(self.lt_abuffer) > self.max_size:
            self.lt_abuffer = self.lt_abuffer[-self.max_size:]
            self.lt_nrbuffer = self.lt_nrbuffer[-self.max_size:]
        self.action_buffer, self.reward_buffer = [], []

    def get_data(self, bs, shuffle=True):
        lt_abuffer = np.concatenate(self.lt_abuffer, axis=0)
        lt_nrbuffer = np.concatenate(self.lt_nrbuffer, axis=0)
        lt_abuffer = lt_abuffer[lt_nrbuffer>=self.r_bias]
        lt_nrbuffer = lt_nrbuffer[lt_nrbuffer>=self.r_bias]
        if shuffle is True:
            slice_ = np.random.choice(lt_nrbuffer.shape[0], size=lt_nrbuffer.shape[0], replace=False)
            lt_abuffer = lt_abuffer[slice_]
            lt_nrbuffer = lt_nrbuffer[slice_]

        for i in range(0, len(lt_nrbuffer), bs):
            b = min(i + bs, len(lt_nrbuffer))
            p_batch = None
            a_batch = lt_abuffer[i:b]
            yield None, p_batch, a_batch, None, lt_nrbuffer[i:b]


class BayesProb:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, data, **kwargs):
        """
        data : np.array-like
        """
        return self

    def sample(self, size=1):
        pass


class EmpiricalGaussianKDE(BayesProb):
    def __init__(self, integerize=False, **kwargs):
        super().__init__(**kwargs)
        self.integerize = integerize
        self.data = []
        self.kernel = None

    def update(self, data, reset=True):
        if reset is True:
            self.reset()
        self.data.append(data)
        self.kernel = ss.gaussian_kde(self.data)

    def reset(self):
        self.data = []
        self.kernel = None

    def sample(self, size=1):
        a = self.kernel.resample(size=size)
        if self.integerize: a = int(a)
        if size == 1:
            return a[0]
        else:
            return a


class Binomial(BayesProb):
    def __init__(self, alpha, beta, n=1, **kwargs):
        super().__init__(alpha=alpha, beta=beta, **kwargs)
        self.x = 0
        self.n = n

    def update(self, data):
        assert all(self.n >= np.asarray(data))
        self.x = np.asarray(data)
        return self

    @property
    def post_a(self):
        return self.alpha + self.x.sum()

    @property
    def post_b(self):
        return self.beta + len(self.x)*self.n - self.x.sum()

    def sample(self, size=1):
        a = ss.betabinom.rvs(n=self.n, a=self.post_a, b=self.post_b, size=size)
        if size == 1:
            return a[0]
        else:
            return a


class TruncatedIntegerNormal(BayesProb):
    """TODO: this needs some better theoretic grounds; right now it seems off"""
    def __init__(self, mu_0, sigma2_0, **kwargs):
        super().__init__(mu_0=mu_0, sigma2_0=sigma2_0, **kwargs)
        self.mu = 0
        self.n = 0
        self.sigma2 = 1

    def update(self, data):
        self.mu = np.mean(data)
        self.n = len(data)
        self.sigma2 = np.var(data)
        return self

    @property
    def post_loc(self):
        return 1 / (1 / self.sigma2_0 + self.n / self.sigma2) * (
                    self.mu_0 / self.sigma2_0 + self.mu * self.n / self.sigma2)

    @property
    def post_scale(self):
        return np.sqrt(1 / (1/self.sigma2_0 + self.n/self.sigma2))

    def sample(self, n=1, size=1):
        return int(ss.norm.rvs(loc=self.post_loc, scale=self.post_scale, size=size))


class Poisson(BayesProb):
    def __init__(self, alpha, beta, **kwargs):
        super().__init__(alpha=alpha, beta=beta, **kwargs)
        self.x_sum = 0
        self.n = 0

    def update(self, data):
        self.x_sum = np.sum(data)
        self.n = len(data)

    @property
    def post_alpha(self):
        return self.alpha + self.x_sum

    @property
    def post_beta(self):
        return self.beta + self.n

    def sample(self, size=1):
        a = ss.nbinom.rvs(n=self.post_alpha, p=self.post_beta/(1+self.post_beta), size=size)
        if size == 1:
            return a[0]
        else:
            return a


class ProbaModelBuildGeneticAlgo(object):
    def __init__(self, model_space, buffer_type, buffer_size=1, batch_size=100):
        """the workhorse for building model hyperparameters using Bayesian genetic algorithms

        I tried to match the arguments with BaseController, whenever possible; however, this means some arguments will
        need better naming and/or documentations in the future, as a unified nomenclature to make better sense

        Parameters
        ----------
        model_space : amber.architect.ModelSpace
            model space with computation operation's hyperparameters follow a Bayesian distribution, instead of fixed
            values
        buffer_type : str, or callable
            buffer identifier or callable to parse to amber.buffer.get_buffer
        buffer_size : int
            population size; how many history episodes/epochs to keep in memory
        batch_size : int
            within each epidose/epoch, how many individuals to keep
        """
        self.model_space = model_space
        assert isinstance(self.model_space, ModelSpace)
        for i in range(len(self.model_space)):
            assert len(self.model_space[i]) == 1, "pmbga does not support more than 1 operations per layer; check %i" % i
        #buffer_fn = get_buffer(buffer_type)
        assert buffer_type.lower() == 'population'
        buffer_fn = PopulationBuffer
        self.buffer = buffer_fn(max_size=buffer_size,
                                #ewa_beta=max(1 - 1. / buffer_size, 0.9),
                                discount_factor=0.,
                                is_squeeze_dim=True)
        self.batch_size = batch_size
        self.model_space_probs = self._get_probs_in_model_space()

    def get_action(self):
        arcs_tokens = []
        for i in range(len(self.model_space)):
            op = self.model_space[i][0]
            tmp = {}
            for k, v in op.Layer_attributes.items():
                if isinstance(v, BayesProb):
                    v_samp = v.sample()
                    tmp[k] = v_samp
                else:
                    tmp[k] = v
            arcs_tokens.append(Operation(op.Layer_type, **tmp))
        return arcs_tokens

    def store(self, action, reward):
        """
        Parameters
        ----------
        action : list
            A list of architecture tokens sampled from posterior distributions

        reward : float
            Reward for this architecture, as evaluated by ``amber.architect.manager``
        """
        self.buffer.store(state=[], prob=[], action=action, reward=reward)
        return self

    def _get_probs_in_model_space(self):
        d = {}
        for i in range(len(self.model_space)):
            for k, v in self.model_space[i][0].Layer_attributes.items():
                if isinstance(v, BayesProb):
                    d[(i, self.model_space[i][0].Layer_type, k)] = v
        return d

    def _parse_probs_in_arc(self, arc, update_dict):
        for i in range(len(self.model_space)):
            obs = arc[i]
            for k, v in self.model_space[i][0].Layer_attributes.items():
                if isinstance(v, BayesProb):
                    update_dict[v].append(obs.Layer_attributes[k])
        return update_dict

    def train(self, episode, working_dir):
        try:
            self.buffer.finish_path(self.model_space, episode, working_dir)
        except Exception as e:
            print("cannot finish path in buffer because: %s" % e)
            sys.exit(1)
        # parse buffers; only arcs w/ reward > r_bias will be yielded
        gen = self.buffer.get_data(bs=self.batch_size)
        arcs = []
        rewards = []
        for data in gen:
            arcs.extend(data[2])
            rewards.extend(data[4])
        # match sampled values to priors
        update_dict = defaultdict(list)
        for a, r in zip(*[arcs, rewards]):
            update_dict = self._parse_probs_in_arc(arc=a, update_dict=update_dict)

        # update posterior with data
        for k, v in update_dict.items():
            k.update(data=v)


def convert_model_arc_to_param_file(arc, fp):
    """
    A temporary solution for lack of general-purpose KINN model builder;
    save param yaml files for each configuration
    """
    # SP_ATTR = ['SOURCE', 'TARGET', 'EDGE', 'CONTRIB', 'RANGE_ST', 'RANGE_D']
    EXP_CONFIG = """# Example of a parameter file 
Title: 'test_1'
Input:
  seq_length: 50
  values: ["A", "G", "T", "C"]
States:
  - '0' 
  - '1' 
  - '2'
  - '3'
Data:
  npoints: 20000
  contrib_rate_names: ['k_{30}']
  pheno_map: 'sigmoid()'
    """
    config = yaml.safe_load(EXP_CONFIG)
    config['Rates'] = []
    k = 0
    contribs = []
    for op in arc:
        if op.Layer_attributes['EDGE'] == 0:
            continue
        rate_id = "k_%i" % k
        rate = {
            'name': rate_id,
            'state_list': [op.Layer_attributes['SOURCE'], op.Layer_attributes['TARGET']],
            'input_range': [op.Layer_attributes['RANGE_ST'],
                            op.Layer_attributes['RANGE_ST']+op.Layer_attributes['RANGE_D']],
            'weight_distr': 'nuc_distr(length, ind_scale=[0,0,0,0,])'
        }
        config['Rates'].append(rate)
        if op.Layer_attributes['CONTRIB'] == 1:
            contribs.append(rate_id)
        k += 1
    config['Data']['contrib_rate_names'] = contribs
    s = yaml.dump(config)
    with open(fp, 'w') as f:
        f.write(s)
    return s



