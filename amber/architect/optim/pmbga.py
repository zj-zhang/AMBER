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
from collections import defaultdict
from ..buffer import Buffer, PopulationBuffer
from ..modelSpace import ModelSpace, Operation

# sort out more deps
#from bpm_ml_blending.gen_kinetic_weight_mats import KineticModel
import yaml


class BayesProb:
    def __init__(self, store_prior=True, **kwargs):
        self.__dict__.update(kwargs)
        if store_prior is True:
            self.prior_dist = self.sample(size=2000)
        else:
            self.prior_dict = []

    def update(self, data, **kwargs):
        """
        data : np.array-like
        """
        return self

    def sample(self, size=1):
        pass


class Categorical(BayesProb):
    def __init__(self, choices, prior_cnt, **kwargs):
        self.obs_cnt = np.array([0]*len(choices))
        super().__init__(choices=choices, prior_cnt=prior_cnt, **kwargs)
        self.choice_lookup = {c:i for i,c in enumerate(self.choices)}

    def update(self, data, reset=True):
        assert all([c in self.choice_lookup for c in data]), "invalid category choices in data"
        if reset is True:
            self.obs_cnt = np.array([0]*len(self.choices))
        for c in data:
            self.obs_cnt[self.choice_lookup[c]] += 1
        return self

    def sample(self, size=1):
        cnt = (self.obs_cnt + self.prior_cnt)
        p = cnt/np.sum(cnt)
        onehots = ss.multinomial.rvs(n=1, p=p, size=size)
        cats = [self.choices[np.argmax(x)] for x in onehots]
        if size == 1:
            return cats[0]
        else:
            return cats


class EmpiricalGaussianKDE(BayesProb):
    def __init__(self, integerize=False, lb=None, ub=None, **kwargs):
        self.integerize = integerize
        self.lb = lb if lb is not None else -np.inf
        self.ub = ub if ub is not None else np.inf
        self.data = []
        self.kernel = None
        #center = 0 if (lb is None or ub is None) else (lb+ub)/2
        self.update(ss.uniform.rvs(loc=self.lb, scale=self.ub-self.lb, size=1000))
        super().__init__(**kwargs)

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
        a = np.clip(a, self.lb, self.ub)
        if self.integerize is True:
            a = int(a) if size == 1 else np.array(a, dtype=int)
        return a

class Binomial(BayesProb):
    def __init__(self, alpha, beta, n=1, **kwargs):
        self.x = []
        self.n = n
        super().__init__(alpha=alpha, beta=beta, **kwargs)

    def update(self, data):
        assert all(self.n >= np.asarray(data))
        self.x = np.asarray(data)
        return self

    @property
    def post_a(self):
        return self.alpha + sum(self.x)

    @property
    def post_b(self):
        return self.beta + len(self.x)*self.n - sum(self.x)

    def sample(self, size=1):
        a = ss.betabinom.rvs(n=self.n, a=self.post_a, b=self.post_b, size=size)
        if size == 1:
            return a[0]
        else:
            return a


class TruncatedNormal(BayesProb):
    def __init__(self, mu_0, k0, sigma2_0, v0, integerize=False, lb=None, ub=None, **kwargs):
        """truncated normal with unknown mean and variance.
        Conjugate prior is inverse gamma

        Parameters
        ----------
        mu_0 : prior guess of where the mean is
        k0 : certainty about mu_0; compare with number of data samples.
        sigma2_0 : prior guess of variance.
        v0 : cerntainty of sigma2 parameter; compare with number of data samples.

        See Also
        ----------
        This code has a small typo that will mess up the variance
        https://richrelevance.com/2013/07/31/bayesian-analysis-of-normal-distributions-with-python/
        """
        self.mu = 0
        self.n = 0
        self.sigma2 = 0
        self.lb = lb if lb is not None else -np.inf
        self.ub = ub if ub is not None else np.inf
        self.integerize = integerize
        super().__init__(mu_0=mu_0, k0=k0, sigma2_0=sigma2_0, v0=v0, **kwargs)

    def update(self, data):
        self.mu = np.mean(data)
        self.n = len(data)
        self.sigma2 = np.var(data)
        self.sigma2 = np.max([self.sigma2, 0.001])
        return self

    def _sample_one(self):
        kN = self.k0 + self.n
        mN = (self.k0/kN)*self.mu_0 + (self.n/kN)*self.mu
        vN = self.v0 + self.n
        vN_times_sigma2N = self.v0*self.sigma2_0 + self.sigma2*self.n + (self.n*self.k0*(self.mu_0-self.mu)**2)/kN
        alpha = vN/2
        beta = vN_times_sigma2N/2
        self.post_alpha = alpha
        self.post_beta = beta
        # heirarhical sampling
        while 1:
            sig_sq_samples = beta * ss.invgamma.rvs(alpha, size=1)
            mean_norm = mN
            #var_norm = np.sqrt(sig_sq_samples) / kN # DO NOT DIVIDE kN HERE; fzz, 2021.11.13
            var_norm = np.sqrt(sig_sq_samples)
            a = ss.norm.rvs(mean_norm, scale=var_norm, size=1)[0]
            #a = ss.norm.rvs(loc=self.post_loc, scale=self.post_scale, size=1)
            if a > self.lb and a < self.ub:
                break
        return a

    def sample(self, size=1):
        a = self._sample_one() if size == 1 else [self._sample_one() for _ in range(size)]
        if self.integerize is True:
            a = int(a) if size == 1 else np.array(a, dtype=int)
        return a

    @property
    def post_loc(self):
        kN = self.k0 + self.n
        mN = (self.k0/kN)*self.mu_0 + (self.n/kN)*self.mu
        return mN
 
    @property
    def post_scale(self):
        kN = self.k0 + self.n
        vN = self.v0 + self.n
        vN_times_sigma2N = self.v0*self.sigma2_0 + self.sigma2*self.n + (self.n*self.k0*(self.mu_0-self.mu)**2)/kN
        alpha = vN/2
        beta = vN_times_sigma2N/2
        return alpha, beta
 
    #@property
    #def post_scale(self):
    #    return np.sqrt(1/(1/self.sigma2_0 + self.n/self.sigma2))



class Poisson(BayesProb):
    def __init__(self, alpha, beta, **kwargs):
        """
        Parameters
        ----------
        alpha : float
            number of total occurences
        beta : float
            size of interval
        """
        self.x_sum = 0
        self.n = 0
        super().__init__(alpha=alpha, beta=beta, **kwargs)

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


class ZeroTruncatedNegativeBinomial(Poisson):
    """TODO: this is for sure to be slow...; there should be a closed-form solution to this
    """
    def _sample_one(self):
        a = 0
        while a == 0:
            a = super().sample(size=1)
        return a

    def sample(self, size=1):
        if size == 1:
            return self._sample_one()
        else:
            return [self._sample_one() for _ in range(size)]


class ProbaModelBuildGeneticAlgo(object):
    def __init__(self, model_space, buffer_type, buffer_size=1, batch_size=100, *args, **kwargs):
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
        ewa_beta = kwargs.get("ewa_beta", 0)
        if ewa_beta is None or ewa_beta=='auto':
            ewa_beta = 1 - 1./buffer_size
        self.buffer = buffer_fn(max_size=buffer_size,
                                ewa_beta=ewa_beta,
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
        return arcs_tokens, None

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
            self.buffer.finish_path(state_space=self.model_space, global_ep=episode, working_dir=working_dir)
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
        print("datapoints: ", len(v), "/ total: ", len(np.concatenate(self.buffer.lt_nrbuffer, axis=0)))
