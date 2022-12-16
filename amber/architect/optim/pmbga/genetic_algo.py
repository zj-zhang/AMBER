"""
pmbga = Probabilistic Model-Building Genetic Algorithms

From Wiki:
Estimation of distribution algorithms (EDAs), sometimes called probabilistic model-building genetic algorithms (
PMBGAs),[1] are stochastic optimization methods that guide the search for the optimum by building and sampling
explicit probabilistic models of promising candidate solutions. Optimization is viewed as a series of incremental
updates of a probabilistic model, starting with the model encoding an uninformative prior over admissible solutions
and ending with the model that generates only the global optima.
"""

import sys
import numpy as np
from collections import defaultdict
from .bayes_prob import *
from ...buffer import PopulationBuffer
from ...modelSpace import ModelSpace, Operation


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
