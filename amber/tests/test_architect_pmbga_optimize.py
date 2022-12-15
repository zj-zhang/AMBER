"""Test architect genetic algorithm optimizer"""

import unittest
import numpy as np
import copy
import tempfile
from amber.utils import testing_utils
from amber.architect.optim import pmbga
from amber.architect import ModelSpace, Operation


class TestPmbga(testing_utils.TestCase):
    def setUp(self):
        super(TestPmbga, self).setUp()
        self.model_space = ModelSpace.from_dict([
            [dict(Layer_type='conv1d', kernel_size=pmbga.bayes_prob.Poisson(8, 1), filters=4)],
            [dict(Layer_type='conv1d', kernel_size=pmbga.bayes_prob.Poisson(4, 1), filters=4)],
        ])
        self.controller = pmbga.ProbaModelBuildGeneticAlgo(
            model_space=self.model_space,
            buffer_type='population',
            buffer_size=1,
            batch_size=5,
        )
        self.tempdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        super(TestPmbga, self).tearDown()
        self.tempdir.cleanup()

    def test_get_action(self):
        arc, _ = self.controller.get_action()
        self.assertTrue(type(arc) is list)
        self.assertTrue(len(arc) == 2)
        for i in range(2):
            self.assertTrue(type(arc[i]) is Operation)
            self.assertTrue(isinstance(arc[i].Layer_attributes['kernel_size'], np.integer))
            self.assertTrue(arc[i].Layer_attributes['filters'] == 4)

    def test_store_and_fetch(self):
        arc_dict = {}
        for i in range(30):
            arc = self.controller.get_action()
            arc_dict[i] = arc
            self.controller.store(action=arc, reward=i)

        self.controller.buffer.finish_path(self.model_space, 0, self.tempdir.name)
        gen = self.controller.buffer.get_data(5)
        cnt = 0
        for data in gen:
            cnt += 1
            self.assertAllGreater(data[-1], 14.5)
            arc = data[2]
            reward = data[4]
            for a, r in zip(*[arc, reward]):
                #print([str(x) for x in arc_dict[r]], r)
                #print([str(x) for x in a], r)
                self.assertAllEqual(arc_dict[r], a)
        self.assertTrue(cnt == 3)

    def test_train(self):
        for i in range(30):
            arc = [Operation('conv1d', kernel_size=12, filters=4), Operation('conv1d', kernel_size=1, filters=4)]
            self.controller.store(action=arc, reward=i)
        self.controller.train(episode=0, working_dir=self.tempdir.name)
        self.assertLess(8, self.controller.model_space_probs[(0, 'conv1d', 'kernel_size')].sample(size=100).mean())
        self.assertGreater(4, self.controller.model_space_probs[(1, 'conv1d', 'kernel_size')].sample(size=100).mean())


class TestBayesProbs(unittest.TestCase):
    def test_categorical(self):
        pr = pmbga.bayes_prob.Categorical(choices=[1,3,5], prior_cnt=1)
        pr.update([1,1,1], reset=True)
        d = pr.sample(10)
        assert all([_ in [1,3,5] for _  in d])
        pr.update([1,1,1], reset=False)
        d = pr.sample(1)
        assert d in [1,3,5]
    
    def test_empirical_gaussian(self):
         pr = pmbga.bayes_prob.EmpiricalGaussianKDE(integerize=True, lb=0, ub=30)
         pr.update([1,2,3,3,2,1], reset=True)
         d = pr.sample(1)
         assert d == int(d)
         assert 0 <= d <= 30
         pr = pmbga.bayes_prob.EmpiricalGaussianKDE(integerize=False, lb=0, ub=30)
         pr.update([1,2,3,3,2,1], reset=False)
         d = pr.sample(10)
        
    def test_binomial(self):
        pr = pmbga.bayes_prob.Binomial(alpha=5, beta=5, n=1)
        pr.update([0,1,0,1], reset=True)
        pr.sample(1)
        pr = pmbga.bayes_prob.Binomial(alpha=5, beta=5, n=2)
        pr.update([1,2,0,1], reset=False)

    def test_truncated_normal(self):
        pr = pmbga.bayes_prob.TruncatedNormal(mu_0=5, k0=1, sigma2_0=2, v0=1)
        pr.update([3,4,5,6,7])
        pr.sample(10)
        assert pr.post_loc > 0
        assert pr.post_scale[0] > 0 and pr.post_scale[1] > 0
        pr = pmbga.bayes_prob.TruncatedNormal(mu_0=5, k0=1, sigma2_0=2, v0=1, integerize=True, lb=3, ub=10)
        pr.update([3,4,5,6,7])
        d = pr.sample(10)
        assert all(3<= np.array(d))
        assert all(np.array(d) <= 10)
    
    def test_poisson(self):
        pr = pmbga.bayes_prob.Poisson(alpha=3, beta=10)
        pr.update([0,5,7])
        pr.sample(10)
        assert pr.post_alpha > 0 and pr.post_beta > 0
    
    def test_ztnb(self):
        pr = pmbga.bayes_prob.ZeroTruncatedNegativeBinomial(alpha=3, beta=10)
        pr.update([0,5,7])
        d = pr.sample(10)
        assert pr.post_alpha > 0 and pr.post_beta > 0
        assert all(np.array(d)> 0 )
       

if __name__ == '__main__':
    unittest.main()
