"""Test architect genetic algorithm optimizer"""

import tensorflow as tf
import numpy as np
import copy
import tempfile
from amber.utils import testing_utils
from amber.architect import pmbga
from amber.architect import ModelSpace, Operation


class TestPmbga(testing_utils.TestCase):
    def setUp(self):
        super(TestPmbga, self).setUp()
        self.model_space = ModelSpace.from_dict([
            [dict(Layer_type='conv1d', kernel_size=pmbga.Poisson(8, 1), filters=4)],
            [dict(Layer_type='conv1d', kernel_size=pmbga.Poisson(4, 1), filters=4)],
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


if __name__ == '__main__':
    tf.test.main()
