"""Test architect helpers (that is, buffer, store, manager and train environments)
"""

# TODO: parameterize tests for different sub classes

import tensorflow as tf
import unittest
import numpy as np
import tempfile
import copy
from amber.utils import testing_utils
from amber import architect


class TestBuffer(unittest.TestCase):
    def setUp(self):
        super(TestBuffer, self).setUp()
        self.tempdir = tempfile.TemporaryDirectory()
        self.buffer_getter = architect.buffer.get_buffer('ordinal')
        self.buffer = self.buffer_getter(max_size=4, is_squeeze_dim=True)
        self.state_space, _ = testing_utils.get_example_conv1d_space(num_layers=2)
        self.num_ops_per_layer = len(self.state_space[0])
        for _ in range(8):
            state, proba, act = self._get_data()
            self.buffer.store(state=state, prob=proba, action=act, reward=1)

    def tearDown(self):
        self.tempdir.cleanup()
        super(TestBuffer, self).tearDown()

    def _get_data(self):
        state = np.random.sample(4).reshape((1, 1, 4))
        proba = [
            # 1st layer operations, (1, n)
            np.random.sample(self.num_ops_per_layer).reshape((1, self.num_ops_per_layer)),
            # 2nd layer operations, (1, n)
            np.random.sample(self.num_ops_per_layer).reshape((1, self.num_ops_per_layer)),
            # 2nd layer residual con., (1, 1, 2)
            np.random.sample(2).reshape((1, 1, 2))
        ]
        act = np.random.choice(2, 3).astype('int')
        return state, proba, act

    def test_finish_path(self):
        self.buffer.finish_path(
            state_space=self.state_space,
            global_ep=0,
            working_dir=self.tempdir.name
        )
        # after path finish, long-term should be filled
        self.assertNotEqual(len(self.buffer.lt_sbuffer), 0)
        self.assertNotEqual(len(self.buffer.lt_abuffer), 0)
        self.assertNotEqual(len(self.buffer.lt_pbuffer), 0)
        self.assertNotEqual(len(self.buffer.lt_adbuffer), 0)

    def test_get(self):
        self.buffer.finish_path(
            state_space=self.state_space,
            global_ep=0,
            working_dir=self.tempdir.name
        )
        for data in self.buffer.get_data(bs=2):
            states, probas, acts, ads, rewards = data
            self.assertEqual(len(ads), 2)
            for pr in probas:
                self.assertEqual(len(pr), 2)
            self.assertEqual(len(acts), 2)
            self.assertEqual(len(rewards), 2)
            self.assertEqual(type(probas), list)


class TestManager(unittest.TestCase):
    pass



if __name__ == '__main__':
    unittest.main()
