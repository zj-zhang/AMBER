"""
Test modeler DAGs (underlying modelers)
"""

import tensorflow as tf
import torch
import numpy as np

from amber.utils import testing_utils
from amber import modeler
from amber import architect
from amber.modeler.dag_pytorch import EnasConv1dDAGpyTorch
import logging, sys
logging.disable(sys.maxsize)

class TestEnasPyTorchConvDAG(testing_utils.TestCase):
    def setUp(self):
        try:
            self.session = tf.Session()
        except AttributeError:
            self.session = tf.compat.v1.Session()

        input_op = architect.Operation('input', shape=(100, 4), name="input")
        output_op = architect.Operation('dense', units=1, activation='sigmoid', name="output")
        model_space, _ = testing_utils.get_example_conv1d_space(num_layers=12, num_pool=4)
        self.controller = architect.GeneralController(model_space=model_space, buffer_type='ordinal', with_skip_connection=True, session=self.session)
        self.dag = EnasConv1dDAGpyTorch(model_space=model_space, input_node=input_op,  output_node=output_op, model_compile_dict={}, reduction_factor=2)

    def test_forward(self):
        x = torch.randn((10, 100, 4))
        for _ in range(10):
            arc, p = self.controller.get_action()
            y_pred = self.dag(arc, x)
            self.assertTrue(y_pred.shape == (10,1))


if __name__ == '__main__':
    tf.test.main()
