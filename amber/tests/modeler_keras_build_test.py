"""Test Keras modeler"""

import tensorflow as tf
import numpy as np

from amber.utils import testing_utils
from amber import modeler
from amber import architect


class TestKerasBuilder(testing_utils.TestCase):
    def setUp(self):
        self.model_space, _ = testing_utils.get_example_conv1d_space(num_layers=2)
        self.target_arc = [0, 0, 1]
        self.input_op = architect.Operation('input', shape=(10, 4), name="input")
        self.output_op = architect.Operation('dense', units=1, activation='sigmoid', name="output")
        self.model_compile_dict = {'loss': 'binary_crossentropy', 'optimizer': 'sgd'}
        self.x = np.random.choice(2, 40).reshape((1, 10, 4))
        self.y = np.random.sample(1).reshape((1, 1))
        self.modeler = modeler.KerasResidualCnnBuilder(
            inputs_op=self.input_op,
            output_op=self.output_op,
            model_space=self.model_space,
            fc_units=5,
            flatten_mode='flatten',
            model_compile_dict=self.model_compile_dict
        )

    def test_get_model(self):
        model = self.modeler(self.target_arc)
        old_loss = model.evaluate(self.x, self.y)
        model.fit(self.x, self.y, batch_size=1, epochs=20, verbose=0)
        new_loss = model.evaluate(self.x, self.y)
        self.assertLess(new_loss, old_loss)


if __name__ == '__main__':
    tf.test.main()
