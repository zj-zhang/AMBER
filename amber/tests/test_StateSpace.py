# -*- coding: UTF-8 -*-
import unittest

from BioNAS.Controller.model import *
from BioNAS.Controller.model_space import *


class StateSpace_Test(unittest.TestCase):

    def test_StateSpace(self):
        state_space = ModelSpace()

        # add layer by individual states
        state_space.add_layer(0)
        state_space.add_state(0, State('conv1d', filters=10, kernel_size=10))
        s1 = State('conv1d', filters=3, kernel_size=10)
        state_space.add_state(0, s1)
        s2 = State('conv1d', filters=3, kernel_size=15)
        state_space.add_state(0, s2)

        # add layer by entire list
        state_space.add_layer(1, [State('maxpool1d', pool_size=10, strides=10),
                                  State('avgpool1d', pool_size=10, strides=10), ])
        state_space[2] = [State('Flatten'), State('GlobalMaxPool1D')]

        # testing prints
        print(state_space)
        print('length=', len(state_space))
        state_space.print_state_space()
        state_space.get_random_model_states()

    def test_ModelBuild_from_StateSpace(self):
        state_space = ModelSpace()
        state_space.add_layer(0, [State('conv1d', filters=3, kernel_size=10),
                                  State('conv1d', filters=10, kernel_size=10), ])
        state_space.add_layer(1, [State('maxpool1d', pool_size=10, strides=10),
                                  State('avgpool1d', pool_size=10, strides=10), ])
        model_states = state_space.get_random_model_states()

        input_state = State('Input', shape=(200, 4))
        output_state = State('Dense', units=1, activation='sigmoid')

        model_compile_dict = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ['acc']}
        model = build_sequential_model(model_states, input_state, output_state, model_compile_dict)


if __name__ == '__main__':
    unittest.main()
