"""Test architect helpers (that is, buffer, store, manager and train environments)
"""

# TODO: parameterize tests for different sub classes

import tensorflow as tf
import unittest
import numpy as np
import copy
from amber.utils import testing_utils
from amber import architect


class TestBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer_getter = architect.buffer.get_buffer('ordinal')



class TestManager(unittest.TestCase):
    pass



if __name__ == '__main__':
    unittest.main()
