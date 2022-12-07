import torch
from torch.testing._internal.common_utils import TestCase as torch_testCase
import contextlib
import numpy as np
from .cache import *
session_cache = {}

def Session(*args):
    pass

def clear_session(*args):
    pass

def set_session(*args):
    pass

class device_scope:
    def __init__(self, device, *args, **kwargs):
        self.device = device
    def __enter__(self):
        GLOBAL_DEFAULT_GRAPH.set_device(self.device)
    def __exit__(self, *args):
        GLOBAL_DEFAULT_GRAPH.set_device(GLOBAL_DEFAULT_GRAPH.DEFAULT_DEVICE)

class variable_scope:
    def __init__(self, name, *args, **kwargs):
        self.name = name
    def __enter__(self):
        GLOBAL_DEFAULT_GRAPH.append_var_scope(self.name)
    def __exit__(self, *args):
        GLOBAL_DEFAULT_GRAPH.strip_var_scope()
 

class TestCase(torch_testCase):
    @staticmethod
    def assertLess(a,b):
        assert a < b

    @staticmethod
    def assertAllClose(a,b, *args, **kwargs):
        np.testing.assert_allclose(a,b, *args, **kwargs)

    @staticmethod
    def assertLen(a,b):
        assert len(a) == b

    @staticmethod
    def assertEqual(a,b):
        assert a==b

    @staticmethod
    def assertAllEqual(a,b):
        assert all(a==b)