import torch
from torch.testing._internal.common_utils import TestCase as torch_testCase
import contextlib
import numpy as np
session_cache = {}

def Session():
    pass

def clear_session():
    pass

def set_session():
    pass

@contextlib.contextmanager
def device_scope(*args, **kwargs):
    yield

@contextlib.contextmanager
def variable_scope(*args, **kwargs):
    yield


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