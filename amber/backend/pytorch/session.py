"""
For managing global sessions and graphs, see also: https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/python/framework/ops.py#L5108-L5141
"""
import torch
from torch.testing._internal.common_utils import TestCase as torch_testCase
import numpy as np
from . import cache

SessionType = cache.compute_graph

def Session(*args):
    return cache.compute_graph()

def clear_session(*args):
    pass

def set_session(sess):
    assert isinstance(sess, cache.compute_graph)
    cache.CURRENT_GRAPH = sess    


class session_scope():
    def __init__(self, sess=None, *args, **kwargs):
        sess = sess or cache.compute_graph()
        self.sess = sess
        assert isinstance(self.sess, cache.compute_graph)
        cache.session_cache.append(self.sess)
    
    def __enter__(self):
        cache.CURRENT_GRAPH = self.sess

    def __exit__(self, *args):
        cache.session_cache = cache.session_cache[:-1]
        self.sess.tearDown()
        cache.CURRENT_GRAPH = cache.GLOBAL_DEFAULT_GRAPH


class device_scope:
    def __init__(self, device, *args, **kwargs):
        self.device = device
    def __enter__(self):
        cache.CURRENT_GRAPH.set_device(self.device)
    def __exit__(self, *args):
        cache.CURRENT_GRAPH.set_device(cache.CURRENT_GRAPH.DEFAULT_DEVICE)

class variable_scope:
    def __init__(self, name, *args, **kwargs):
        self.name = name
    def __enter__(self):
        cache.CURRENT_GRAPH.append_var_scope(self.name)
    def __exit__(self, *args):
        cache.CURRENT_GRAPH.strip_var_scope()
 

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