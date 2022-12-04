import torch
from torch.testing._internal.common_utils import TestCase
import contextlib
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