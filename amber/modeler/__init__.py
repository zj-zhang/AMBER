"""
Modeler is an interface class that interacts outside with manager, and inside coordinates with dag and child.
- Dag builds the underlying tensors
- child facilitates training and evaluating
"""

from . import resnet, supernet, sequential, sparse_ffnn
from . import architectureDecoder

__all__ = [
    'resnet',
    'supernet',
    'sequential',
    'sparse_ffnn'
]
