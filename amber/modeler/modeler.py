"""
Modeler is an interface class that interacts outside with manager, and inside coordinates with dag and child.
- Dag builds the underlying tensors
- child facilitates training and evaluating
"""

from ._enas_modeler import DAGModelBuilder, EnasAnnModelBuilder, EnasCnnModelBuilder
from ._keras_modeler import build_sequential_model, build_multi_gpu_sequential_model, \
    build_multi_gpu_sequential_model_from_string, build_sequential_model_from_string


__all__ = [
    'DAGModelBuilder',
    'EnasCnnModelBuilder',
    'EnasAnnModelBuilder',
    'build_sequential_model',
    'build_sequential_model_from_string',
    'build_multi_gpu_sequential_model_from_string',
    'build_multi_gpu_sequential_model'
]