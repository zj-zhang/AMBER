"""
:mod:`offline_learn` provides utilities for learning from a fixed historical data

because of this, it also serves as simulated benchmarking for architecture search

This module's documentation is Work-In-Progress.

"""


from . import grid_search, mock_manager, gold_standard

__all__ = [
    'grid_search',
    'mock_manager',
    'gold_standard'
]
