"""
Zero-shot controller takes in descriptive features from dataset when calling `.get_action()` method,
then generate new architectures for new tasks using a trained controller based on the data-descriptive
features for the new task.

ZZ, May 13, 2020
"""

from ._general_controller import GeneralController


class ZeroShotController(GeneralController):
    def __init__(self, data_descriptive_feature_len, *args, **kwargs):
        self.data_descriptive_feature_len = data_descriptive_feature_len
        super().__init__(*args, **kwargs)
