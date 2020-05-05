#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
ZZJ
Nov. 16, 2018
"""

from .general_obj_math import *


class GeneralKnowledgeObjectiveFunction(object):
    """Scaffold of objective knowledge-functions
    """

    def __init__(self):
        self.W_model = None
        self.W_knowledge = None
        self._build_obj_func()

    def __call__(self, model, data, **kwargs):
        self.model_encoder(model, data, **kwargs)
        return self.obj_fn(self.W_model, self.W_knowledge, **kwargs)

    def __str__(self):
        return "Objective function for augmenting neural architecture search"

    def model_encoder(self, model, data, **kwargs):
        """encode $\hat{W}$ from model
        """
        raise NotImplementedError("abstract method")

    def knowledge_encoder(self, **kwargs):
        """encode $\tilde{W}$ from existing knowledge
        """
        raise NotImplementedError("abstract method")

    def _build_obj_func(self, **kwargs):
        def obj_fn(W_model, W_knowledge, **kwargs):
            return None

        self.obj_fn = obj_fn

    def get_obj_val(self, **kwargs):
        return self.obj_fn(self.W_model, self.W_knowledge, **kwargs)



