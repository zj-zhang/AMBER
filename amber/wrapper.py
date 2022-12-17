# -*- coding: utf-8 -*-

"""
Overall wrapper class for building an AMBER App.
"""

from . import backend as F
import os
from . import getter


class Amber:
    """The main wrapper class for AMBER

    For most direct users, this is the main entry for running an AMBER model search.
    The purpose of this class can be summarized as follows:

        - provides a unified set of class methods for managing AMBER initialization, run, and storage
        - provides class attributes that reference to underlying AMBER components, for fine-tuning and debugging
        - provides the base class for future developments

    The class can be initialized by passing two dictionaries together, ``types`` and ``specs``, then an Amber instance
    will be constructed at initialization. Alternatively, if one just passed ``types``, then only place holders will be
    built, and tensors can be constructed later using `from_dict` method.

    This class facilitates the CLI and GUI caller, and should always be maintained.

    Parameters
    ----------
    types: dict
        The types dictionary specifies the type of controller, manager to be constructed. It should have the following
        keys: `['controller_type', 'modeler_type', 'knowledge_fn_type', 'reward_fn_type', 'manager_type', 'env_type']`.
    specs: dict, or None
        Parameter specifications for each type. It can be different depending on different types of AMBER components
        being constructed. Most of the specification dictionary works by just copy-pasting from an example script, while
        it can also get very long and detailed for customized runs.

    Attributes
    ----------
    type_dict: dict
    is_built: bool
    session : amber.backend.Session, or None
    model_space: amber.architect.ModelSpace
    controller: amber.architect.BaseController
    manager : amber.architect.GeneralManager
    env : amber.architect.ControllerTrainEnvironment

    Example
    ----------
    PENDING EDITION
    """

    def __init__(self, types, specs=None):
        self.type_dict = types
        self.is_built = False
        self.model_space = None
        self.controller = None
        self.model_fn = None
        self.knowledge_fn = None
        self.reward_fn = None
        self.manager = None
        self.env = None

        # use one amber.backend.Session throughout one DA instance
        self.session = F.Session()
        F.set_session(self.session)

        if specs is not None:
            self.from_dict(specs)

    def from_dict(self, d):
        assert type(d) is dict
        print("BUILDING")
        print("-" * 10)
        self.model_space = getter.get_model_space(d['model_space'])
        self.controller = getter.get_controller(controller_type=self.type_dict['controller_type'],
                                                model_space=self.model_space,
                                                session=self.session,
                                                **d['controller'])

        self.model_fn = getter.get_modeler(model_fn_type=self.type_dict['modeler_type'],
                                           model_space=self.model_space,
                                           session=self.session,
                                           controller=self.controller,
                                           **d['model_builder'])

        self.knowledge_fn = getter.get_knowledge_fn(knowledge_fn_type=self.type_dict['knowledge_fn_type'],
                                                    knowledge_data_dict=d['knowledge_fn']['data'],
                                                    **d['knowledge_fn']['params'])

        self.reward_fn = getter.get_reward_fn(reward_fn_type=self.type_dict['reward_fn_type'],
                                              knowledge_fn=self.knowledge_fn,
                                              **d['reward_fn'])

        self.manager = getter.get_manager(manager_type=self.type_dict['manager_type'],
                                          model_fn=self.model_fn,
                                          reward_fn=self.reward_fn,
                                          data_dict=d['manager']['data'],
                                          session=self.session,
                                          **d['manager']['params'])

        self.env = getter.get_train_env(env_type=self.type_dict['env_type'],
                                        controller=self.controller,
                                        manager=self.manager,
                                        **d['train_env'])
        self.is_built = True
        return self

    def run(self):
        assert self.is_built
        self.env.train()
        self.controller.save_weights(os.path.join(self.env.working_dir, "controller_weights.h5"))


class AmberSpecifications:
    """Creates a set of parameter specification for given AMBER component types

    """
    KEY_ATTR_MAP = {
        'train_env': '_train_env',
        'controller': '_controller',
        'model_space': '_model_space',
        'manager': '_manager',
        'model_builder': '_model_builder',
        'reward_fn': '_reward_fn',
        'knowledge_fn': '_knowledge_fn'
    }

    def __init__(self, name, types=None):
        self.name = name
        # store the types
        self._types = types
        # private attributes do not make accessible unless use set method
        self._train_env = {}
        # for controller
        self._controller = {}
        self._model_space = None
        # for manager
        self._manager = {}
        self._model_builder = {}
        self._reward_fn = {}
        self._knowledge_fn = {}

    def __repr__(self):
        return f"{self.name} - Amber Parameter Specifications"

    def from_dict(self, d):
        assert isinstance(d, dict)
        for k, v in d.items():
            assert k in self.KEY_ATTR_MAP, ValueError("Input key provided named '%s' is invalid" % k)
            setattr(self, self.KEY_ATTR_MAP[k], v)

    def to_dict(self):
        import copy
        d = {}
        for k in self.KEY_ATTR_MAP:
            d[k] = copy.copy(getattr(self, self.KEY_ATTR_MAP[k]))
        return d

    def save_pickle(self, fp):
        import pickle
        with open(fp, "wb") as f:
            pickle.dump(self, f)

    @property
    def types(self):
        assert self._types is not None
        return self._types

    @property
    def controller(self):
        return self._controller

    @property
    def manager(self):
        return self._manager
