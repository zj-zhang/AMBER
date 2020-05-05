# -*- coding: utf-8 -*-

"""
Overall wrapper class for DeepAmbre
"""

from tensorflow import Session

from . import _wrapper


class Amber:
    """The main wrapper class for AMBER

    This class facilitates the GUI and TUI caller, and should always be maintained

    Parameters
    ----------
        types: TODO

        specs: TODO

    Attributes
    ----------
        type_dict: TODO
        is_built: TODO
        model_space: TODO
        controller: TODO


    Example
    ----------
    TODO: exmaplar use

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

        # use one tf.Session throughout one DA instance
        self.session = Session()

        if specs is not None:
            self.from_dict(specs)

    def from_dict(self, d):
        assert type(d) is dict
        self.model_space = _wrapper.get_model_space(d['model_space'])
        self.controller = _wrapper.get_controller(controller_type=self.type_dict['controller_type'],
                                                  model_space=self.model_space,
                                                  session=self.session,
                                                  **d['controller'])

        self.model_fn = _wrapper.get_model_builder(model_fn_type=self.type_dict['model_fn_type'],
                                                   model_space=self.model_space,
                                                   session=self.session,
                                                   **d['model_builder'])

        self.knowledge_fn = _wrapper.get_knowledge_fn(knowledge_fn_type=self.type_dict['knowledge_fn_type'],
                                                      knowledge_data_dict=d['knowledge_fn']['data'],
                                                      **d['knowledge_fn']['params'])

        self.reward_fn = _wrapper.get_reward_fn(reward_fn_type=self.type_dict['reward_fn_type'],
                                                knowledge_fn=self.knowledge_fn,
                                                **d['reward_fn'])

        self.manager = _wrapper.get_manager(manager_type=self.type_dict['manager_type'],
                                            model_fn=self.model_fn,
                                            reward_fn=self.reward_fn,
                                            data_dict=d['manager']['data'],
                                            session=self.session,
                                            **d['manager']['params'])

        self.env = _wrapper.get_train_env(env_type=self.type_dict['env_type'],
                                          controller=self.controller,
                                          manager=self.manager,
                                          **d['train_env'])
        self.is_built = True
        return self

    def run(self):
        assert self.is_built
        self.env.train()
