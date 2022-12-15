# -*- coding: utf-8 -*-

"""
This script wraps around the AMBER package by serializing a string to an AMBER object.
Useful in order to get a working TrainEnv from a configuration text file.
"""

# Author : zzjfrank
# Create : 11.6.2019
# Update : 04.23.2022 - add examples and docstrings. FZZ
# Update : 10.17.2022 - allow subclassing to be passed to getter instead of only strings. FZZ

import os


class DataToParse:
    """A unified wrapper for reading and caching data files. Given a input file path, determines the file types by the
    file extensions, and uses corresponding data reader to unpack the data object, and return the connected file handler.

    We need this because the configuration files only inputs a file path as string, and we need proper ways to handle
    the data files in different types.

    Currently supports data in `pickle`, `h5py`, and `numpy`. File types are automatically determined in ``_extention()``,
    but can also be overwrite by specifying ``method=STRING``.

    Parameters
    ----------
    obj : str, or Callable
        a data object of flexible type.
        If `str`, expect it to be a filepath.
        If `Callable`, expect it to return a dataset when called with `kws` as keyword arguments.
        Other types are under development as needed.
    method: str, or None
        method to unpack the data. If None, will try to automatically determine by `_extension` method.
    kws: dict, or None
        keyword arguments to pass to a callable object

    Examples
    --------

    The class is often useful in constructing an `Amber` object, when passing irregular data files to a Manager:

    .. code-block:: python

        from amber.getter import DataToParse
        data = DataToParse("./data/example.npy")
        arr = data.unpack()
        # the same syntax also works if the file is h5py
        data = DataToParse("./data/example.h5")
        with data.unpack() as store:
            arr = store['X'][()]

    It is also used internally in serializing an `Amber` object, e.g.

    .. code-block:: python

        knowledge_data = DataToParse(var_dict['knowledge_data']).unpack()

    """

    def __init__(self, obj, method=None, kws=None):
        self.obj = obj
        self.method = method
        self.kws = kws or {}
        if isinstance(self.obj, str):
            self._extension()
        else:
            assert callable(self.obj), TypeError("You passed a non-string object to DataToParse but is not Callalbe")
            self.method = 'callable'

    def _extension(self):
        ext = os.path.splitext(self.obj)[1]
        if ext in (".pkl", ".pickle"):
            self.method = "pickle"
        elif ext in (".npy",):
            self.method = "numpy"
        elif ext in (".h5", ".hdf5"):
            self.method = "hdf5"
        else:
            raise Exception("Unknown data format: %s" % self.obj)

    def __str__(self):
        s = "DataToParse-%s" % self.obj
        return s

    def unpack(self):
        """returns a file handler connected to the data file"""
        if self.method != "callable":
            assert os.path.isfile(self.obj), "File does not exist: %s" % self.obj
            assert self.method is not None, (
                "Cannot determine parse method for file: %s" % self.obj
            )
            print("unpacking data.. %s" % self.obj)
            if self.method == "pickle":
                import pickle
                return pickle.load(open(self.obj, "rb"))
            elif self.method == "numpy":
                import numpy
                return numpy.load(self.obj)
            elif self.method == "hdf5":
                import h5py
                return h5py.File(self.obj, "r")
        else:
            return self.obj(**self.kws)



def load_data_dict(d):
    """helper function that converts a dictionary's values to data objects by creating ``DataToParse`` instances"""
    d_u = {}
    for k, v in d.items():
        if type(v) is DataToParse:
            d_u[k] = v.unpack()
        elif type(v) is str:
            assert os.path.isfile(v), "cannot find file: %s" % v
            d_u[k] = DataToParse(v).unpack()
        else:
            d_u[k] = v
    return d_u


# this is the ultimate goal; needs controller and manager
def get_train_env(env_type, controller, manager, *args, **kwargs):
    """Serialize a controller training environment, given controller and manager objects

    This is the last step of getting a working Amber run - build an environment for controller and manager to interact.

    The environment depends on

        - a search algorithm for optimizing neural network architectures, and
        - a manager for coordinating several types of works, including modelers, objective evaluations, buffer of history, and post-train processing.

    Parameters
    ----------
    env_type : str
        string for the environment type
    controller : amber.architect.GeneralController
        an instance inherited from amber controller, to be included as the agent in the training environment
    manager : amber.architect.GeneralManager
        an instance inherited from amber manager, to be included in the training environment

    Raises
    -------
    TypeError:
        if ``env_type`` is not a string
    ValueError:
        if ``env_type`` string is not found in current implementations

    """
    from .architect.trainEnv import ControllerTrainEnvironment
    assert isinstance(env_type, (str,)) or issubclass(env_type, ControllerTrainEnvironment), TypeError("env_type must be a subclass amber.architect.trainEnv, string")
    if (not isinstance(env_type, (str,))) and issubclass(env_type, ControllerTrainEnvironment):
        env = env_type(controller=controller, manager=manager, *args, **kwargs)
    elif env_type == "ControllerTrainEnv":
        from .architect.trainEnv import ControllerTrainEnvironment

        env = ControllerTrainEnvironment(
            controller=controller, manager=manager, *args, **kwargs
        )
    elif env_type == "EnasTrainEnv":
        from .architect.trainEnv import EnasTrainEnv

        env = EnasTrainEnv(controller=controller, manager=manager, *args, **kwargs)
    
    elif env_type == "MultiManagerEnvironment":
        from .architect.trainEnv import MultiManagerEnvironment

        env = MultiManagerEnvironment(controller=controller, manager=manager, *args, **kwargs)
    
    elif env_type == "ParallelMultiManagerEnvironment":
        from .architect.trainEnv import ParallelMultiManagerEnvironment

        env = ParallelMultiManagerEnvironment(controller=controller, manager=manager, *args, **kwargs)

    else:
        raise ValueError("cannot understand train env type: %s" % env_type)
    print("env_type = %s" % env_type)
    return env


# controller; needs model_space
def get_controller(controller_type, model_space, session, **kwargs):
    """Serialize a controller of given type, using the specified model space, and a session if needed (can be None otherwise)

    This step constructs the neural network architecture search algorithm, one of the two components in the final
    training environment.

    The search algorithm takes two neccessary arguments as inputs: a string to specify what search algorithm to use,
    and a model space to sample model architectures from.

    An amber.backend.session may be passed to run training and inference, if the controller is implemented in Tensorflow.

    Parameters
    ----------
    controller_type : str
        string for the controller type
    model_space : amber.architect.ModelSpace
        a model space instance for the search algorithms
    session : amber.backend.Session, or None
        amber.backend session
    kwargs : dict
        keyword arguments for controller, such as buffer type
    """
    from .architect.optim.controller import BaseController
    assert isinstance(controller_type, (str,)) or issubclass(controller_type, BaseController), TypeError("controller_type must be a subclass amber.architect.BaseController, or string")

    if (not isinstance(controller_type, (str,))) and issubclass(controller_type, BaseController):
        controller = controller_type(model_space=model_space, session=session, **kwargs)
    elif controller_type == "General" or controller_type == "GeneralController":
        from .architect.optim.controller import GeneralController

        controller = GeneralController(
            model_space=model_space, session=session, **kwargs
        )
    elif controller_type == "Operation" or controller_type == "OperationController":
        from .architect.optim.controller import OperationController

        controller = OperationController(model_space=model_space, **kwargs)
    elif controller_type == "MultiIO" or controller_type == "MultiIOController":
        from .architect.optim.controller import MultiIOController

        controller = MultiIOController(
            model_space=model_space, session=session, **kwargs
        )
    elif controller_type == "ZeroShot" or controller_type == "ZeroShotController":
        from .architect.optim.controller import ZeroShotController

        controller = ZeroShotController(
            model_space=model_space, session=session, **kwargs
        )
    else:
        raise Exception("cannot understand controller type: %s" % controller_type)
    print("controller = %s" % controller_type)
    return controller


# model_space
def get_model_space(arg):
    """Serialize a model space, most often from a python dictionary, or a pre-defined amber.architect.ModelSpace

    Parameters
    ----------
    arg : dict, or amber.architect.ModelSpace

    Examples
    --------

    Suppose we want to construct a very simple two-hidden-layer model space for building convolutional neural nets.

    We need to first specify the input and output node shapes, which are fixed regardless of the network architecture,
    then the remaining two layers will be implemented as a ModelSpace like so:

    .. code-block:: python

        from amber.getter import get_model_space
        ms = get_model_space([
                # first layer convolution
                [{'Layer_type': 'conv1d', 'kernel_size':4}, {'Layer_type': 'conv1d', 'kernel_size':8} ],
                # second layer pooling
                [{'Layer_type': 'globalmaxpool1d'}, {'Layer_type': 'globalavgpool1d'} ]
            ])
        print(ms)
        # output: OperationSpace with 2 layers and 4 total combinations

    """
    from .architect.modelSpace import ModelSpace, BaseModelSpace

    if type(arg) in (dict, list):
        model_space = ModelSpace.from_dict(arg)
    elif issubclass(type(arg), BaseModelSpace):
        model_space = arg
    else:
        raise Exception("cannot understand non-string model_space arg: %s" % arg)
    return model_space


# manager; needs data, model_fn, reward_fn
def get_manager(manager_type, model_fn, reward_fn, data_dict, session, *args, **kwargs):
    """Serialize a manager of given type.

    This step constructs the child neural network manager, one of the two components in the final
    training environment.

    The manager has many dependent workers to construct, train, evaluate, and save the child models.

    1. The most crucial workers are:

        - ``model_fn`` : a builder worker that converts architecture to a trainable model
        - ``reward_fn``: an evaluation worker for scoring the trained model; the evaluation can be composite and have multiple components, such as validation loss + interpretation consistency.

    2. The other workers can be passed through keyword arguments, such as:

        - ``store_fn`` : a function that specifies what to do for a trained model, e.g. save the weights, plot the model, or perform other relevant post-processing tasks

    Parameters
    ----------
    manager_type : str
        string for the manager type
    model_fn : amber.modeler.BaseModeler
        an instance of amber model builer
    reward_fn : amber.architect.reward
        an instance of amber reward
    data_dict : dictionary
        a dictionary for data filepaths, must have keys "train_data" and "validation_data"

    """
    from .architect.manager import BaseNetworkManager
    assert isinstance(manager_type, (str,)) or issubclass(manager_type, BaseNetworkManager), TypeError("manager_type must be a subclass amber.architect.BaseNetworkManager, or string")

    data_dict_unpacked = load_data_dict(data_dict)
    if (not isinstance(manager_type, (str,))) and issubclass(manager_type, BaseNetworkManager):
        manager = manager_type(
            model_fn=model_fn,
            reward_fn=reward_fn,
            train_data=data_dict_unpacked["train_data"],
            validation_data=data_dict_unpacked["validation_data"],
            *args,
            **kwargs
        )
    elif manager_type == "General" or manager_type == "GeneralManager":
        from .architect.manager import GeneralManager

        manager = GeneralManager(
            model_fn=model_fn,
            reward_fn=reward_fn,
            train_data=data_dict_unpacked["train_data"],
            validation_data=data_dict_unpacked["validation_data"],
            *args,
            **kwargs
        )
    elif manager_type == "EnasManager" or manager_type == "Enas":
        from .architect.manager import EnasManager

        manager = EnasManager(
            model_fn=model_fn,
            reward_fn=reward_fn,
            train_data=data_dict_unpacked["train_data"],
            validation_data=data_dict_unpacked["validation_data"],
            session=session,
            *args,
            **kwargs
        )
    elif manager_type == "Mock" or manager_type == "MockManager":
        from .offline_learn.mock_manager import MockManager

        manager = MockManager(model_fn=model_fn, reward_fn=reward_fn, *args, **kwargs)
    elif manager_type == "Distributed" or manager_type == "DistributedManager":
        from .architect.manager import DistributedGeneralManager

        train_data_kwargs = kwargs.pop("train_data_kwargs", None)
        validate_data_kwargs = kwargs.pop("validate_data_kwargs", None)
        devices = kwargs.pop("devices", None)
        manager = DistributedGeneralManager(
            devices=devices,
            train_data_kwargs=train_data_kwargs,
            validate_data_kwargs=validate_data_kwargs,
            model_fn=model_fn,
            reward_fn=reward_fn,
            train_data=data_dict_unpacked["train_data"],
            validation_data=data_dict_unpacked["validation_data"],
            *args,
            **kwargs
        )

    else:
        raise Exception("cannot understand manager type: %s" % manager_type)
    print("manager = %s" % manager_type)
    return manager


# model_fn
def get_modeler(model_fn_type, model_space, session, *args, **kwargs):
    """Serialize a model builder of the given type.

    Getting a modeler is one of the two essential steps in constructing a manager.

    The model builder / modeler will often depend on the model space where the search algorithm (i.e. controller) is
    sampling architectures from. This is particularly true for efficient NAS (enas)-based modelers, because a super-graph
    or super-net is built first, then sub-graphs are trained and sampled from the super-graph.
    See the ENAS paper by `Pham et al. 2018 <https://arxiv.org/abs/1802.03268>`_ for more details.

    Notably, there are also exceptions where a modeler does **not** depend on a model space, e.g. if the search algorithm
    returns a list of `amber.architect.Operation` instead of integers representing tokens, then passing a model space is
    redundent.

    Parameters
    ----------
    model_fn_type : str
        string for the model builder type
    model_space : amber.architect.ModelSpace
        model space where the search algorithms sample models
    session : amber.backend.Session
        GPU session, can be None if not applicable for certain model builders
    """
    from .modeler.base import BaseModelBuilder
    assert isinstance(model_fn_type, (str,)) or issubclass(model_fn_type, BaseModelBuilder), TypeError("model_fn_type must be a amber.modeler.ModelBuilder, or string; got %s" % type(model_fn_type))

    from .backend import Operation

    if (not isinstance(model_fn_type, (str,))) and issubclass(model_fn_type, BaseModelBuilder):
        inp_op_list = kwargs.pop("inputs_op")
        inputs_op = [Operation(**x) if not isinstance(x, Operation) else x for x in inp_op_list]
        out_op_list = kwargs.pop("outputs_op")
        output_op = [Operation(**x) if not isinstance(x, Operation) else x for x in out_op_list]
        model_fn = model_fn_type(
            model_space=model_space,
            num_layers=len(model_space),
            inputs_op=inputs_op,
            output_op=output_op,
            session=session,
            *args,
            **kwargs
        )
    elif model_fn_type in ('SparseFfnn', "SparseFfnnModelBuilder",
         "DAG", "DAGModelBuilder"): # includes legacy names
        from .modeler.sparse_ffnn import SparseFfnnModelBuilder

        assert "inputs_op" in kwargs and "outputs_op" in kwargs
        inp_op_list = kwargs.pop("inputs_op")
        inputs_op = [Operation(**x) if not isinstance(x, Operation) else x for x in inp_op_list]
        out_op_list = kwargs.pop("outputs_op")
        output_op = [Operation(**x) if not isinstance(x, Operation) else x for x in out_op_list]
        model_fn = SparseFfnnModelBuilder(
            model_space=model_space,
            num_layers=len(model_space),
            inputs_op=inputs_op,
            output_op=output_op,
            session=session,
            *args,
            **kwargs
        )
    elif model_fn_type in ("SupernetDnn", "EnasAnnModelBuilder"):
        from .modeler.supernet import EnasAnnModelBuilder

        inp_op_list = kwargs.pop("inputs_op")
        inputs_op = [Operation(**x) if not isinstance(x, Operation) else x for x in inp_op_list]
        out_op_list = kwargs.pop("outputs_op")
        output_op = [Operation(**x) if not isinstance(x, Operation) else x for x in out_op_list]
        model_fn = EnasAnnModelBuilder(
            model_space=model_space,
            num_layers=len(model_space),
            inputs_op=inputs_op,
            output_op=output_op,
            session=session,
            *args,
            **kwargs
        )
    elif model_fn_type in ("SupernetCnn", "EnasCnnModelBuilder"):
        from .modeler.supernet import EnasCnnModelBuilder

        inp_op_list = kwargs.pop("inputs_op")
        inputs_op = [Operation(**x) if not isinstance(x, Operation) else x for x in inp_op_list]
        out_op_list = kwargs.pop("outputs_op")
        output_op = [Operation(**x) if not isinstance(x, Operation) else x for x in out_op_list]
        controller = kwargs.pop("controller", None)
        model_fn = EnasCnnModelBuilder(
            model_space=model_space,
            num_layers=len(model_space),
            inputs_op=inputs_op,
            output_op=output_op,
            session=session,
            controller=controller,
            *args,
            **kwargs
        )
    elif model_fn_type in ("KerasModelBuilder", "SequentialModelBuilder"):
        from .modeler.sequential import SequentialModelBuilder

        inp_op_list = kwargs.pop("inputs_op")
        inputs_op = [Operation(**x) if not isinstance(x, Operation) else x for x in inp_op_list]
        assert (
            len(inputs_op) == 1
        ), "SequentialModelBuilder only accepts one input; try SequentialMultiIOModelBuilder for multiple inputs"
        out_op_list = kwargs.pop("outputs_op")
        output_op = [Operation(**x) if not isinstance(x, Operation) else x for x in out_op_list]
        assert (
            len(output_op) == 1
        ), "SequentialModelBuilder only accepts one output; try SequentialMultiIOModelBuilder for multiple outputs"
        model_fn = SequentialModelBuilder(
            inputs_op=inputs_op[0],
            output_op=output_op[0],
            model_space=model_space,
            *args,
            **kwargs
        )
    elif model_fn_type in ("KerasMultiIOModelBuilder", "SequentialMultiIOModelBuilder"):
        from .modeler.sequential import SequentialMultiIOModelBuilder

        inp_op_list = kwargs.pop("inputs_op")
        inputs_op = [Operation(**x) if not isinstance(x, Operation) else x for x in inp_op_list]
        out_op_list = kwargs.pop("outputs_op")
        output_op = [Operation(**x) if not isinstance(x, Operation) else x for x in out_op_list]
        model_fn = SequentialMultiIOModelBuilder(
            model_space=model_space,
            inputs_op=inputs_op,
            output_op=output_op,
            session=session,
            *args,
            **kwargs
        )

    elif model_fn_type in ("KerasBranchModelBuilder", "SequentialBranchModelBuilder"):
        from .modeler.sequential import SequentialBranchModelBuilder

        inp_op_list = kwargs.pop("inputs_op")
        inputs_op = [Operation(**x) if not isinstance(x, Operation) else x for x in inp_op_list]
        out_op_list = kwargs.pop("outputs_op")
        output_op = [Operation(**x) if not isinstance(x, Operation) else x for x in out_op_list]
        assert len(output_op) == 1
        model_fn = SequentialBranchModelBuilder(
            model_space=model_space,
            inputs_op=inputs_op,
            output_op=output_op[0],
            *args,
            **kwargs
        )

    else:
        raise Exception("cannot understand model_builder type: %s" % model_fn_type)
    print("modeler = %s" % model_fn_type)
    return model_fn


# reward_fn; depends on knowledge function
def get_reward_fn(reward_fn_type, knowledge_fn, *args, **kwargs):
    """Serialize a reward function for model evaluations.

    Getting a reward function is one of the two essential steps in constructing a manager.

    A reward function is usually the primary objective we are trying to optimize, such as the accuracy for classification,
    or the Pearson correlation/variance explained (R^2) for regression.
    It can be as simple as the negative of the loss function; since it's a reward, a larger value is what
    the search algorithm is optimizing.


    Of course it is also possible to use a composite of several metrics, such as

    .. math ::

        Reward(model) = Accuracy + \lambda_1 AUROC + \lambda_2 AUPR

    This can be easily done by passing a custom `method` argument to ``amber.architect.reward.LossAucReward``.

    See the documentation for `LossAucReward <amber.architect.html?highlight=lossaucreward#amber.architect.reward.LossAucReward>`_.

    Please note, that the reward function is **always** applied to evaluate the trained model's performance on the
    validation data. The validation data is specified in the ``manager``'s data_dict.

    In a special case, we can also augment the validation data-based reward functions with additional interpretation-based
    functions, called `knowledge functions` in AMBER.

    What's the difference of the knowledege function versus the validation data-based metrics? Apparently the
    interpretation-based knowledge data do not fall within the prediction task that the model is built for,
    otherwise this would have been appended to the validation data.

    Think of the knowledge function as a general similar measure, but not doing a forward pass of the neural nework, but
    by other transformations such as saliency (taking derivative w.r.t. to data), network pruning, or knowledge
    distillation. Such transformation's output will be compared to some interpretation-based data; because in certain
    domains such as biology, we have learned so much about how a particular system works, and having a way to encode these
    pre-existing knowledege into a set of interpretation-based data, and measure if the neural network makes predictions
    in consistency with our knowledge, is extremely useful.

    Then the optimization of neural network architecture becomes a typical multi-objective optimization problem.

    See the preprint manuscript here for joint optimization of prediction accuracy and motif extraction from convolution
    neural networks `arXiv <https://arxiv.org/abs/1909.00337>`_.

    """
    from .architect.base import BaseReward
    assert isinstance(reward_fn_type, (str,)) or issubclass(reward_fn_type, BaseReward), TypeError("reward_fn_type must be a subclass amber.architect.reward.Reward, or string")

    if (not isinstance(reward_fn_type, (str,))) and issubclass(reward_fn_type, BaseReward):
        reward_fn = reward_fn_type(knowledge_function=knowledge_fn, *args, **kwargs)
    elif reward_fn_type == "KnowledgeReward":
        from .architect.reward import KnowledgeReward

        reward_fn = KnowledgeReward(knowledge_fn, *args, **kwargs)
    elif reward_fn_type == "LossReward":
        from .architect.reward import LossReward

        assert knowledge_fn is None, (
            "Incompatability: LossReward must have knownledge_fn=None; got %s"
            % knowledge_fn
        )
        reward_fn = LossReward(*args, **kwargs)
    elif reward_fn_type in ("MockReward", "Mock_Reward"):
        from .architect.reward import MockReward

        reward_fn = MockReward(*args, **kwargs)
    elif reward_fn_type == "LossAucReward":
        from .architect.reward import LossAucReward

        # assert knowledge_fn is None, \
        #    "Incompatability: LossAucReward must have knownledge_fn=None; got %s" % knowledge_fn
        reward_fn = LossAucReward(knowledge_function=knowledge_fn, *args, **kwargs)
    elif reward_fn_type == "SparseCategoricalReward":
        from .architect.reward import SparseCategoricalReward
        reward_fn = SparseCategoricalReward(knowledge_function=knowledge_fn, *args, **kwargs)
    else:
        raise Exception("cannot understand reward_fn type: %s" % reward_fn_type)
    print("reward = %s" % reward_fn_type)
    return reward_fn


# knowledge_fn
def get_knowledge_fn(knowledge_fn_type, knowledge_data_dict, *args, **kwargs):
    """Serialize a knowledge function of given type, for a specified interpretation-based data in knowledge data
    dictionary

    See the documentation for `reward_fn <amber-cli.html#amber.getter.get_reward_fn>`_ for how the knowledge function is
    combined into consideration in the model's reward.

    """
    if knowledge_data_dict is not None:
        knowledge_data_dict = load_data_dict(knowledge_data_dict)
    # if knowledge_fn_type == "ght" or knowledge_fn_type == "GraphHierarchyTree":
    #     from .objective import GraphHierarchyTree

    #     k_fn = GraphHierarchyTree(*args, **kwargs)
    # elif (
    #     knowledge_fn_type == "ghtal" or knowledge_fn_type == "GraphHierarchyTreeAuxLoss"
    # ):
    #     from .objective import GraphHierarchyTreeAuxLoss

    #     k_fn = GraphHierarchyTreeAuxLoss(*args, **kwargs)
    # elif knowledge_fn_type == "Motif":
    #     from .objective import MotifKLDivergence

    #     k_fn = MotifKLDivergence(*args, **kwargs)
    # elif knowledge_fn_type == "AuxilaryAcc":
    #     from .objective import AuxilaryAcc

    #     k_fn = AuxilaryAcc(*args, **kwargs)
    if knowledge_fn_type == "None" or knowledge_fn_type == "zero":
        k_fn = None
    else:
        raise Exception("cannot understand knowledge_fn type: %s" % knowledge_fn_type)
    if k_fn is not None:
        if hasattr(k_fn, "knowledge_encoder"):
            k_fn.knowledge_encoder(**knowledge_data_dict)
    print("knowledge = %s" % knowledge_fn_type)
    return k_fn
