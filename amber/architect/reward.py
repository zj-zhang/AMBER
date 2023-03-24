# -*- coding: UTF-8 -*-

"""Reward function for processing a train child model

Reward function takes in trained model and validation data as input, returns reward
"""


import numpy as np
import copy
from .commonOps import unpack_data
from ..utils.io import read_history
from .base import BaseReward as Reward
from .training_free import get_ntk, Linear_Region_Collector, curve_complexity


class KnowledgeReward(Reward):
    """DOCSTRING: To be added after it's thoroughly tested

    Parameters
    ----------
    knowledge_function
    Lambda
    loss_c
    knowledge_c
    """

    def __init__(self, knowledge_function, Lambda=1.0, loss_c=None, knowledge_c=None):
        self.knowledge_function = knowledge_function
        self.Lambda = Lambda
        self.loss_c = float(loss_c) if loss_c is not None else None
        self.knowledge_c = float(knowledge_c) if knowledge_c is not None else None

    def __call__(self, model, data, **kwargs):
        X, y = data
        loss_and_metrics = model.evaluate(X, y)
        # Loss function values will always be the first value
        if type(loss_and_metrics) in (list, tuple):
            L = loss_and_metrics[0]
        else:
            L = loss_and_metrics
            loss_and_metrics = [loss_and_metrics]
        K = self.knowledge_function(model=model, data=data, **kwargs)
        reward_metrics = {'knowledge': K}
        # return -(L + self.Lambda * K), loss_and_metrics, reward_metrics
        if self.loss_c is None:
            L = - L
        else:
            L = self.loss_c / L
        if self.knowledge_c is None:
            K = - K
        else:
            K = self.knowledge_c / K
        return L + self.Lambda * K, loss_and_metrics, reward_metrics


class LossReward(Reward):
    """The most basic reward function; returns negative loss as rewards
    """

    def __init__(self, validation_steps=None, *args, **kwargs):
        self.knowledge_function = None
        self.validation_steps = validation_steps
        self.c = 1.

    def __call__(self, model, data, *args, **kwargs):
        X, y = unpack_data(data)
        #loss_and_metrics = model.evaluate(X, y, steps=self.validation_steps, verbose=0)
        # TODO: figure out what happened between Keras versions? FZZ 2021.8.1
        loss_and_metrics = model.evaluate(X, y, verbose=0)
        # Loss function values will always be the first value
        if type(loss_and_metrics) is list:
            L = loss_and_metrics[0]
        elif type(loss_and_metrics) is dict:
            L = loss_and_metrics['val_loss']
            loss_and_metrics = [L]
        elif isinstance(loss_and_metrics, float):
            L = loss_and_metrics
            loss_and_metrics = [loss_and_metrics]
        else:
            raise Exception("Cannot understand return type of model.evaluate; got %s" % type(loss_and_metrics))
        return -L, loss_and_metrics, None


class NTKReward(Reward):
    """Reward function based on NTK
    """

    def __init__(self, criterion=None, *args, **kwargs):
        self.knowledge_function = None
        self.criterion = criterion
        super(NTKReward, self).__init__()

    def __call__(self, model, data, *args, **kwargs):
        # X, y = unpack_data(data)
        # y_hat = model.forward(X)
        # be explicit about observation and score
        cond, loss = get_ntk(data, model, criterion=self.criterion)
        loss_and_metrics = [loss]
        return -cond, loss_and_metrics, None
    # return self.c/L, loss_and_metrics, None


class LRReward(Reward):
    """Reward function based on Linear Regions
    """

    def __init__(self, criterion=None, *args, **kwargs):
        self.knowledge_function = None
        self.criterion = criterion
        super(LRReward, self).__init__()

    def __call__(self, model, data, *args, **kwargs):
        # be explicit about observation and score
        assert isinstance(data, list) # multiple batch of samples
        lrc_model = Linear_Region_Collector(data, model, train_mode=True)
        _lr = lrc_model.forward_batch_sample()
        lrc_model.clear()
        loss_and_metrics = None
        return _lr, loss_and_metrics, None
    # return self.c/L, loss_and_metrics, None


class LengthReward(Reward):
    """Reward function based on Length Distorsion
    """

    def __init__(self, criterion=None, *args, **kwargs):
        self.knowledge_function = None
        self.criterion = criterion
        super(LengthReward, self).__init__()

    def __call__(self, model, data, *args, **kwargs):
        # be explicit about observation and score
        assert isinstance(data, list) # multiple batch of samples
        complexity = curve_complexity(data, model)
        loss_and_metrics = None
        return complexity, loss_and_metrics, None


class AucReward(Reward):
    """Reward function for evaluating AUC as reward

    This reward function employs a scorer to evaluate a trained model's prediction. Scorers can be parsed as strings for
    AUROC and AUPR; or a callable function that takes ``y_true`` and ``y_score`` as input arguments.

    In the case of multi-tasking, the average scorer outputs across all tasks will be returned as the reward.

    Parameters
    ----------
    method : str or callable
        The method to use to implement scorer. If is string, expects either "auc"/"auroc" or "aupr"/"auprc". Otherwise
        must be callable (see details in Example). Default is "auc".

    knowledge_function : amber.objective
        The augmented rewards for considering additional biological knowledge and how consistent the trained model is.
        Default is None.

    Lambda : float
        The weight of augmented knowledge reward, compared to regular validation data-based loss as 1.

    loss_c : float or None
        If is None, return the negative loss as is. A constant to scale loss reward, if specified. Default is None.

    knowledge_c : float or None
        If is None, return the knowledge reward as is. A constant to scale knowledge reward, if specified. Default is None.


    Examples
    --------
    Constructing the reward function for AUPR on the basis of validation data::

        >>> from amber.architect.reward import LossAucReard
        >>> reward_fn = LossAucReward(method="aupr")
        >>> reward_fn(model, val_data)

    Alternatively, use the spearman correlation instead of AUPR for regression tasks::

        >>> from amber.architect.reward import LossAucReard
        >>> import scipy.stats as ss
        >>> reward_fn = LossAucReward(method=lambda y_true, y_score: ss.spearmanr(y_true, y_score).correlation)
        >>> reward_fn(model, val_data)
    """
    def __init__(self, method='auc', knowledge_function=None, Lambda=1, loss_c=None, knowledge_c=None, *args, **kwargs):
        if method == 'auc' or method == 'auroc':
            from sklearn.metrics import roc_auc_score
            self.scorer = roc_auc_score
        elif method == 'aupr' or method == 'auprc':
            from sklearn.metrics import average_precision_score
            self.scorer = average_precision_score
        elif callable(method):
            self.scorer = method
        else:
            raise Exception("cannot understand scorer method: %s" % method)
        self.knowledge_function = knowledge_function
        self.Lambda = Lambda
        self.loss_c = float(loss_c) if loss_c is not None else None
        self.knowledge_c = float(knowledge_c) if knowledge_c is not None else None
        self.pred = kwargs.pop('pred', None)
        self.pred_bs = kwargs.pop('batch_size', 256)

    def get_pred(self, model, data):
        X, y = unpack_data(data, unroll_generator_y=True)
        if self.pred is None:
            pred = model.predict(X, batch_size=self.pred_bs)
        else:
            pred = self.pred
        if type(y) is not list:
            y = [y]
        if type(pred) is not list:
            pred = [pred]
        return pred, y

    def call_scorer(self, pred, y):
        auc_list = []
        for i in range(len(y)):
            tmp = []
            if len(y[i].shape) == 1: y[i] = np.expand_dims(y[i], axis=-1)
            if len(pred[i].shape) == 1: pred[i] = np.expand_dims(pred[i], axis=-1)
            for j in range(y[i].shape[1]):
                try:
                    score = self.scorer(y_true=y[i][:, j], y_score=pred[i][:, j])
                    tmp.append(score)
                except ValueError:  # only one-class present
                    pass
            auc_list.append(tmp)
        return auc_list

    def normalize_loss_and_knowledge(self, L, model, data):
        if self.knowledge_function is not None:
            K = self.knowledge_function(model, data)
            if self.knowledge_c is not None:
                old_K = K
                K = old_K / self.knowledge_c
        else:
            K = 0
        if self.loss_c is not None:
            old_L = L
            L = old_L / self.loss_c
        return L, K

    def __call__(self, model, data, *args, **kwargs):
        pred, y = self.get_pred(model=model, data=data)
        auc_list = self.call_scorer(pred=pred, y=y)
        L = np.nanmean(np.concatenate(auc_list, axis=0))
        self.auc_list = auc_list
        L, K= self.normalize_loss_and_knowledge(L=L, model=model, data=data)
        reward = L + self.Lambda * K
        loss_and_metrics = [L]
        reward_metrics = {'knowledge': K}
        return reward, loss_and_metrics, reward_metrics

    def min(self, data):
        """For dealing with non-valid model"""
        X, y = unpack_data(data, unroll_generator_y=True)
        if type(y) is not list: y = [y]
        pred = copy.deepcopy(y)
        _ = [np.random.shuffle(p) for p in pred]
        self.pred = pred
        reward, loss_and_metrics, reward_metrics = self.__call__(None, (X,y))
        self.pred = None  # release self.pred after use
        return reward, loss_and_metrics, reward_metrics

# alias
LossAucReward = AucReward


class SparseCategoricalReward(AucReward):
    def call_scorer(self, pred, y):
        return [ [self.scorer(y_true=y[i], y_score=pred[i])] for i in range(len(y)) ]


class F1Reward(AucReward):
    def __init__(self, method='f1', average='macro', *args, **kwargs):
        assert method == 'f1'
        super().__init__(*args, **kwargs)
        self.average = average
        self.scorer = self.my_f1

    def my_f1(self, y_true, y_score):
        from sklearn.metrics import f1_score
        return f1_score(y_true=y_true, y_pred=y_score, average=self.average)

    @staticmethod
    def _detect_reformat(x_):
        """transform to sparse label integers (0/1/2) by handling different formatted target/prediction"""
        x = np.array(x_).squeeze()
        shape = x.shape
        if len(shape) == 1:
            return x
        else:
            x = np.argmax(x, axis=-1)
            return x

    def call_scorer(self, pred, y):
        #print("before", np.array(pred).shape, np.array(y).shape)
        y_ = self._detect_reformat(y)
        pred_ = self._detect_reformat(pred)
        #print("after", pred_.shape, y_.shape)
        return [[ self.scorer(y_true=y_, y_score=pred_) ]]


def MockReward(train_history_list, metric, stringify_states, metric_name_dict, Lambda=1.):
    """MockReward is a resampler from a train history

    MockReward returns a callable reward function that works similarly to a real one : takes in an architecture sequence,
    and return its reward signals. However, the rewards are all pre-computed in the train history files instead of training
    from scratch, which is useful for fast benchmarking of architecture searching algorithms

    Parameters
    ----------
    train_history_list : list of strings
        A list of file paths to train history files

    metric : list or set
        A specified subset of metrices to return

    stringify_states : bool
        If true, will expect the architecture sequence to be categorically encoded and use it to stringify according to
        the given model space.

    metric_name_dict : dict
        A dictionary mapping metric name (keys) to their indices (values)

    Lambda : float
        The weight for augmented reward, compared to the loss-derived reward on the basis of validation data has a weight
        of 1.

    Returns
    -------
    reward_fn : callable
        A callable reward function

    See Also
    ---------
    amber.architect.reward.LossReward : A regular loss-based reward function
    """
    train_history_df = read_history(train_history_list, metric_name_dict=metric_name_dict)

    def get_mock_reward(model_states, train_history_df, metric, stringify_states=False):
        if stringify_states:
            model_states_ = [str(x) for x in model_states]
        else:
            model_states_ = model_states
        idx_bool = np.array([train_history_df['L%i' % (i + 1)] == model_states_[i] for i in range(len(model_states_))])
        index = np.apply_along_axis(func1d=lambda x: all(x), axis=0, arr=idx_bool)
        if np.sum(index) == 0:
            idx = train_history_df['loss'].idxmax()
            return train_history_df[metric].iloc[idx]
        else:
            # mu, sd = np.mean(train_history_df[metric].iloc[index]), np.std(train_history_df[metric].iloc[index])
            return train_history_df[metric].iloc[np.random.choice(np.where(index)[0])]

    def reward_fn(model_states, *args, **kwargs):
        mock_reward = get_mock_reward(model_states, train_history_df, metric, stringify_states)
        this_reward = -(mock_reward['loss'] + Lambda * mock_reward['knowledge'])
        loss_and_metrics = [mock_reward['loss']] + [mock_reward[x] for x in metric if x != 'loss' and x != 'knowledge']
        reward_metrics = {'knowledge': mock_reward['knowledge']}
        return this_reward, loss_and_metrics, reward_metrics

    return reward_fn
