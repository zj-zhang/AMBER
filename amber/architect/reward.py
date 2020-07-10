# -*- coding: UTF-8 -*-

'''Reward function for weighted sum of Loss and Knowledge
ZZJ
Nov. 18, 2018
'''

import numpy as np
from .common_ops import unpack_data
from ..utils.io import read_history


class Reward:
    def __init__(self, *args, **kwargs):
        raise NotImplemented("Abstract method.")

    def __call__(self, model, data, *args, **kwargs):
        raise NotImplemented("Abstract method.")


class KnowledgeReward(Reward):
    def __init__(self, knowledge_function, Lambda, loss_c=None, knowledge_c=None):
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
    def __init__(self, *args, **kwargs):
        self.knowledge_function = None
        self.c = 1.

    def __call__(self, model, data, *args, **kwargs):
        X, y = unpack_data(data)
        loss_and_metrics = model.evaluate(X, y)
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
    # return self.c/L, loss_and_metrics, None


class LossAucReward(Reward):
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

    def __call__(self, model, data, *args, **kwargs):
        X, y = unpack_data(data, unroll_generator_y=True)
        if self.pred is None:
            pred = model.predict(X)
        else:
            pred = self.pred
        auc_list = []
        if type(y) is not list:
            y = [y]
        if type(pred) is not list:
            pred = [pred]
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
        L = np.nanmean(np.concatenate(auc_list, axis=0))
        self.auc_list = auc_list
        # metrics = [np.median(x) for x in auc_list]
        # loss_and_metrics = [L] + metrics
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
        reward = L + self.Lambda * K
        loss_and_metrics = [L]
        reward_metrics = {'knowledge': K}
        return reward, loss_and_metrics, reward_metrics


def MockReward(train_history_list, metric, stringify_states, metric_name_dict, Lambda=1.):
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
