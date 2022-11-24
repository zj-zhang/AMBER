import warnings
from .. import backend as F
import numpy as np


def unpack_data(data, unroll_generator_x=False, unroll_generator_y=False, callable_kwargs=None):
    is_generator = False
    unroll_generator = unroll_generator_x or unroll_generator_y
    if type(data) in (tuple, list):
        x, y = data[0], data[1]
    elif hasattr(data, '__iter__') or hasattr(data, '__next__'):
        x = data
        y = None
        is_generator = True
    elif callable(data):
        callable_kwargs = callable_kwargs or {}
        x, y = unpack_data(data=data(**callable_kwargs),
                unroll_generator_x=unroll_generator_x,
                unroll_generator_y=unroll_generator_y)
    else:
        raise Exception("cannot unpack data of type: %s"%type(data))
    if is_generator and unroll_generator:
        gen = data if hasattr(data, '__next__') else iter(data)
        d_ = [d for d in zip(*gen)]
        if unroll_generator_x ^ unroll_generator_y:
            if hasattr(data, "shuffle"):
                assert data.shuffle == False
        x = np.concatenate(d_[0], axis=0) if unroll_generator_x else data
        y = np.concatenate(d_[1], axis=0) if unroll_generator_y else None
    return x, y


def batchify(x, y=None, batch_size=None, shuffle=True, drop_remainder=True):
    if not type(x) is list: x = [x]
    if y is not None and type(y) is not list: y = [y]
    # assuming batch_size is axis=0
    n = len(x[0])
    idx = np.arange(n)
    if batch_size is None:
        batch_size = n
    if shuffle:
        idx = np.random.choice(idx, n, replace=False)
    for i in range(0, n, batch_size):
        tmp_x = [x_[idx[i:i + batch_size]] for x_ in x]
        if drop_remainder and tmp_x[0].shape[0] != batch_size:
            continue
        if y is not None:
            tmp_y = [y_[idx[i:i + batch_size]] for y_ in y]
            yield tmp_x, tmp_y
        else:
            yield tmp_x


def numpy_shuffle_in_unison(List):
    rng_state = np.random.get_state()
    for x in List:
        np.random.set_state(rng_state)
        np.random.shuffle(x)


def count_model_params(model_params):
    num_vars = 0
    for var in model_params:
        num_vars += np.prod([dim.value for dim in var.get_shape()])
    return num_vars


def proximal_policy_optimization_loss(curr_prediction, curr_onehot, old_prediction, old_onehotpred, rewards, advantage, clip_val, beta=None):
    rewards_ = F.squeeze(rewards, axis=1)
    advantage_ = F.squeeze(advantage, axis=1)

    entropy = 0
    r = 1
    for t, (p, onehot, old_p, old_onehot) in \
            enumerate(zip(curr_prediction, curr_onehot, old_prediction, old_onehotpred)):
        # print(t)
        # print("p", p)
        # print("old_p", old_p)
        # print("old_onehot", old_onehot)
        ll_t = F.log(F.reduce_sum(old_onehot * p))
        ll_0 = F.log(F.reduce_sum(old_onehot * old_p))
        r_t = F.exp(ll_t - ll_0)
        r = r * r_t
        # approx entropy
        entropy += -F.reduce_mean(F.log(F.reduce_sum(onehot * p, axis=1)))

    surr_obj = F.reduce_mean(F.abs(1 / (rewards_ + 1e-8)) *
                              F.minimum(r * advantage_,
                                         F.clip_by_value(r,
                                                          clip_value_min=1 - clip_val,
                                                          clip_value_max=1 + clip_val) * advantage_)
                              )
    if beta:
        # maximize surr_obj for learning and entropy for regularization
        return - surr_obj + beta * (- entropy)
    else:
        return - surr_obj


def get_kl_divergence_n_entropy(curr_prediction, curr_onehot, old_prediction, old_onehotpred):
    """compute approx
    return kl, ent
    """
    kl = []
    ent = []
    for t, (p, onehot, old_p, old_onehot) in \
            enumerate(zip(curr_prediction, curr_onehot, old_prediction, old_onehotpred)):
        # print(t, old_p, old_onehot, p, onehot)
        kl.append(F.reshape(F.get_metric('kl_div')(old_p, p), [-1]))
        ent.append(F.reshape(F.get_loss('binary_crossentropy', y_true=onehot, y_pred=p), [-1]))
    return F.reduce_mean(F.concat(kl, axis=0)), F.reduce_mean(F.concat(ent, axis=0))


def lstm(x, prev_c, prev_h, w):
    ifog = F.matmul(F.concat([x, prev_h], axis=1), w)
    i, f, o, g = F.split(ifog, 4, axis=1)
    i = F.sigmoid(i)
    f = F.sigmoid(f)
    o = F.sigmoid(o)
    g = F.tanh(g)
    next_c = i * g + f * prev_c
    next_h = o * F.tanh(next_c)
    return next_c, next_h


def stack_lstm(x, prev_c, prev_h, w):
    next_c, next_h = [], []
    for layer_id, (_c, _h, _w) in enumerate(zip(prev_c, prev_h, w)):
        inputs = x if layer_id == 0 else next_h[-1]
        curr_c, curr_h = lstm(inputs, _c, _h, _w)
        next_c.append(curr_c)
        next_h.append(curr_h)
    return next_c, next_h
