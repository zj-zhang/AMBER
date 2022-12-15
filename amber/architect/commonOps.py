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


def batchify(x, y=None, batch_size=None, shuffle=True, drop_remainder=False):
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


# def numpy_shuffle_in_unison(List):
#     rng_state = np.random.get_state()
#     for x in List:
#         np.random.set_state(rng_state)
#         np.random.shuffle(x)


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

