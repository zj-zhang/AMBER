"""compile metrics for trained controllers
"""

import pandas as pd
import numpy as np
import os
from collections import Counter, defaultdict
import scipy
import pickle
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

model_space = pickle.load(open('model_space.pkl', 'rb'))


def add_pseudo_cnt(tmp, model_space):
    for i in range(len(model_space)):
        for j in range(len(model_space[i])):
            tmp['layer_%i'%i][j] += 1
    return tmp


def get_controller_probs(wd, model_space):
    B = 36
    prob_dict = {}
    for b in range(B):
        df = pd.read_table(os.path.join(wd, str(b), "sampled_arcs.tsv"), index_col=0)
        tmp = df.apply(lambda x: Counter(x), axis=0).to_dict()
        tmp = add_pseudo_cnt(tmp, model_space)
        tmp2 = {}
        for layer_id in tmp:
            tot = sum(tmp[layer_id].values())
            tmp2[layer_id] = np.zeros(len(tmp[layer_id]))
            for k in tmp[layer_id]:
                tmp2[layer_id][k] = tmp[layer_id][k] / tot
        prob_dict[b] = tmp2
    return prob_dict


def recp_kl_div(pr1, pr2):
    kl_dict = {}
    assert pr1.keys() == pr2.keys()
    for k in pr1:
        kl = 0
        for layer_id in range(len(model_space)):
            kl += np.mean(scipy.special.kl_div(
                    pr1[k]['layer_%i'%layer_id], pr2[k]['layer_%i'%layer_id]) + 
                    scipy.special.kl_div(pr2[k]['layer_%i'%layer_id], pr1[k]['layer_%i'%layer_id])
                    ) / 2
        kl /= len(model_space)
        kl_dict[k] = kl
    return kl_dict


def pr_diff(pr1, pr2):
    _dict = {}
    assert pr1.keys() == pr2.keys()
    for k in pr1:
        d = 0
        for layer_id in range(len(model_space)):
            d += np.mean(np.abs(
                    pr1[k]['layer_%i'%layer_id] - pr2[k]['layer_%i'%layer_id])
                    )
        d /= len(model_space)
        _dict[k] = d
    return _dict


def get_diffs_across_runs(run_1, run_2, diff_func, model_space):
    """
    run_1 : a list of folder paths
    run_2 : a list of folder paths, or None
    diff_func : callable for mutable measurements
    """
    diffs_dict = defaultdict(list)
    run_1_probs = [get_controller_probs(x, model_space) for x in run_1]
    if run_2 is None:
        gen = itertools.combinations(run_1_probs, 2)
    else:
        run_2_probs = [get_controller_probs(x, model_space) for x in run_2]
        gen = [(a,b) for a in run_1_probs for b in run_2_probs]
    for run_a, run_b in gen:
        d = diff_func(run_a, run_b)
        for k,v in d.items():
            diffs_dict[k].append(v)
    return diffs_dict



def main():
    runs = {}
    for ds in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        runs[ds] = ['../real_deepsea/outputs/ds-%s/rep_%i/plots.train_from_hist/'%(ds, i) for i in range(4)]
    
    runs[1] = ['../real_deepsea/outputs/full/rep_%i/plots.train_from_hist/'%(i) for i in range(4)]
    # within group var
    dfs = []
    for ds in runs:
        d = get_diffs_across_runs(run_1=runs[ds], run_2=None,
                diff_func=recp_kl_div, model_space=model_space)

        a = pd.DataFrame(d).melt().rename(columns={'variable':'task', 'value':'kl-div'})
        a['ds'] = ds
        a['type'] = 'within'
        dfs.append(a)

    # match to full data
    for ds in runs:
        if ds == 1: continue
        d = get_diffs_across_runs(run_1=runs[ds], run_2=runs[1],
                diff_func=recp_kl_div, model_space=model_space)

        a = pd.DataFrame(d).melt().rename(columns={'variable':'task', 'value':'kl-div'})
        a['ds'] = ds
        a['type'] = 'between'
        dfs.append(a)


    df = pd.concat(dfs)
    print(df.groupby(['ds', 'type'])['kl-div'].mean())

    # plot
    fig, ax = plt.subplots(1,1, figsize=(8,8))
    sns.boxplot(x='ds', y='kl-div', hue='type', data=df, ax=ax)
    ax.set_xlabel('downsample ratio')
    ax.set_ylabel('5x5 rep pairwise KL-div')
    fig.savefig('downsample-KLdiv.png')





