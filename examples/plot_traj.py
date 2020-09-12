import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from amber.plots._plotsV1 import sma


def plot_zs_hist(hist_fp, config_fp, save_fp, zoom_first_n=None):
    zs_hist = pd.read_table(hist_fp, header=None, sep=",")
    configs = pd.read_table(config_fp)
    zs_hist['task'] = zs_hist[0].apply(lambda x: configs.loc[int(x.split('-')[0])]['feat_name'])
    zs_hist['trial'] = zs_hist[0].apply(lambda x: int(x.split('-')[1]))
    zs_hist['auc'] = sma(zs_hist[2], window=20)
    zs_hist = zs_hist.drop([0,1,2,3,4,5,6], axis=1)
    if zoom_first_n is not None:
        zs_hist = zs_hist.loc[zs_hist.trial <= zoom_first_n]
    ax = sns.lineplot(x="trial", y="auc", hue="task", data=zs_hist)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(save_fp)
    plt.close()


def plot_single_run(feat_dirs, save_fp, zoom_first_n=None):
    dfs = []
    for d in feat_dirs:
        hist = pd.read_table(os.path.join(d, "train_history.csv"), header=None, sep=",")
        hist['task'] = os.path.basename(d)
        hist['trial'] = hist[0]
        hist['auc'] = sma(hist[2], window=20)
        hist = hist.drop([0,1,2,3,4,5,6], axis=1)
        if zoom_first_n is not None:
            hist = hist.loc[hist.trial <= zoom_first_n]
        dfs.append(hist)
    df = pd.concat(dfs)
    ax = sns.lineplot(x="trial", y="auc", hue="task", data=df)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig(save_fp)
    plt.close()


plot_zs_hist("./outputs/zs_50.2/train_history.csv", "data/zero_shot/train_feats.config_file.8_random_feats.tsv", "zs_hist.png", zoom_first_n=300)
plot_zs_hist("./outputs/zero_shot_deepsea/train_history.csv", "data/zero_shot_deepsea/debug_feats.config.4_cats.tsv", "zs_deepsea.png", zoom_first_n=300)

feat_dirs = ["./outputs/%s"%x for x in os.listdir("./outputs/") if x.startswith("FEAT")]
plot_single_run(feat_dirs, "single_runs.png", zoom_first_n=300)


