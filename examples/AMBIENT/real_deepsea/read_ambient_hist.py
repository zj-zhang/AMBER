"""read in previous train histories for 
"""

import pandas as pd


def read_single_hist(fp, group_id=0):
    d=pd.read_csv(fp,  header=None)
    dd = pd.DataFrame.from_dict(dict(
        manager=[x.split('-')[0] for x in d[0]],
        trial=[x.split('-')[1] for x in d[0]],
        reward=d[2],
        arc=[','.join([str(x) for x in r[3::]]) for _, r in d.iterrows()]
    ))
    dd['group'] = group_id
    return dd


def group_uniqueness(data):
    cnt_df = data[['manager', 'arc', 'group']].groupby(['manager', 'group']).nunique()
    tot_df = data[['manager', 'arc', 'group']].groupby(['manager', 'group']).size()
    cnt_df['uniqueness'] = cnt_df['arc'] / tot_df
    cnt_df = cnt_df.reset_index()
    return cnt_df





fp_dict = {
    'old_0': '~/workspace/src/AMBER-ZeroShot/examples/outputs/new_20200919/long_and_dilation.ppo.0/train_history.csv',
    'old_1': '~/workspace/src/AMBER-ZeroShot/examples/outputs/new_20200919/long_and_dilation.ppo.1/train_history.csv',
    'old_2': '~/workspace/src/AMBER-ZeroShot/examples/outputs/new_20200919/long_and_dilation.ppo.2/train_history.csv',
    'new_0': '/mnt/ceph/users/zzhang/amber_github/AMBER/examples/outputs/new_20200919/long_and_dilation.ppo.re0/train_history.csv',
    'new_1': '/mnt/ceph/users/zzhang/amber_github/AMBER/examples/outputs/new_20200919/long_and_dilation.ppo.re1/train_history.csv',
    'new_2': '/mnt/ceph/users/zzhang/amber_github/AMBER/examples/outputs/new_20211022/long_and_dilation.rl.3/train_history.csv',
    'p0': '/mnt/ceph/users/zzhang/amber_github/AMBER/examples/outputs/new_20211022/long_and_dilation.ppo.0/train_history.csv',
    'p1': '/mnt/ceph/users/zzhang/amber_github/AMBER/examples/outputs/new_20211022/long_and_dilation.ppo.1/train_history.csv',
    'p2': '/mnt/ceph/users/zzhang/amber_github/AMBER/examples/outputs/new_20211022/long_and_dilation.ppo.2/train_history.csv',
    'r0': '/mnt/ceph/users/zzhang/amber_github/AMBER/examples/outputs/new_20211030/long_and_dilation.rl.0/train_history.csv',
    'r1': '/mnt/ceph/users/zzhang/amber_github/AMBER/examples/outputs/new_20211030/long_and_dilation.rl.1/train_history.csv',
    'r2': '/mnt/ceph/users/zzhang/amber_github/AMBER/examples/outputs/new_20211030/long_and_dilation.rl.2/train_history.csv',
}

ds = [read_single_hist(v,k) for k,v in fp_dict.items()]
data = pd.concat(ds)

data.to_csv('./data/merged_ambient_hist.tsv', sep="\t")




