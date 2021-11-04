"""can we seed the ambient controller from the known history?

how much does the amount of history data influence?
"""

# FZZ, 2021.11.01

from zs_config import read_metadata, get_model_space_long_and_dilation
from ambient_nas import get_controller
from amber.plots import sankey

import pandas as pd
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
from collections import defaultdict
import argparse


def prep(wd, is_training=True, downsample_history=None):
    # configs
    use_ppo_loss = False
    dfeature_name_file = "./data/zero_shot_deepsea/dfeatures_ordered_list.txt"
    config_file = "./data/zero_shot_deepsea/train_feats.config_file.tsv"

    model_space, layer_embedding_sharing = get_model_space_long_and_dilation()
    session = tf.Session()
    # meta is the name, cell line, index and filepaths for each feature
    meta = read_metadata()
    # dfeature_names is the list of dfeatures
    dfeature_names = pd.read_table(dfeature_name_file, header=None)[0].to_list() 
    dfeatures = pd.read_table(config_file, index_col=0)[dfeature_names]

    controller = get_controller(
            model_space=model_space,
            session=session,
            data_description_len=len(dfeature_names),
            layer_embedding_sharing=layer_embedding_sharing,
            use_ppo_loss=use_ppo_loss,
            is_training=is_training
    )

    # read history data. store in controller
    amb_hist = pd.read_table('./data/merged_ambient_hist.tsv', index_col=0)
    # remove redundency
    amb_hist = amb_hist[['manager','reward', 'arc']].groupby(['manager', 'arc']).mean().reset_index()
    if downsample_history is not None:
        assert type(downsample_history) is float and 0 < downsample_history < 1
        amb_hist = amb_hist.sample(frac=downsample_history, replace=False)
        print("after downsampling, have %i arcs" % amb_hist.shape[0])
    for _, row in amb_hist.iterrows():
        arc = [int(x) for x in row['arc'].split(',')]
        reward = row['reward']
        manager_index = row['manager']
        desc = dfeatures.iloc[manager_index].to_numpy()[np.newaxis, :]
        controller.store(prob=[], action=arc, reward=reward, description=desc, manager_index=manager_index)
    controller.buffer.finish_path(model_space, 0, wd)
    print(controller.buffer.lt_abuffer[0].shape)
    return meta, dfeatures, model_space, controller


def custom_train(controller, wd, delta_thresh=0.001):
    epochs = 300
    old_loss = 0
    patience = 5
    p_cnt = 0
    for epoch in range(epochs):
        loss = 0
        t = 0
        # get data from buffer
        buffer = controller.buffer
        batch_size = 512
        for batch_data in buffer.get_data(bs=batch_size):
            p_batch, a_batch, ad_batch, nr_batch = \
                [batch_data[x] for x in ['prob', 'action', 'advantage', 'reward']]
            desc_batch = batch_data['description']
            feed_dict = {controller.input_arc[i]: a_batch[:, [i]]
                         for i in range(a_batch.shape[1])}
            feed_dict.update({controller.advantage: ad_batch})
            #feed_dict.update({controller.old_probs[i]: p_batch[i]
            #                  for i in range(len(controller.old_probs))})
            feed_dict.update({controller.reward: nr_batch})
            feed_dict.update({controller.data_descriptive_feature: desc_batch})
            # train
            _ = controller.session.run(controller.train_op, feed_dict=feed_dict)
            curr_loss = controller.session.run([controller.loss], feed_dict=feed_dict)
            loss += curr_loss[0] / batch_size
            t += 1

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        loss /= t
        delta = (old_loss - loss ) / np.abs(loss)
        old_loss = loss
        print(f"{current_time} - Epoch {epoch} - Loss {loss} - Delta {delta}")
        if delta < delta_thresh:
            p_cnt += 1
        else:
            p_cnt = 0
        if p_cnt >= patience:
            print("Stop due to convergence")
            break
    controller.save_weights(os.path.join(wd, 'ambient.trainedFromHist.hdf5'))


def plot_sankey(controller, model_space, dfeatures, meta, wd):
    outdir = os.path.join(wd, "plots.train_from_hist")
    os.makedirs(outdir, exist_ok=True)
    controller.load_weights(os.path.join(wd, 'ambient.trainedFromHist.hdf5'))

    dft_cols = ['TFs_and_others', 'RNA_polymerase', 'Chromatin', 'Histone']
    actions_dict = defaultdict(list)
    for i in range(len(dfeatures)):
        subdir = os.path.join(outdir, str(i))
        os.makedirs(subdir, exist_ok=True)
        act = sankey.plot_sankey(controller, model_space, save_fn=os.path.join(subdir, "sankey.png"),
                get_kwargs={"description_feature": dfeatures.iloc[i].to_numpy()[np.newaxis,:]},
                B=1000)
        actions_dict[dft_cols[dfeatures.iloc[i][dft_cols].argmax()]].append(act)
        act.to_csv(os.path.join(subdir, "sampled_arcs.tsv"), sep="\t")

    actions_dict = {k: pd.concat(v) for k,v in actions_dict.items()}
    sankey.plot_grouped_sankey(actions_dict, model_space, save_fn=os.path.join(wd, "Sankey.byTarget.png"), palette=['lightcoral', 'cornflowerblue', 'khaki', 'forestgreen'])


def test(controller, model_space, dfeatures, meta, wd, n_rep=10):
    test_feats = [
        'FEAT480',  # "YY1"   "H1-hESC"
        'FEAT282',  # "DNase" "WI-38"
        'FEAT304',  # "Pol2"  "A549"
        'FEAT144'   # "H3k4me3" "NHLF"
    ]
    # get test descriptors - below is not working
    configs = {k:configs[k] for k in configs if configs[k]['feat_name'] in test_feats}

    # train
    full_training_patience = 40
    zs_res = {}

    global_manager_trial_cnt = {k:0 for k in configs}
    global_manager_record = pd.DataFrame(columns=['manager', 'feat_name', 'amber', 'step', 'arc', 'reward'])

    for k in configs:
        feat_name = configs[k]['feat_name']
        print('-'*10); print(feat_name); print('-'*10)
        res_list = []
        manager = configs[k]['manager']
        manager._earlystop_patience = full_training_patience
        manager.verbose=0
        for i in range(n_rep):
            arc, prob = controller.get_action(np.expand_dims(configs[k]['dfeatures'],0))
            reward, _ = manager.get_rewards(trial=global_manager_trial_cnt[k], model_arc=arc)
            global_manager_trial_cnt[k] += 1
            print(reward)
            res_list.append(reward)
            global_manager_record = global_manager_record.append({
                'manager': k,
                'feat_name': feat_name,
                'amber': 'zs',
                'step': i,
                'arc': ','.join([str(a) for a in arc]),
                'reward': reward
            }, ignore_index=True)
        zs_res[k] = res_list



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wd', type=str, required=True)
    ap.add_argument('--analysis', choices=['train', 'reload'], type=str, required=True)
    ap.add_argument('--downsample', default=None, type=float, required=False)
    args = ap.parse_args()
    os.makedirs(args.wd, exist_ok=True)

    if args.analysis == "train":
        meta, dfeatures, model_space, controller = prep(args.wd, is_training=True, downsample_history=args.downsample)
        custom_train(controller, args.wd)
    else:
        meta, dfeatures, model_space, controller = prep(args.wd, is_training=False, downsample_history=args.downsample)
        plot_sankey(controller=controller, model_space=model_space,
                dfeatures=dfeatures, meta=meta, wd=args.wd)


if __name__ == "__main__":
    main()
