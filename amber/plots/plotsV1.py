# -*- coding: UTF-8 -*-

from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import json

matplotlib.use('Agg')


def reset_style():
    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['axes.titlesize'] = 14
    rcParams['axes.labelsize'] = 14
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 8
    rcParams['xtick.labelsize'] = 14
    rcParams['ytick.labelsize'] = 14
    rcParams['legend.fontsize'] = 14


reset_style()


def reset_plot(width_in_inches=4.5,
               height_in_inches=4.5):
    dots_per_inch = 200
    plt.close('all')
    return plt.figure(
        figsize=(width_in_inches, height_in_inches),
        dpi=dots_per_inch)


def plot_training_history(history, par_dir):
    # print(history.history.keys())
    reset_plot()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    plt.savefig(os.path.join(par_dir, 'loss.png'))
    plt.gcf().clear()


def plot_controller_performance(controller_hist_file, metrics_dict, save_fn=None, N_sma=10):
    """
    Example:
        controller_hist_file = 'train_history.csv'
        metrics_dict = {'acc': 0, 'loss': 1, 'knowledge': 2}
    """
    # plt.clf()
    reset_plot()
    plt.grid(b=True, linestyle='--', linewidth=0.8)
    df = pd.read_csv(controller_hist_file, header=None)
    #assert df.shape[0] > N_sma
    if df.shape[0] <= N_sma:
        N_sma = 1
    df.columns = ['trial', 'loss_and_metrics', 'reward'] + ['layer_%i' % i for i in range(df.shape[1] - 3)]
    # N_sma = 20

    plot_idx = []
    for metric in metrics_dict:
        metric_idx = metrics_dict[metric]
        df[metric] = [float(x.strip('\[\]').split(',')[metric_idx]) for x in df['loss_and_metrics']]
        df[metric + '_SMA'] = np.concatenate(
            [[None] * (N_sma - 1), np.convolve(df[metric], np.ones((N_sma,)) / N_sma, mode='valid')])
        # df[metric+'_SMA'] /= np.max(df[metric+'_SMA'])
        plot_idx.append(metric + '_SMA')

    ax = sns.scatterplot(data=df[plot_idx])
    ax.set_xlabel('Steps')
    ax.set_ylabel('Simple Moving Average')
    if save_fn:
        plt.savefig(save_fn)
    else:
        plt.show()


def plot_environment_entropy(entropy_record, save_fn):
    '''plot the entropy change for the state-space
    in the training environment. A smaller entropy
    indicates converaged controller.
    '''
    # plt.clf()
    reset_plot()
    ax = sns.lineplot(x=np.arange(len(entropy_record)), y=entropy_record)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Entropy')
    plt.savefig(save_fn)


def sma(data, window=10):
    if len(data) < window:
        return np.array(data)
    else:
        return np.concatenate([np.cumsum(data[:window - 1]) / np.arange(1, window),
                           np.convolve(data, np.ones((window,)) / window, mode='valid')])


def plot_action_weights(working_dir):
    save_path = os.path.join(working_dir, 'weight_data.json')
    if os.path.exists(save_path):
        with open(save_path, 'r+') as f:
            df = json.load(f)
        # plt.clf()
        for layer in df:
            reset_plot(width_in_inches=6, height_in_inches=4.5)
            ax = plt.subplot(111)
            for layer_name, data in df[layer]['operation'].items():
                d = np.array(data)
                if d.shape[0] >= 2:
                    avg = np.apply_along_axis(np.mean, 0, d)
                else:
                    avg = np.array(d).reshape(d.shape[1])
                ax.plot(avg, label=layer_name)
                if d.shape[0] >= 6:
                    std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
                    min_, max_ = avg - 1.96 * std, avg + 1.96 * std
                    ax.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)

            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            ax.set_xlabel('Number of steps')
            ax.set_ylabel('Weight of layer type')
            # plt.title('Weight of Each Layer Type at Layer {}'.format(layer[-1]))
            plt.savefig(os.path.join(working_dir, 'weight_at_layer_{}.png'.format(layer.strip('L'))),
                        bbox_inches='tight')
    else:
        raise IOError('File does not exist')


def plot_wiring_weights(working_dir, with_input_blocks, with_skip_connection):
    if (not with_input_blocks) and (not with_skip_connection):
        return
    save_path = os.path.join(working_dir, 'weight_data.json')
    if os.path.exists(save_path):
        with open(save_path, 'r+') as f:
            df = json.load(f)
        for layer in df:
            # reset the plots, size and content
            reset_plot(width_in_inches=6, height_in_inches=4.5)
            if with_input_blocks:
                fig1 = plt.figure()
                ax1 = fig1.add_subplot(111)
                for layer_name, data in df[layer]['input_blocks'].items():
                    d = np.array(data)
                    if d.shape[0] >= 2:
                        avg = np.apply_along_axis(np.mean, 0, d)
                    else:
                        avg = np.array(d).reshape(d.shape[1])
                    ax1.plot(avg, label=layer_name)
                    if d.shape[0] >= 6:
                        std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
                        min_, max_ = avg - 1.96 * std, avg + 1.96 * std
                        ax1.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)

                box = ax1.get_position()
                ax1.set_position([box.x0, box.y0, box.width * 0.75, box.height])

                # Put a legend to the right of the current axis
                ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                ax1.set_xlabel('Number of steps')
                ax1.set_ylabel('Weight of input blocks')
                # plt.title('Weight of Each Layer Type at Layer {}'.format(layer[-1]))
                fig1.savefig(os.path.join(working_dir, 'inp_at_layer_{}.png'.format(layer.strip('L'))),
                             bbox_inches='tight')

            if with_skip_connection and int(layer.strip('L')) > 0:
                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111)
                for layer_name, data in df[layer]['skip_connection'].items():
                    d = np.array(data)
                    if d.shape[0] >= 2:
                        avg = np.apply_along_axis(np.mean, 0, d)
                    else:
                        avg = np.array(d).reshape(d.shape[1])
                    ax2.plot(avg, label=layer_name)
                    if d.shape[0] >= 6:
                        std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
                        min_, max_ = avg - 1.96 * std, avg + 1.96 * std
                        ax2.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)

                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0, box.width * 0.75, box.height])

                # Put a legend to the right of the current axis
                ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                ax2.set_xlabel('Number of steps')
                ax2.set_ylabel('Weight of skip connection')
                fig2.savefig(os.path.join(working_dir, 'skip_at_layer_{}.png'.format(layer.strip('L'))),
                             bbox_inches='tight')
    else:
        raise IOError('File does not exist')


def plot_stats2(working_dir):
    save_path = os.path.join(working_dir, 'nas_training_stats.json')
    if os.path.exists(save_path):
        with open(save_path, 'r') as f:
            df = json.load(f)
        # plt.clf()
        reset_plot()
        # ax = plt.subplot(111)
        data = df['Loss']
        d = np.stack(list(map(lambda x: sma(x), np.array(data))), axis=0)
        avg = np.apply_along_axis(np.mean, 0, d)
        ax = sns.lineplot(x=np.arange(1, len(avg) + 1), y=avg,
                          color='b', label='Loss', legend=False)
        if d.shape[0] >= 6:
            std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
            min_, max_ = avg - 1.96 * std, avg + 1.96 * std
            ax.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)

        data = df['Knowledge']
        if np.array(data).shape[1] > 0: # if have data
            ax2 = ax.twinx()
            d = np.stack(list(map(lambda x: sma(x), np.array(data))), axis=0)
            avg = np.apply_along_axis(np.mean, 0, d)
            sns.lineplot(x=np.arange(1, len(avg) + 1), y=avg,
                        color='g', label='Knowledge', ax=ax2, legend=False)
            if d.shape[0] >= 6:
                std = np.apply_along_axis(np.std, 0, d) / np.sqrt(d.shape[0])
                min_, max_ = avg - 1.96 * std, avg + 1.96 * std
                ax2.fill_between(range(avg.shape[0]), min_, max_, alpha=0.2)
            ax2.set_ylabel('Knowledge')

        ax.figure.legend()
        ax.set_xlabel('Number of steps')
        ax.set_ylabel('Loss')
        plt.savefig(os.path.join(working_dir, 'nas_training_stats.png'), bbox_inches='tight')
    else:
        raise IOError('File not found')

