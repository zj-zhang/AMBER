import matplotlib
import matplotlib.pyplot as plt
from pandas import read_csv

matplotlib.use('Agg')

def accum_opt(data, find_min):
    tmp = []
    best = np.inf if find_min else -np.inf
    for d in data:
        if find_min and d < best:
            best = d
        elif (not find_min) and d > best:
            best = d
        tmp.append(best)
    return tmp

def plot_all_stats(file_list, baseline_file):
    dfs = [read_csv(f) for f in file_list]
    base_df = read_csv(baseline_file)
    n_run = 20

    find_min_dict = {'Knowledge': True, 'Loss': True, 'Accuracy': False}
    for key in ['Knowledge', 'Accuracy', 'Loss']:
        plt.close()
        for i, df in enumerate(dfs):
            stat = df[key]
            name = file_list[i][:-4].split('/')[-1]
            # plt.plot(sma(stat[:n_run], 20), label=name)
            plt.plot(accum_opt(stat[:n_run], find_min_dict[key]), label=name)
        plt.hlines(y=sum(base_df[key]) / float(base_df.shape[0]), xmin=0, xmax=n_run, linestyle="dashed",
                   label="baseline")
        plt.legend(loc='best')
        plt.xlabel('Number of trajectories', fontsize=16)
        plt.ylabel(key, fontsize=16)
        plt.savefig('./tmp/all_stats_{}.png'.format(key.lower()))


if __name__ == "__main__":
    plot_all_stats(["./tmp/nas.csv", "./working/bohb.csv", "./working/smac.csv", "./working/random.csv"],
                   "./working/random.csv")
