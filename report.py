from PSO import particle_swarm_optimization
from PSO import PSOConfig
import numpy as np
from numpy.random import seed
from fitness_function import FuncInf
from fitness_function import rastrigin_function
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def report(id):
    N = [128, 256, 512, 1024, 2048]
    selection_mode = [True, False]
    combinations = np.stack(np.meshgrid(N, selection_mode), -1).reshape((-1, 2))
    seeds = [id + i for i in range(10)]
    user_config = PSOConfig
    func_inf = FuncInf("Rastrigin function", 10, True, (-5.12, 5.12), None, rastrigin_function)

    print("## Writing...")
    with open("fitness.txt", "w+") as f:
        # f.write("N\tM\tS\t\tF\t\tP\n")
        f.write("N\tF\n")   # write fitness function each popsize
        for comb in combinations:
            # best_sol = [0, "", 0, np.Inf, None]
            for s in seeds:
                user_config.NUM_PARTICLES = comb[0]
                user_config.RING_TOPO = comb[1]
                user_config.NUM_GENS = 1000000 // (2*comb[0])
                seed(s)
                result = particle_swarm_optimization(user_config, func_inf, False)
                # mode = "RT" if comb[1] == True else "ST"
                f.write("{}\t{}\n".format(comb[0], result[1]))      # write fitness function each popsize
                # if best_sol[3] > result[1]:
                #     best_sol = [comb[0], mode, s, result[1], result[0]]

            # f.write("{}\t{}\t{}\t{}\t\t{}\n".format(best_sol[0], best_sol[1], best_sol[2], best_sol[3], best_sol[4]))
                
    print("## Done!")

# report(18521578)

file_name = "report/fitness.txt"

def read_file(file_name):
    data = []
    with open(file_name) as f:
        for idx, line in enumerate(f):
            if idx != 0:
                data.append(line.replace("\n", "").split("\t"))

    data = np.array(data, dtype=np.float)
        
    return data

from copy import deepcopy
data = read_file(file_name)
def sort_data(data):
    rt_data = data[:50]
    st_data = data[50:]

    sorted_data = None
    for i in range(0, 50, 10):
        if i == 0:
            sorted_data = np.vstack((rt_data[i:i+10], st_data[i:i+10]))
        else:
            stacked_data = np.vstack((rt_data[i:i+10], st_data[i:i+10]))
            sorted_data = np.vstack((sorted_data, stacked_data))

        
    return sorted_data

sorted_data = sort_data(data)

def plot_data(data):
    fig, axes = plt.subplots(3, 2)
    plt.delaxes(ax=axes[2, 1])

    sns.distplot(data[:10, 1], hist=False, bins=3, kde=True, hist_kws={'edgecolor':'black'}, ax=axes[0, 0])
    sns.distplot(data[10:20, 1], hist=False, bins=3, kde=True, hist_kws={'edgecolor':'black'}, ax=axes[0, 0])
    axes[0, 0].set_title("Pop Size: {}".format(data[0, 0]))


    sns.distplot(data[20:30, 1], hist=False, bins=3, kde=True, hist_kws={'edgecolor':'black'}, ax=axes[0, 1])
    sns.distplot(data[30:40, 1], hist=False, bins=3, kde=True, hist_kws={'edgecolor':'black'}, ax=axes[0, 1])
    axes[0, 1].set_title("Pop Size: {}".format(data[20, 0]))


    sns.distplot(data[40:50, 1], hist=False, bins=3, kde=True, hist_kws={'edgecolor':'black'}, ax=axes[1, 0])
    sns.distplot(data[50:60, 1], hist=False, bins=3, kde=True, hist_kws={'edgecolor':'black'}, ax=axes[1, 0])
    axes[1, 0].set_title("Pop Size: {}".format(data[40, 0]))


    sns.distplot(data[60:70, 1], hist=False, bins=3, kde=True, hist_kws={'edgecolor':'black'}, ax=axes[1, 1])
    sns.distplot(data[70:80, 1], hist=False, bins=3, kde=True, hist_kws={'edgecolor':'black'}, ax=axes[1, 1])
    axes[1, 1].set_title("Pop Size: {}".format(data[60, 0]))


    sns.distplot(data[80:90, 1], hist=False, bins=3, kde=True, hist_kws={'edgecolor':'black'}, ax=axes[2, 0])
    sns.distplot(data[90:100, 1], hist=False, bins=3, kde=True, hist_kws={'edgecolor':'black'}, ax=axes[2, 0])
    axes[2, 0].set_title("Pop Size: {}".format(data[80, 0]))

    for ax in axes.flat:
        ax.set(xlabel='Fitness', ylabel='Density')
        ax.set()

    lines = tuple(axes[0, 0].get_lines())
    fig.legend(lines, ("Ring Topology", "Star Topology"), loc='lower right')
    fig.tight_layout(pad=1.0)
    
    plt.show()

from numpy import round
def get_mean_std(data):
    result = None
    for i in range(0, len(data), 10):
        mean = round(np.mean(data[i:i+10, 1]), 4)
        std = round(np.std(data[i:i+10, 1]), 4)
        if i == 0:
            result = np.hstack((mean, std))
        else:
            stacked_items = np.hstack((mean, std))
            result = np.vstack((result, stacked_items))
    return result
# plot_data(sorted_data)
# print(sorted_data)
ring_dt = get_mean_std(data[:50])
star_dt = get_mean_std(data[50:])

ring_df = pd.DataFrame(ring_dt, index=[128, 256, 512, 1024, 2048], columns=["Mean", "Standard Deviation"])
star_df = pd.DataFrame(star_dt, index=[128, 256, 512, 1024, 2048], columns=["Mean", "Standard Deviation"])

ring_df.to_csv("ring_mean_std.csv")
star_df.to_csv("star_mean_std.csv")
