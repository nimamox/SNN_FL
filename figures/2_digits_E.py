import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
plt.rc('font', family='serif')
plt.rc('font', serif='Times New Roman')

LOCAL_ITERS = [1, 10, 100]
RESULTS_FILE = ['E{}_M10_s100'.format(x) for x in LOCAL_ITERS]
matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = "stix"


DATASET_NAMES = ['2_digits_per_client']

local_iters = [100, 10, 1]
for DATASET in DATASET_NAMES:
    result_dict = dict()
    dataset_path = f'../result/{DATASET}'
    files = sorted(os.listdir(dataset_path))
    for file in files:
        for i, rfile in enumerate(RESULTS_FILE):
            if rfile in file:
                with open(os.path.join(dataset_path, file, 'metrics.json'), 'r') as load_f:
                    load_dict = eval(json.load(load_f))
                result_dict[str(LOCAL_ITERS[i])] = load_dict['loss_on_train_data']

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    COLORS = {"1": colors[0],
              "10": colors[1],
              "100": colors[2]}
    LABELS = {"1": r"$E=1$",
              "10": r"$E=10$",
              "100": r"$E=100$"}
    LINESTYLE = {"1": '-',
                 "10": '--',
                 "100": '-.'}

    plt.figure(figsize=(4, 3))
    i = 0
    for wn, stat in result_dict.items():
        plt.plot(np.arange(len(stat)) * local_iters[i], np.array(stat), linewidth=1.0, color=COLORS[wn], label=LABELS[wn], linestyle=LINESTYLE[wn])
        i += 1

    plt.grid(True)
    # 0: ‘best', 1: ‘upper right', 2: ‘upper left', 3: ‘lower left'
    plt.legend(loc=0, borderaxespad=0., prop={'size': 10})
    plt.xlabel('Iteration', fontdict={'size': 10})
    plt.ylabel('Loss', fontdict={'size': 10})
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    # before = 15
    # a = plt.axes([0.31, 0.6, .3, .3])
    # for wn, stat in result_dict.items():
    #     plt.plot(np.arange(len(stat))[-before:-5], np.array(stat)[-before:-5], linewidth=1.0, color=COLORS[wn], label=LABELS[wn])

    # plt.xticks(fontsize=7)
    # plt.yticks(fontsize=7)
    fig = plt.gcf()
    fig.savefig(f'2_digits_E.pdf')

