#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np


def main(argv):
    measure = 'Task Completion Rate'
    measure = 'Robot Execution Time'
    x = ['Easy', 'Moderate', 'Difficult']

    strategy = ['Ours', 'GROP']
    # x = [1, 2, 3]
    if measure == 'Task Completion Rate':
        y = {}
        e = {}
        y['Ours'] = [0.5204, 0.3608, 0.2878]
        e['Ours'] = [0.015, 0.025, 0.015]
        y['GROP'] = [0.4101, 0.2981, 0.2579]
        e['GROP'] = [0.0075, 0.02, 0.015]

        ymin = {}
        ymax = {}
        #generating plots:
        for s in strategy:
            ymax[s] = []
            ymin[s] = []
            for i, ee in enumerate(e[s]):
                ymax[s].append(y[s][i] + ee)
                ymin[s].append(y[s][i] - ee)
    else:
        y = {}
        e = {}
        y['Ours'] = [107.9, 118.68, 131.25]
        e['Ours'] = [5.35, 2.78, 3.64]
        y['GROP'] = [138.06, 141.22, 157.07]
        e['GROP'] = [3.33, 6.24, 5.13]

        ymin = {}
        ymax = {}
        #generating plots:
        for s in strategy:
            ymax[s] = []
            ymin[s] = []
            for i, ee in enumerate(e[s]):
                ymax[s].append(y[s][i] + ee)
                ymin[s].append(y[s][i] - ee)

    markers = itertools.cycle(('v', '*', 'o'))
    slabel = itertools.cycle(('Ours', 'GROP', 'PETLON'))
    color = itertools.cycle(('#ff8243', '#c043ff', '#82ff43'))
    #plt.style.use('seaborn-whitegrid')
    #Combine success rate and cost to a one-dimention graph
    fig, ax = plt.subplots()
    plt.grid(True)
    #plt.ylim(65, 87.5)
    #plt.xlim(-0.1, 5)
    plt.xlabel("Task Difficulty", fontsize=14)
    if measure == "Task Completion Rate":
        plt.ylabel(measure + " (%)", fontsize=14)
    #plt.title("Accuracy", fontsize = 18)
    for s in strategy:
        color1 = next(color)

        ax.plot(range(len(x)),
                y[s],
                label=next(slabel),
                color=color1,
                marker=next(markers),
                mec='black',
                markersize=7)
        x_temp = ['Difficult', 'Easy', 'Moderate']

        ax.fill_between(x_temp, ymax[s], ymin[s], alpha=0.5, color=color1)
    plt.xticks(range(len(x)), x)
    if measure == "Task Completion Rate":
        ax.legend(loc='upper right', numpoints=1, fontsize=14)
    else:
        ax.legend(loc='upper left', numpoints=1, fontsize=14)
    fig.savefig(measure + '.png')


if __name__ == '__main__':
    main(sys.argv)
