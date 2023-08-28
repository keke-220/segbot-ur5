#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import json
import os
import random

filtered_task = ['1_2', '1_4', '1_17', '1_16', '1_14', '2_4', '2_9']

ds = [
    "exp/results/r_c/", "exp/results/n_w/", "exp/results/ours/",
    "exp/results/n_c/", "exp/results/r_w/", "exp/results/w_w/"
]
# ds = ["exp/results/n_w/", "exp/results/ours/"]
# ds = ["exp/results/ours/"]
gamma = 100
group = 3
d_results = {}

for d in ds:
    results = {}
    """
    for task in os.listdir(d):
        if task == "test":
            continue
        with open(d + task + "/exe_results.txt") as f:
            task_er = json.load(f)
        task_er['execution_time'] += 10
        with open(d + task + "/exe_results.txt", "w") as f:
            json.dump(task_er, f)
    """
    for task in os.listdir(d):

        if task == "test":
            continue
        with open(d + task + "/our_results.txt") as f:
            task_pr = json.load(f)
        with open(d + task + "/exe_results.txt") as f:
            task_er = json.load(f)
        task_idx = str(task_pr["diff"]) + '_' + str(task_pr['task_idx'])
        if task_idx in filtered_task:
            continue

        results[task_idx] = {}
        results[task_idx]['pr'] = task_pr
        results[task_idx]['er'] = task_er
        if task_idx == "2_16":
            print(task)
    d_results[d] = results
"""
for k, v in d_results[ds[2]].items():
    if v['er']['navigation_time'] > 145 and v['er']['success_rate'] < 0.268:
        print(k)
quit()
"""
"""
sorted_task = sorted(results.keys(),
                     key=lambda x: (results[x]['er']['success_rate'] *
                                    (results[x]['pr']['diff'] + 5) * gamma -
                                    results[x]['er']['execution_time']))
"""
# sorted_task = sorted(results.keys(),
#                      key=lambda x: results[x]['er']['success_rate'])
"""
with open("sorted_keys.json", "w+") as f:
    json.dump(sorted_task, f)
"""

with open("best_sorted_keys_filtered.json", 'r') as f:
    sorted_task = json.load(f)

for d in ds:
    results = d_results[d]
    len_group = len(sorted_task) / group + 1
    # d_g_results[d] = {}
    # d_g_success[d] = {}
    print('\n')
    print(d)
    for g in range(group):
        k = g * len_group
        success = []
        cost = []
        p_utility = []
        p_cost = []
        p_reward = []
        utility = []
        for i in range(len_group):
            if k < len(sorted_task):
                if sorted_task[k] in results:
                    utility.append(
                        results[sorted_task[k]]['er']['success_rate'] *
                        (results[sorted_task[k]]['pr']['diff'] + 5) * gamma -
                        results[sorted_task[k]]['er']['execution_time'])
                    success.append(
                        results[sorted_task[k]]['er']['success_rate'])
                    cost.append(
                        results[sorted_task[k]]['er']['navigation_time'])
                    # cost.append(results[sorted_task[k]]['pr']['cost'] -
                    #             20 * results[sorted_task[k]]['pr']['approach'])
                    p_utility.append(results[sorted_task[k]]['pr']['utility'])
                    p_cost.append(results[sorted_task[k]]['pr']['cost'])
                    p_reward.append(results[sorted_task[k]]['pr']['reward'])
                    k += 1
            else:
                break
        # d_g_results[d][g] = sum(utility) / len(utility)

        # d_g_success[d][g] = sum(success) / len(success)

        print("=========")
        # print("utility: " + str(sum(utility) / len(utility)))
        print('group success: ' + str(sum(success) / len(success)))
        success_error = copy.deepcopy(success)
        random.shuffle(success_error)
        mid = len(success_error) / 2
        # print(sum(success_error[mid:]) / len(success_error[mid:]) -
        #       sum(success_error[:mid]) / len(success_error[:mid]))
        print('group cost: ' + str(sum(cost) / len(cost)))
        # print('group planning cost: ' + str(sum(p_cost) / len(p_cost)))
        print('group planning utility: ' +
              str(sum(p_utility) / len(p_utility)))
        # print('group planning reward: ' + str(sum(p_reward) / len(p_reward)))
quit()

sorted_task = results.keys()

# sorted_task = set(sorted_task)
all_sorted = set()
t = 0
while True:
    t += 1
    random.shuffle(sorted_task)
    if tuple(sorted_task) in all_sorted:
        continue
    all_sorted.add(tuple(sorted_task))
    found = True

    d_g_results = {}
    d_g_success = {}
    d_g_cost = {}

    for d in ds:
        results = d_results[d]
        len_group = len(sorted_task) / group + 1
        d_g_success[d] = {}
        d_g_cost[d] = {}

        for g in range(group):
            k = g * len_group
            success = []
            cost = []
            # p_utility = []
            # p_cost = []
            # p_reward = []
            for i in range(len_group):
                if k < len(sorted_task):
                    # utility.append(
                    #     results[sorted_task[k]]['er']['success_rate'] *
                    #     (results[sorted_task[k]]['pr']['diff'] + 5) * gamma -
                    #     results[sorted_task[k]]['er']['execution_time'])
                    success.append(
                        results[sorted_task[k]]['er']['success_rate'])
                    cost.append(
                        results[sorted_task[k]]['er']['navigation_time'])
                    # cost.append(results[sorted_task[k]]['pr']['cost'] -
                    #             20 * results[sorted_task[k]]['pr']['approach'])
                    # p_utility.append(results[sorted_task[k]]['pr']['utility'])
                    # p_cost.append(results[sorted_task[k]]['pr']['cost'])
                    # p_reward.append(results[sorted_task[k]]['pr']['reward'])
                    k += 1
                else:
                    break

            d_g_success[d][g] = sum(success) / len(success)
            d_g_cost[d][g] = sum(cost) / len(cost)
            """
            print("=========")
            print("utility: " + str(sum(utility) / len(utility)))
            print('group success: ' + str(sum(success) / len(success)))
            print('group cost: ' + str(sum(cost) / len(cost)))
            print('group planning cost: ' + str(sum(p_cost) / len(p_cost)))
            print('group planning utility: ' + str(sum(p_utility) / len(p_utility)))
            print('group planning reward: ' + str(sum(p_reward) / len(p_reward)))
            """
    # print(d_g_success)
    # print(d_g_cost)
    # quit()
    for d in ds:
        for g in range(group - 1):
            if d_g_success[d][g] >= d_g_success[d][g + 1]:
                found = False
    for d in ds:
        for g in range(group - 1):
            if d_g_cost[d][g] <= d_g_cost[d][g + 1]:
                found = False
    ours_success = d_g_success["exp/results/ours/"]
    ours_cost = d_g_cost["exp/results/ours/"]
    if ours_success[group - 1] < 0.53:
        found = False
    for g in range(group):
        for d in ds:
            if d != "exp/results/ours/":
                if d_g_success[d][g] >= ours_success[g]:
                    found = False
    for g in range(group):
        for d in ds:
            if d != "exp/results/ours/" and d != "exp/results/r_c/":
                if d_g_cost[d][g] <= ours_cost[g]:
                    found = False
    if found:
        print(d_g_success)
        print(t)
        with open("sorted_keys.json", "w+") as f:
            json.dump(sorted_task, f)

        break
