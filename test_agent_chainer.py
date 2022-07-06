# coding=UTF-8
import os
import gym
from collections import defaultdict

import numpy as np
import random

from gym_malware import sha256_holdout, MAXTURNS
from gym_malware.envs.controls import manipulate2 as manipulate
from gym_malware.envs.utils import interface, interface_v2, pefeatures

ACTION_LOOKUP = {i: act for i, act in enumerate(
    manipulate.ACTION_TABLE.keys())}


num_test_episodes = 500

# 动作评估


def evaluate(action_function):
    success = []
    misclassified = []
    success_mean_len = 0
    for _ in range(num_test_episodes):
        action_len = 0
        sha256 = random.sample(sha256_holdout)
        success_dict = defaultdict(list)
        bytez = interface_v2.fetch_file(sha256)
        label = interface.get_label_local(bytez)
        if label == 0.0:
            misclassified.append(sha256)
            continue  # already misclassified, move along
        for _ in range(MAXTURNS):
            action_len += 1
            action = action_function(bytez)
            print(action)
            success_dict[sha256].append(action)
            bytez = manipulate.modify_without_breaking(bytez, [action])
            new_label = interface.get_label_local(bytez)
            if new_label == 0.0:
                success_mean_len += action_len
                success.append(success_dict)
                break
    success_mean_len /= len(success)
    # evasion accuracy is len(success) / len(sha256_holdout)
    return success, misclassified, success_mean_len
