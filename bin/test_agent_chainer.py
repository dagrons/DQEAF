from collections import defaultdict

import numpy as np

from gym_malware import sha256_holdout, MAXTURNS
from gym_malware.envs.controls import manipulate2 as manipulate
from gym_malware.envs.utils import interface, pefeatures

ACTION_LOOKUP = {i: act for i, act in enumerate(manipulate.ACTION_TABLE.keys())}

from bin.train_agent_chainer import create_ddqn_agent
import gym
import gym_malware
import os


# 动作评估
def evaluate(action_function):
    success = []
    misclassified = []
    for sha256 in sha256_holdout:
        success_dict = defaultdict(list)
        bytez = interface.fetch_file(sha256)
        label = interface.get_label_local(bytez)
        if label == 0.0:
            misclassified.append(sha256)
            continue  # already misclassified, move along
        for _ in range(MAXTURNS):
            action = action_function(bytez)
            print(action)
            success_dict[sha256].append(action)
            bytez = manipulate.modify_without_breaking(bytez, [action])
            new_label = interface.get_label_local(bytez)
            if new_label == 0.0:
                success.append(success_dict)
                break
    return success, misclassified  # evasion accuracy is len(success) / len(sha256_holdout)


# 获取保存的模型
def get_latest_model_from(basedir):
    dirs = os.listdir(basedir)
    lastmodel = -1
    for d in dirs:
        try:
            if int(d) > lastmodel:
                lastmodel = int(d)
        except ValueError:
            continue

    assert lastmodel >= 0, "No saved models!"
    return os.path.join(basedir, str(lastmodel))


# 测试模型，传入两种model的路径
def test_models(model, score_model):
    # baseline: choose actions at random
    random_action = lambda bytez: np.random.choice(list(manipulate.ACTION_TABLE.keys()))
    random_success, misclassified = evaluate(random_action)
    total = len(sha256_holdout) - len(misclassified)  # don't count misclassified towards success

    fe = pefeatures.PEFeatureExtractor()

    def agent_policy(agent):
        def f(bytez):
            # first, get features from bytez
            feats = fe.extract(bytez)
            action_index = agent.act(feats)
            return ACTION_LOOKUP[action_index]

        return f

    # ddqn
    env = gym.make('malware-test-v0')
    agent = create_ddqn_agent(env)
    last_model_dir = get_latest_model_from(model)
    agent.load(last_model_dir)
    success, _ = evaluate(agent_policy(agent))

    env_score = gym.make('malware-score-test-v0')
    agent_score = create_ddqn_agent(env_score)
    last_model_dir = get_latest_model_from(score_model)
    agent_score.load(last_model_dir)
    score_success, _ = evaluate(agent_policy(agent_score))

    print("Success rate of random chance: {}\n".format(len(random_success) / total))
    print("Success rate (black box): {}\n".format(len(success) / total))
    print("Success rate (score): {}\n".format(len(score_success) / total))


import sys

if __name__ == '__main__':
    model = "models/create_ddqn_agent"
    score_model = model + "_score"
    try:
        rounds = sys.argv[1]
        model += "_" + rounds
        score_model += "_" + rounds
        print(model)
        print(score_model)

    except:
        print("you need to input the training rounds!")

    test_models(model, score_model)