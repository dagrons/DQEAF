# coding=UTF-8
# ! /usr/bin/python
import argparse
import linecache
import os
import sys

import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import gym
import numpy as np
from chainer import optimizers
from chainerrl import experiments, explorers, links, replay_buffer, misc

from bin.test_agent_chainer import evaluate
from gym_malware import sha256_holdout
from gym_malware.envs.controls import manipulate2 as manipulate
from gym_malware.envs.utils import pefeatures
from hook.plot_hook import PlotHook
from hook.training_scores_hook import TrainingScoresHook

ACTION_LOOKUP = {i: act for i, act in enumerate(manipulate.ACTION_TABLE.keys())}


# 用于快速调用chainerrl的训练方法，参数如下：
# python train.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='models')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--final-exploration-steps', type=int, default=10 ** 4)
    parser.add_argument('--start-epsilon', type=float, default=1.0)
    parser.add_argument('--end-epsilon', type=float, default=0.1)
    parser.add_argument('--noisy-net-sigma', action='store_true')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--steps', type=int, default=1100)
    parser.add_argument('--prioritized-replay', action='store_false')
    parser.add_argument('--episodic-replay', action='store_true')
    parser.add_argument('--replay-start-size', type=int, default=1000)
    parser.add_argument('--target-update-interval', type=int, default=10 ** 2)
    parser.add_argument('--target-update-method', type=str, default='hard')
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--eval-n-runs', type=int, default=200)
    parser.add_argument('--eval-interval', type=int, default=10 ** 3)
    parser.add_argument('--n-hidden-channels', type=int, default=512)
    parser.add_argument('--n-hidden-layers', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--minibatch-size', type=int, default=None)
    parser.add_argument('--test-random', action='store_true')
    parser.add_argument('--rounds', type=int, default=10)
    args = parser.parse_args()

    class QFunction(chainer.Chain):
        def __init__(self, obs_size, n_actions, n_hidden_channels=None):
            super(QFunction, self).__init__()
            if n_hidden_channels is None:
                n_hidden_channels = [512, 64]
            net = []
            inpdim = obs_size
            for i, n_hid in enumerate(n_hidden_channels):
                net += [('l{}'.format(i), L.Linear(inpdim, n_hid))]
                # net += [('norm{}'.format(i), L.BatchNormalization(n_hid))]
                net += [('_act{}'.format(i), F.relu)]
                inpdim = n_hid

            net += [('output', L.Linear(inpdim, n_actions))]

            with self.init_scope():
                for n in net:
                    if not n[0].startswith('_'):
                        setattr(self, n[0], n[1])

            self.forward = net

        def __call__(self, x, test=False):
            """
            Args:
                x (ndarray or chainer.Variable): An observation
                test (bool): a flag indicating whether it is in test mode
            """
            for n, f in self.forward:
                if not n.startswith('_'):
                    x = getattr(self, n)(x)
                else:
                    x = f(x)

            return chainerrl.action_value.DiscreteActionValue(x)

    # 创建ddqn agent
    def create_ddqn_agent(env, args):
        obs_size = env.observation_space.shape[0]
        action_space = env.action_space
        n_actions = action_space.n

        # q_func = q_functions.FCStateQFunctionWithDiscreteAction(
        #     obs_size, n_actions,
        #     n_hidden_channels=args.n_hidden_channels,
        #     n_hidden_layers=args.n_hidden_layers)
        q_func = QFunction(obs_size, n_actions)
        if args.gpu:
            q_func.to_gpu(args.gpu)

        # Draw the computational graph and save it in the output directory.
        if not args.test and not args.gpu:
            chainerrl.misc.draw_computational_graph(
                [q_func(np.zeros_like(env.observation_space, dtype=np.float32)[None])],
                os.path.join(args.outdir, 'model'))

        # Use epsilon-greedy for exploration
        explorer = explorers.LinearDecayEpsilonGreedy(
            args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
            action_space.sample)
        # explorer = explorers.Boltzmann()
        # explorer = explorers.ConstantEpsilonGreedy(
        #     epsilon=0.3, random_action_func=env.action_space.sample)

        if args.noisy_net_sigma:
            links.to_factorized_noisy(q_func)
            # Turn off explorer
            explorer = explorers.Greedy()

        opt = optimizers.Adam()
        opt.setup(q_func)

        rbuf_capacity = 5 * 10 ** 5
        if args.episodic_replay:
            if args.minibatch_size is None:
                args.minibatch_size = 4
            if args.prioritized_replay:
                betasteps = (args.steps - args.replay_start_size) // args.update_interval
                rbuf = replay_buffer.PrioritizedEpisodicReplayBuffer(rbuf_capacity, betasteps=betasteps)
            else:
                rbuf = replay_buffer.EpisodicReplayBuffer(rbuf_capacity)
        else:
            if args.minibatch_size is None:
                args.minibatch_size = 32
            if args.prioritized_replay:
                betasteps = (args.steps - args.replay_start_size) // args.update_interval
                rbuf = replay_buffer.PrioritizedReplayBuffer(rbuf_capacity, betasteps=betasteps)
            else:
                rbuf = replay_buffer.ReplayBuffer(rbuf_capacity)

        # Chainer only accepts numpy.float32 by default, make sure
        # a converter as a feature extractor function phi.
        phi = lambda x: x.astype(np.float32, copy=False)

        agent = chainerrl.agents.DoubleDQN(q_func, opt, rbuf, gamma=args.gamma,
                                           explorer=explorer, replay_start_size=args.replay_start_size,
                                           target_update_interval=args.target_update_interval,
                                           update_interval=args.update_interval,
                                           phi=phi, minibatch_size=args.minibatch_size,
                                           target_update_method=args.target_update_method,
                                           soft_update_tau=args.soft_update_tau,
                                           episodic_update=args.episodic_replay,
                                           episodic_update_len=16)

        return agent

    # 开始训练
    def train_agent(args, use_score=False):
        ENV_NAME = 'malware-score-v0' if use_score else 'malware-v0'
        env = gym.make(ENV_NAME)
        ENV_TEST_NAME = 'malware-score-test-v0' if use_score else 'malware-test-v0'
        test_env = gym.make(ENV_TEST_NAME)

        np.random.seed(123)
        env.seed(123)
        # Set a random seed used in ChainerRL
        misc.set_random_seed(123)

        agent = create_ddqn_agent(env, args)

        q_hook = PlotHook('Average Q Value')
        loss_hook = PlotHook('Average Loss', plot_index=1)
        scores_hook = TrainingScoresHook('scores.txt', args.outdir)

        chainerrl.experiments.train_agent_with_evaluation(
            agent, env,
            steps=args.steps,  # Train the graduation_agent for this many rounds steps
            max_episode_len=env.maxturns,  # Maximum length of each episodes
            eval_interval=args.eval_interval,  # Evaluate the graduation_agent after every 1000 steps
            eval_n_runs=args.eval_n_runs,  # 100 episodes are sampled for each evaluation
            outdir=args.outdir,  # Save everything to 'result' directory
            step_hooks=[q_hook, loss_hook, scores_hook],
            successful_score=7,
            eval_env=test_env
        )

        # 保证训练一轮就成功的情况下能成功打印scores.txt文件
        scores_hook(None, None, 1000)

        return env, agent

    # 获取保存的模型目录
    def get_latest_model_dir_from(basedir):
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

    # test
    if not args.test:
        print("training...")

        # 反复多次重新训练模型，避免手工操作
        for _ in range(args.rounds):
            args.outdir = experiments.prepare_output_dir(
                args, args.outdir, argv=sys.argv)
            print('Output files are saved in {}'.format(args.outdir))

            env, agent = train_agent(args)

            with open(os.path.join(args.outdir, 'scores.txt'), 'a') as f:
                f.write(
                    "total_turn/episode->{}({}/{})\n".format(env.total_turn / env.episode, env.total_turn, env.episode))

            # 标识成功失败
            dirs = os.listdir(args.outdir)
            second_line = linecache.getline(os.path.join(args.outdir, 'scores.txt'), 2)
            success_score = second_line.strip('\n').split('\t')[3]

            # 训练提前结束，标识成功
            success_flag = False
            for file in dirs:
                if file.endswith('_finish') and not file.startswith(str(args.steps)):
                    success_flag = True
                    break

            os.rename(args.outdir, '{}-{}{}'.format(args.outdir, success_score, '-success' if success_flag else ''))

            # 重置outdir到models
            args.outdir = 'models'
    else:
        print("testing...")
        model_fold = os.path.join(args.outdir, args.load)
        scores_file = os.path.join(model_fold, 'scores.txt')

        # baseline: choose actions at random
        if args.test_random:
            random_action = lambda bytez: np.random.choice(list(manipulate.ACTION_TABLE.keys()))
            random_success, misclassified = evaluate(random_action)
            total = len(sha256_holdout) - len(misclassified)  # don't count misclassified towards success

            with open(scores_file, 'a') as f:
                random_result = "random: {}({}/{})\n".format(len(random_success) / total, len(random_success), total)
                f.write(random_result)
                f.write("==========================\n")

        total = len(sha256_holdout)
        fe = pefeatures.PEFeatureExtractor()

        def agent_policy(agent):
            def f(bytez):
                # first, get features from bytez
                feats = fe.extract2(bytez)
                action_index = agent.act(feats)
                return ACTION_LOOKUP[action_index]

            return f

        # ddqn
        env = gym.make('malware-test-v0')
        agent = create_ddqn_agent(env, args)
        mm = get_latest_model_dir_from(model_fold)
        agent.load(mm)
        success, _ = evaluate(agent_policy(agent))
        blackbox_result = "black: {}({}/{})".format(len(success) / total, len(success), total)
        with open(scores_file, 'a') as f:
            f.write("{}->{}\n".format(mm, blackbox_result))


if __name__ == '__main__':
    main()
