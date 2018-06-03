"""An example of training DQN against OpenAI Gym Envs.

This script is an example of training a DQN agent against OpenAI Gym envs.
Both discrete and continuous action spaces are supported. For continuous action
spaces, A NAF (Normalized Advantage Function) is used to approximate Q-values.

To solve CartPole-v0, run:
    python train_dqn_gym.py --env CartPole-v0

To solve Pendulum-v0, run:
    python train_dqn_gym.py --env Pendulum-v0
"""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
import gym_malware
import datetime
from builtins import *  # NOQA
from future import standard_library

from bin.visdom_plot_hook import VisdomPlotHook

standard_library.install_aliases()  # NOQA

import argparse
import os
import sys

from chainer import optimizers
import gym

gym.undo_logger_setup()  # NOQA
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl.agents import DoubleDQN
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import links
from chainerrl import misc
from chainerrl import q_functions
from chainerrl import replay_buffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=123,
                        help='Random seed [0, 2 ** 32)')
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--final-exploration-steps',
                        type=int, default=10 ** 4)
    parser.add_argument('--start-epsilon', type=float, default=1.0)
    parser.add_argument('--end-epsilon', type=float, default=0.1)
    parser.add_argument('--noisy-net-sigma', type=float, default=None)
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--prioritized-replay', action='store_true', default=False)
    parser.add_argument('--episodic-replay', action='store_true', default=False)
    parser.add_argument('--replay-start-size', type=int, default=1000)
    parser.add_argument('--target-update-interval', type=int, default=10 ** 2)
    parser.add_argument('--target-update-method', type=str, default='hard')
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--update-interval', type=int, default=1)
    parser.add_argument('--eval-n-runs', type=int, default=50)
    parser.add_argument('--eval-interval', type=int, default=10 ** 3)
    parser.add_argument('--n-hidden-channels', type=int, default=512)
    parser.add_argument('--n-hidden-layers', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--minibatch-size', type=int, default=None)
    parser.add_argument('--render-train', action='store_true')
    parser.add_argument('--render-eval', action='store_true')
    parser.add_argument('--monitor', action='store_true', default=True)
    parser.add_argument('--reward-scale-factor', type=float, default=1e-3)
    args = parser.parse_args()

    # Set a random seed used in ChainerRL
    if args.gpu >= 0:
        misc.set_random_seed(args.seed)

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    def make_env(test):
        ENV_NAME = 'malware-test-v0' if test else 'malware-v0'
        env = gym.make(ENV_NAME)
        # Use different random seeds for train and test envs
        env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
        env.seed(env_seed)
        if args.monitor:
            env = gym.wrappers.Monitor(env, args.outdir)
        # if not test:
        #     misc.env_modifiers.make_reward_filtered(
        #         env, lambda x: x * args.reward_scale_factor)
        if ((args.render_eval and test) or
                (args.render_train and not test)):
            misc.env_modifiers.make_rendered(env)
        return env

    env = make_env(test=False)
    timestep_limit = 80
    obs_space = env.observation_space
    obs_size = obs_space.shape[0]
    action_space = env.action_space

    n_actions = action_space.n
    q_func = q_functions.FCStateQFunctionWithDiscreteAction(
            obs_size, n_actions,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers)
    if args.gpu >= 0:
        q_func.to_gpu(args.gpu)

    # Use epsilon-greedy for exploration
    explorer = explorers.LinearDecayEpsilonGreedy(
            args.start_epsilon, args.end_epsilon, args.final_exploration_steps,
            action_space.sample)

    if args.noisy_net_sigma is not None:
        links.to_factorized_noisy(q_func)
        # Turn off explorer
        explorer = explorers.Greedy()

    # Draw the computational graph and save it in the output directory.
    if args.gpu < 0:
        chainerrl.misc.draw_computational_graph(
            [q_func(np.zeros_like(obs_space, dtype=np.float32)[None])],
            os.path.join(args.outdir, 'model'))

    opt = optimizers.Adam()
    opt.setup(q_func)

    rbuf_capacity = 5 * 10 ** 5
    if args.episodic_replay:
        if args.minibatch_size is None:
            args.minibatch_size = 4
        if args.prioritized_replay:
            betasteps = (args.steps - args.replay_start_size) \
                        // args.update_interval
            rbuf = replay_buffer.PrioritizedEpisodicReplayBuffer(
                rbuf_capacity, betasteps=betasteps)
        else:
            rbuf = replay_buffer.EpisodicReplayBuffer(rbuf_capacity)
    else:
        if args.minibatch_size is None:
            args.minibatch_size = 32
        if args.prioritized_replay:
            betasteps = (args.steps - args.replay_start_size) \
                        // args.update_interval
            rbuf = replay_buffer.PrioritizedReplayBuffer(
                rbuf_capacity, betasteps=betasteps)
        else:
            rbuf = replay_buffer.ReplayBuffer(rbuf_capacity)

    def phi(obs):
        return obs.astype(np.float32)

    agent = DoubleDQN(q_func, opt, rbuf, gamma=args.gamma,
                      explorer=explorer, replay_start_size=args.replay_start_size,
                      target_update_interval=args.target_update_interval,
                      update_interval=args.update_interval,
                      phi=phi, minibatch_size=args.minibatch_size,
                      target_update_method=args.target_update_method,
                      soft_update_tau=args.soft_update_tau,
                      episodic_update=args.episodic_replay, episodic_update_len=16)

    if args.load:
        agent.load(args.load)

    eval_env = make_env(test=True)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_runs=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        q_hook = VisdomPlotHook('Average Q Value')
        loss_hook = VisdomPlotHook('Average Loss', plot_index=1)

        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_runs=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir, eval_env=eval_env,
            max_episode_len=timestep_limit,
            step_hooks=[q_hook, loss_hook],
            successful_score=7
        )


if __name__ == '__main__':
    main()