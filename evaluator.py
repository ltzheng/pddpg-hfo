from learner import Learner
import os
import argparse
import time
import numpy as np
import torch
from agents.random_agent import RandomAgent
from agents.pddpg_agent import PDDPGAgent
from agents.mapddpg_agent import MAPDDPGAgent
from envs import offense_mid_action
from utils.redis_manager import connect_redis, query_all_obs_actions, sync_agent_obs_actions, sync_agent_policy
import logging


class Evaluator(Learner):
    def __init__(self, agent_type, tensorboard_dir, save_dir=None, player='offense', save_freq=500,
                 seed=1, episodes=50, server_port=6000, eval_episodes=1000, start=0):
        super(Evaluator, self).__init__(agent_type=agent_type, tensorboard_dir=tensorboard_dir,
                                        save_dir=save_dir, player=player,
                                        seed=seed, episodes=episodes, start=start,
                                        server_port=server_port, save_freq=save_freq)
        self.player = player
        self.eval_episodes = eval_episodes
     
    def log_results(self, evaluation_results, i):
        num_results = evaluation_results.shape[0]
        total_returns = sum(evaluation_results[:, 0])
        total_timesteps = sum(evaluation_results[:, 1])
        goal_timesteps = evaluation_results[:, 1][evaluation_results[:, 2] == 1]
        total_goal = sum(evaluation_results[:, 2])
        total_captured = sum(evaluation_results[:, 3])

        avg_returns = total_returns / num_results
        avg_timesteps = total_timesteps / num_results
        avg_goal_timesteps = sum(goal_timesteps) / num_results
        avg_goal_prob = total_goal / num_results
        avg_captured_prob = total_captured / num_results

        print("Avg. evaluation return =", avg_returns)
        print("Avg. timesteps =", avg_timesteps)
        if self.player == 'offense':
            print("Avg. goal prob. =", avg_goal_prob)
            print("Avg. timesteps per goal =", avg_goal_timesteps)
            print("Avg. captured prob. =", avg_captured_prob)
        elif self.player == 'goalie':
            print("Avg. lose prob. =", avg_goal_prob)
            print("Avg. capture prob. =", avg_captured_prob / (1 + avg_goal_prob))

        self.writer.add_scalar('Avg. evaluation return', avg_returns, i)
        self.writer.add_scalar('Avg. timesteps', avg_timesteps, i)
        if self.player == 'offense':
            self.writer.add_scalar("Avg. goal prob.", avg_goal_prob, i)
            self.writer.add_scalar("Avg. timesteps per goal", avg_goal_timesteps, i)
            self.writer.add_scalar("Avg. captured prob.", avg_captured_prob, i)
        elif self.player == 'goalie':
            self.writer.add_scalar("Avg. lose prob.", avg_goal_prob, i)
            self.writer.add_scalar("Avg. capture prob.", avg_captured_prob / (1 + avg_goal_prob), i)

        return avg_returns, avg_timesteps, avg_goal_timesteps, avg_goal_prob, avg_captured_prob

    def run(self):
        # Random seed
        # self.seed += 10000 * proc_id()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # later can sort by approximity (need to be same as in redis_manager)
        self.all_agents = list(self.redis_instance.smembers('teammates'))
        self.all_agents.sort()

        start_time = time.time()

        print("Evaluating self.agent over {} episodes".format(self.episodes))
        self.agent.epsilon_final = 0.
        self.agent.epsilon = 0.
        self.agent.noise = None

        # PDDPG and MAPDDPG evaluation
        if isinstance(self.agent, PDDPGAgent):
            # main loop
            assert self.save_freq < self.episodes + 1
            for i in range(self.start + self.save_freq, self.start + self.episodes + 1, self.save_freq):
                # load model
                returns = []
                timesteps = []
                goals = []
                captureds = []
                self.load_model(self.save_dir, i)

                for j in range(self.eval_episodes):
                    info = {'status': "NOT_SET"}
                    # initialize environment and reward
                    obs = self.env.reset()
                    obs = np.array(obs, dtype=np.float32, copy=False)
                    episode_reward = 0.
                    terminal = False
                    t = 0

                    # get discrete action and continuous parameters
                    act, act_param, _, _ = self.agent.act(obs)
                    action = offense_mid_action(act, act_param)

                    while not terminal:
                        t += 1
                        next_obs, reward, terminal, info = self.env.step(action)
                        next_obs = np.array(next_obs, dtype=np.float32, copy=False)
                        # get discrete action and continuous parameters
                        next_act, next_act_param, _, _ = self.agent.act(next_obs)
                        next_action = offense_mid_action(next_act, next_act_param)

                        action = next_action
                        episode_reward += reward

                    goal = info['status'] == 'GOAL'
                    captured = info['status'] == 'CAPTURED_BY_DEFENSE'
                    timesteps.append(t)
                    returns.append(episode_reward)
                    goals.append(goal)
                    captureds.append(captured)

                evaluation_results = np.column_stack((returns, timesteps, goals, captureds))
                avg_returns, avg_timesteps, avg_goal_timesteps, avg_goal_prob, avg_captured_prob = \
                    self.log_results(evaluation_results, i)

        end_time = time.time()
        print("Evaluation time: %.2f seconds" % (end_time - start_time))

        return avg_returns, avg_timesteps, avg_goal_timesteps, avg_goal_prob, avg_captured_prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-type', type=str, default='PDDPG')
    parser.add_argument('--player', type=str, default='offense')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--server-port', type=int, default=6000)
    parser.add_argument('--episodes', type=int, default=20000)
    parser.add_argument('--eval-episodes', type=int, default=1000)
    parser.add_argument('--save-freq', type=int, default=500)
    parser.add_argument('--tensorboard-dir', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()

    evaluator = Evaluator(agent_type=args.agent_type, player=args.player, seed=args.seed, episodes=args.episodes,
                          server_port=args.server_port, tensorboard_dir=args.tensorboard_dir, start=args.start,
                          save_dir=args.save_dir, eval_episodes=args.eval_episodes, save_freq=args.save_freq)
    evaluator.redis_instance.decr('not ready', amount=1)

    # to act nearly synchronously
    while int(evaluator.redis_instance.get('not ready')) > 0:
        print('\rNumber of not ready learners:', evaluator.redis_instance.get('not ready'), end='')
    print('======Start Evaluation======')
    evaluator.run()
