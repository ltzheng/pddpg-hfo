import os
import argparse
import time
from utils import configure_agent, compute_n_step_returns
from envs.soccer_offense import SoccerOffenseEnv
from envs.soccer_goalie import SoccerGoalieEnv
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import hfo_py
from agents.random_agent import RandomAgent
from agents.pddpg_agent import PDDPGAgent
from agents.mapddpg_agent import MAPDDPGAgent
from envs import offense_mid_action, defense_mid_action, goalie_mid_action
from utils.redis_manager import connect_redis, query_all_obs_actions, sync_agent_obs_actions, sync_agent_policy


def make_env(player, server_port, team, scale_actions):
    # Create the HFO Environment
    hfo = hfo_py.HFOEnvironment()
    # Connect to the server
    agent_args = configure_agent(player=player, team=team, port=server_port)

    hfo.connectToServer(**agent_args)

    if player == 'offense':
        env = SoccerOffenseEnv(hfo)
    # elif player == 'defense':
    #     env = SoccerDefenseEnv(hfo)
    elif player == 'goalie':
        env = SoccerGoalieEnv(hfo)
    else:
        raise ValueError('Wrong player type')
    return env


class Learner(object):
    def __init__(self, agent_type, tensorboard_dir, save_dir=None, player='offense', pretrained=None,
                 seed=1, episodes=20000, server_port=6000, max_steps=15000, save_freq=500, start=0):
        if player == 'offense':
            self.team = 'base_left'
            self.mid_action = offense_mid_action
        else:
            self.team = 'base_right'
            if player == 'goalie':
                self.mid_action = goalie_mid_action
            else:
                self.mid_action = defense_mid_action

        self.seed = seed
        self.save_freq = save_freq
        self.episodes = episodes
        self.scale_actions = True
        self.server_port = server_port
        self.max_steps = max_steps
        self.save_dir = os.path.join('./saved_models', save_dir)
        self.tensorboard_dir = os.path.join('./tensorboard-log', tensorboard_dir)
        self.pretrained = pretrained
        self.start = start
        self.env = make_env(player, self.server_port, self.team, self.scale_actions)

        # configure redis
        self.redis_instance = connect_redis()
        # initialize teammate set
        self.redis_instance.sadd('teammates', self.env.unum)
        print('Number of teammates:', self.redis_instance.scard('teammates'))

        if agent_type == 'PDDPG':
            self.agent = PDDPGAgent(self.env.observation_space, self.env.action_space,
                                    actor_kwargs={'hidden_layers': [1024, 512, 256, 256, 128, 128],
                                                  'init_type': "kaiming",
                                                  'init_std': 0.01, 'activation': 'leaky_relu'},
                                    critic_kwargs={'hidden_layers': [1024, 512, 256, 256, 128, 128],
                                                   'init_type': "kaiming",
                                                   'init_std': 0.01, 'activation': 'leaky_relu'},
                                    batch_size=32,  # batch_size,
                                    learning_rate_actor=0.001,  # learning_rate_actor,  # 0.0001
                                    learning_rate_critic=0.001,  # learning_rate_critic,  # 0.001
                                    gamma=0.99,  # gamma,  # 0.99
                                    tau_actor=0.001,  # tau,
                                    tau_critic=0.001,  # tau,
                                    n_step_returns=True,  # n_step_returns,
                                    epsilon_steps=1000,  # epsilon_steps,
                                    epsilon_final=0.1,  # epsilon_final,
                                    replay_memory_size=500000,  # replay_memory_size,
                                    inverting_gradients=True,  # inverting_gradients,
                                    initial_memory_threshold=1000,  # initial_memory_threshold,
                                    beta=0.2,  # beta,
                                    clip_grad=1.,  # clip_grad,
                                    use_ornstein_noise=False,  # use_ornstein_noise,
                                    adam_betas=(0.9, 0.999),  # default 0.95,0.999
                                    seed=self.seed)

        elif agent_type == 'MAPDDPG':
            self.agent = MAPDDPGAgent(self.env.observation_space, self.env.action_space,
                                      actor_kwargs={'hidden_layers': [1024, 512, 256, 256, 128, 128],
                                                    'init_type': "kaiming",
                                                    'init_std': 0.01, 'activation': 'leaky_relu'},
                                      critic_kwargs={'hidden_layers': [1024, 512, 256, 256, 128, 128],
                                                     'init_type': "kaiming",
                                                     'init_std': 0.01, 'activation': 'leaky_relu'},
                                      batch_size=32,  # batch_size,
                                      learning_rate_actor=0.001,  # learning_rate_actor,  # 0.0001
                                      learning_rate_critic=0.001,  # learning_rate_critic,  # 0.001
                                      gamma=0.99,  # gamma,  # 0.99
                                      tau_actor=0.001,  # tau,
                                      tau_critic=0.001,  # tau,
                                      n_step_returns=True,  # n_step_returns,
                                      epsilon_steps=1000,  # epsilon_steps,
                                      epsilon_final=0.1,  # epsilon_final,
                                      replay_memory_size=500000,  # replay_memory_size,
                                      inverting_gradients=True,  # inverting_gradients,
                                      initial_memory_threshold=1000,  # initial_memory_threshold,
                                      beta=0.2,  # beta,
                                      clip_grad=1.,  # clip_grad,
                                      use_ornstein_noise=False,  # use_ornstein_noise,
                                      adam_betas=(0.9, 0.999),  # default 0.95,0.999
                                      seed=self.seed,
                                      num_agents=int(self.redis_instance.get('num_agents')),
                                      unum=self.env.unum)

        elif agent_type == 'RANDOM':
            self.agent = RandomAgent(observation_space=self.env.observation_space, action_space=self.env.action_space)
        else:
            raise NotImplementedError

        self.writer = SummaryWriter(self.tensorboard_dir)
        print('Agent:', self.agent)

    def load_model(self, dir, i):
        prefix = os.path.join(dir, str(i))
        print('Evaluating model from', prefix, '...')
        self.agent.actor.load_state_dict(torch.load(prefix + '_actor.pt', map_location='cpu'))
        self.agent.critic.load_state_dict(torch.load(prefix + '_critic.pt', map_location='cpu'))
        self.agent.actor.eval()
        self.agent.critic.eval()
        print('Models evaluated successfully')

    def run(self):
        # Random seed
        # self.seed += 10000 * proc_id()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # later can sort by approximity (need to be same as in redis_manager)
        self.all_agents = list(self.redis_instance.smembers('teammates'))
        self.all_agents.sort()

        # Prepare for interaction with environment
        start_time = time.time()

        # if isinstance(self.agent, RandomAgent):
        #     for i in range(self.episodes):
        #         obs = self.env.reset()
        #         obs = np.array(obs, dtype=np.float32, copy=False)
        #         print(obs)
        #
        #         for j in range(self.local_steps_per_episode):
        #             act, act_param = self.agent.act(obs)
        #             action = mid_action(act, act_param)
        #
        #             next_obs, reward, terminal, info = self.env.step(action)
        #             obs = next_obs

        n_step_returns = True
        update_ratio = 0.1

        if self.save_freq > 0 and self.save_dir:
            self.save_dir = os.path.join(self.save_dir, 'agent' + str(self.env.unum))
            os.makedirs(self.save_dir, exist_ok=True)

        if self.pretrained:
            self.load_model(self.pretrained, self.start)

        # train log
        total_reward = 0.
        returns = []
        timesteps = []
        goals = []

        if isinstance(self.agent, PDDPGAgent) and not isinstance(self.agent, MAPDDPGAgent):
            print('\n===========Start Training PDDPG===========')

            # main loop
            for i in range(self.start, self.start + self.episodes):
                # save model
                if self.save_freq > 0 and self.save_dir and (i + 1) % self.save_freq == 0:
                    prefix = os.path.join(self.save_dir, str(i + 1))
                    torch.save(self.agent.actor.state_dict(), prefix + '_actor.pt')
                    torch.save(self.agent.critic.state_dict(), prefix + '_critic.pt')
                    print('Models saved successfully at episode' + str(i + 1))

                info = {'status': "NOT_SET"}

                # initialize environment, reward and transitions
                obs = self.env.reset()
                obs = np.array(obs, dtype=np.float32, copy=False)
                episode_reward = 0.
                transitions = []

                # get discrete action and continuous parameters
                act, act_param, all_actions, all_action_parameters = self.agent.act(obs)
                action = self.mid_action(act, act_param)

                for j in range(self.max_steps):
                    next_obs, reward, terminal, info = self.env.step(action)
                    next_obs = np.array(next_obs, dtype=np.float32, copy=False)

                    # get discrete action and continuous parameters
                    next_act, next_act_param, next_all_actions, next_all_action_parameters = self.agent.act(next_obs)
                    next_action = self.mid_action(next_act, next_act_param)

                    if n_step_returns:
                        transitions.append(
                            [obs, np.concatenate((all_actions.data, all_action_parameters.data)).ravel(), reward,
                             next_obs, np.concatenate((next_all_actions.data,
                                                       next_all_action_parameters.data)).ravel(), terminal])
                    else:
                        self.agent.step(obs, (act, act_param, all_actions, all_action_parameters), reward, next_obs,
                                        (next_act, next_act_param, next_all_actions, next_all_action_parameters),
                                        terminal,
                                        optimise=False)

                    act, act_param, all_actions, all_action_parameters = \
                        next_act, next_act_param, next_all_actions, next_all_action_parameters
                    action = next_action
                    obs = next_obs

                    episode_reward += reward
                    # env.render()

                    if terminal:
                        break

                # decay epsilon
                self.agent.end_episode()

                # calculate n-step returns
                if n_step_returns:
                    nsreturns = compute_n_step_returns(transitions, self.agent.gamma)
                    for t, nsr in zip(transitions, nsreturns):
                        t.append(nsr)
                        self.agent.replay_memory.append(state=t[0], action=t[1], reward=t[2], next_state=t[3],
                                                        next_action=t[4],
                                                        terminal=t[5], time_steps=None, n_step_return=nsr)

                # update networks at the end of each episode
                n_updates = int(update_ratio * j)
                for _ in range(n_updates):
                    self.agent.update()

                # train log
                returns.append(episode_reward)
                timesteps.append(j)
                goals.append(info['status'] == 'GOAL')

                total_reward += episode_reward
                self.writer.add_scalar('Episode reward', episode_reward, i)
                if i % 100 == 0:
                    print('{0:5s} : Total mean reward:{1:.4f} | Episode reward:{2:.4f}'
                          .format(str(i + 1), total_reward / (i + 1), episode_reward))
                    self.writer.add_scalar('Last 100episodes mean reward', np.array(returns[-100:]).mean(), i)

        elif isinstance(self.agent, MAPDDPGAgent):
            print('\n===========Start Training MAPDDPG===========')

            # main loop
            for i in range(self.start, self.start + self.episodes):
                # save model
                if self.save_freq > 0 and self.save_dir and (i + 1) % self.save_freq == 0:
                    prefix = os.path.join(self.save_dir, str(i + 1))
                    torch.save(self.agent.actor.state_dict(), prefix + '_actor.pt')
                    torch.save(self.agent.critic.state_dict(), prefix + '_critic.pt')
                    print('Models saved successfully at episode' + str(i + 1))

                info = {'status': "NOT_SET"}

                # initialize environment, reward and transitions
                obs = self.env.reset()
                obs = np.array(obs, dtype=np.float32, copy=False)
                episode_reward = 0.
                transitions = []

                # get discrete action and continuous parameters
                act, act_param, all_actions, all_action_parameters = self.agent.act(obs)
                action = self.mid_action(act, act_param)
                # update the observation and action of agent i in redis
                sync_agent_obs_actions(self.redis_instance, self.env.unum, obs, all_actions, all_action_parameters)
                # query all agents' observations and actions
                all_agent_obs, all_agent_actions, success1 = query_all_obs_actions(self.redis_instance)

                for j in range(self.max_steps):
                    # # TODO: query other agents' actions (for inference use)

                    # take action in environment
                    next_obs, reward, terminal, info = self.env.step(action)
                    next_obs = np.array(next_obs, dtype=np.float32, copy=False)

                    # get discrete action and continuous parameters
                    next_act, next_act_param, next_all_actions, next_all_action_parameters = self.agent.act(next_obs)
                    next_action = self.mid_action(next_act, next_act_param)

                    # update the observation and action of agent i in redis
                    sync_agent_obs_actions(self.redis_instance, self.env.unum, next_obs,
                                           next_all_actions, next_all_action_parameters)
                    # query all agents' observations and actions
                    all_agent_next_obs, all_agent_next_actions, success2 = query_all_obs_actions(self.redis_instance)

                    if n_step_returns and success1 and success2:
                        transitions.append([all_agent_obs.ravel(), all_agent_actions.ravel(), reward,
                                            all_agent_next_obs.ravel(), all_agent_next_actions.ravel(), terminal])
                    # else:
                    #     self.agent.step(obs, (act, act_param, all_actions, all_action_parameters), reward, next_obs,
                    #                     (next_act, next_act_param, next_all_actions, next_all_action_parameters),
                    #                     terminal,
                    #                     optimise=False)

                    all_agent_actions = all_agent_next_actions
                    action = next_action
                    all_agent_obs = all_agent_next_obs
                    success1 = True

                    episode_reward += reward

                    if terminal:
                        break

                # decay epsilon
                self.agent.end_episode()

                # calculate n-step returns
                if n_step_returns:
                    nsreturns = compute_n_step_returns(transitions, self.agent.gamma)
                    for t, nsr in zip(transitions, nsreturns):
                        t.append(nsr)
                        if not any(elem is None for elem in t):
                            self.agent.replay_memory.append(state=t[0], action=t[1], reward=t[2], next_state=t[3],
                                                            next_action=t[4], terminal=t[5],
                                                            time_steps=None, n_step_return=nsr)

                # update networks at the end of each episode
                n_updates = int(update_ratio * j)
                for _ in range(n_updates):
                    # sync policy in redis
                    sync_agent_policy(self.redis_instance, self.env.unum, self.agent.actor_target)
                    self.agent.update()

                # train log
                returns.append(episode_reward)
                timesteps.append(j)
                goals.append(info['status'] == 'GOAL')

                total_reward += episode_reward
                self.writer.add_scalar('Episode reward', episode_reward, i)
                if i % 100 == 0:
                    print('{0:5s} : Total mean reward:{1:.4f} | Episode reward:{2:.4f}'
                          .format(str(i + 1), total_reward / (i + 1), episode_reward))
                    self.writer.add_scalar('Last 100episodes mean reward', np.array(returns[-100:]).mean(), i)

        train_goal_ratio = goals.count(True) / self.episodes
        print("Training goal ratio: %.2f" % train_goal_ratio)

        end_time = time.time()
        print("Training time: %.2f seconds" % (end_time - start_time))
        print("==========TRAINING END==========")
        # env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent-type', type=str, default='PDDPG')
    parser.add_argument('--player', type=str, default='offense')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--server-port', type=int, default=6000, help='port of hfo server')
    parser.add_argument('--episodes', type=int, default=20000, help='number of training episodes')
    parser.add_argument('--tensorboard-dir', type=str, default=None, help='where to save tensorboard loggings')
    parser.add_argument('--save-dir', type=str, default=None, help='where to save the trained models')
    parser.add_argument('--pretrained', type=str, default=None, help='directory of pretrained models')
    parser.add_argument('--start', type=int, default=0, help='id of pretrained models to load')
    args = parser.parse_args()

    learner = Learner(agent_type=args.agent_type, player=args.player, seed=args.seed, episodes=args.episodes,
                      server_port=args.server_port, tensorboard_dir=args.tensorboard_dir, pretrained=args.pretrained,
                      save_dir=args.save_dir, save_freq=500, start=args.start)
    learner.redis_instance.decr('not ready', amount=1)

    # to act nearly synchronously
    while int(learner.redis_instance.get('not ready')) > 0:
        print('\rNumber of not ready learners:', learner.redis_instance.get('not ready'), end='')
    learner.run()
