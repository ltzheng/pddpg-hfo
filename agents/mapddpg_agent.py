from models.mapddpg import CentralCritic
from models.pddpg import Actor
import torch
import numpy as np
from agents.pddpg_agent import PDDPGAgent
from utils.memory import MemoryNStepReturns
from utils import soft_update_target_network, hard_update_target_network
from utils import OrnsteinUhlenbeckActionNoise
import torch.nn.functional as F
from utils.redis_manager import connect_redis, query_all_policies, get_all_agents, get_agent_idx
from torch.autograd import Variable
import random
import torch.optim as optim
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MAPDDPGAgent(PDDPGAgent):
    """
    DDPG actor-critic agent for parameterised action spaces
    [Hausknecht and Stone 2016]
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 actor_kwargs={},
                 critic_class=CentralCritic,
                 critic_kwargs={},
                 epsilon_initial=1.0,
                 epsilon_final=0.01,
                 epsilon_steps=10000,
                 batch_size=64,
                 gamma=0.99,
                 beta=0.5,  # averaging factor between off-policy and on-policy targets during n-step updates
                 tau_actor=0.001,  # Polyak averaging factor for updating target weights
                 tau_critic=0.001,
                 replay_memory=None,  # memory buffer object
                 replay_memory_size=1000000,
                 learning_rate_actor=0.00001,
                 learning_rate_critic=0.001,
                 initial_memory_threshold=0,
                 clip_grad=10,
                 adam_betas=(0.95, 0.999),
                 use_ornstein_noise=False,
                 # if false, uses epsilon-greedy with uniform-random action-parameter exploration
                 loss_func=F.mse_loss,  # F.smooth_l1_loss
                 inverting_gradients=False,
                 n_step_returns=False,
                 seed=None,
                 num_agents=None,
                 unum=None):

        super(MAPDDPGAgent, self).__init__(observation_space, action_space,
                                           actor_kwargs=actor_kwargs,
                                           critic_kwargs=critic_kwargs,
                                           epsilon_initial=epsilon_initial,
                                           epsilon_final=epsilon_final,
                                           epsilon_steps=epsilon_steps,
                                           batch_size=batch_size,
                                           gamma=gamma,
                                           beta=beta,
                                           tau_actor=tau_actor,
                                           tau_critic=tau_critic,
                                           replay_memory=replay_memory,
                                           replay_memory_size=replay_memory_size,
                                           learning_rate_actor=learning_rate_actor,
                                           learning_rate_critic=learning_rate_critic,
                                           initial_memory_threshold=initial_memory_threshold,
                                           clip_grad=clip_grad,
                                           adam_betas=adam_betas,
                                           use_ornstein_noise=use_ornstein_noise,
                                           loss_func=loss_func,
                                           inverting_gradients=inverting_gradients,
                                           n_step_returns=n_step_returns,
                                           seed=seed,
                                           buffer_next_actions=True,
                                           central_critic=True)

        self.redis_instance = connect_redis()
        self.num_agents = num_agents
        self.unum = unum
        self.idx = get_agent_idx(self.redis_instance, str(self.unum))

        self.buffer_next_actions = True

        # initialize replay buffer
        if replay_memory is None:
            self.replay_memory = MemoryNStepReturns(replay_memory_size, (self.obs_dim * num_agents,),
                                                    (num_agents * (self.act_dim + self.action_parameter_size),),
                                                    next_actions=self.buffer_next_actions,
                                                    n_step_returns=self.n_step_returns)
        else:
            self.replay_memory = replay_memory

        # initialize critic networks
        self.critic = critic_class(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                   num_agents=num_agents, param_dim=self.action_parameter_size,
                                   **critic_kwargs).to(device)
        self.critic_target = critic_class(obs_dim=self.obs_dim, act_dim=self.act_dim,
                                          num_agents=num_agents, param_dim=self.action_parameter_size,
                                          **critic_kwargs).to(device)
        hard_update_target_network(self.critic, self.critic_target)
        self.critic_target.eval()
        # initialize optimizers
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic, betas=adam_betas)

        # # modeling other agents' actors
        # self.all_actor_model = []
        # for i in range(self.num_agents):
        #     self.all_actor_model.append(copy.deepcopy(self.actor_target))

    def __str__(self):
        desc = ("MAPDDPG Agent with frozen initial weight layer\n" +
                "Actor: {}\n".format(self.actor) +
                "Critic: {}\n".format(self.critic) +
                "Actor Alpha: {}\n".format(self.learning_rate_actor) +
                "Critic Alpha: {}\n".format(self.learning_rate_critic) +
                "Gamma: {}\n".format(self.gamma) +
                "Tau Actor: {}\n".format(self.tau_actor) +
                "Tau Critic: {}\n".format(self.tau_critic) +
                "Beta: {}\n".format(self.beta) +
                "Inverting Gradients: {}\n".format(self.inverting_gradients) +
                "Replay Memory: {}\n".format(self.replay_memory_size) +
                "epsilon_initial: {}\n".format(self.epsilon_initial) +
                "epsilon_final: {}\n".format(self.epsilon_final) +
                "epsilon_steps: {}\n".format(self.epsilon_steps) +
                "Clip norm: {}\n".format(self.clip_grad) +
                "Batch Size: {}\n".format(self.batch_size) +
                "Ornstein Noise?: {}\n".format(self.use_ornstein_noise) +
                "Seed: {}\n".format(self.seed))
        return desc

    # def step(self, obs, action, reward, next_state, next_action, terminal, time_steps=1, optimise=True):
    #     action, action_params, all_actions, all_action_parameters = action
    #     self._step += 1

    #     self._add_sample(obs, np.concatenate((all_actions.data, all_action_parameters.data)).ravel(), reward,
    #                      next_state, terminal)
    #     if optimise and self._step >= self.batch_size and self._step >= self.initial_memory_threshold:
    #         self.update()

    # def _add_sample(self, obs, action, reward, next_obs, terminal):
    #     assert not self.n_step_returns
    #     assert len(action) == self.act_dim + self.action_parameter_size
    #     self.replay_memory.append(obs, action, reward, next_obs, terminal)

    def update(self):
        if self.replay_memory.nb_entries < self.batch_size or \
                self.replay_memory.nb_entries < self.initial_memory_threshold:
            return

        # Sample a batch from replay memory
        if self.n_step_returns:
            if self.buffer_next_actions:
                all_obs, all_actions, rewards, all_next_obs, all_next_actions, terminals, n_step_returns = \
                    self.replay_memory.sample(self.batch_size, random_machine=self.np_random)
            else:
                all_obs, all_actions, rewards, all_next_obs, terminals, n_step_returns = \
                    self.replay_memory.sample(self.batch_size, random_machine=self.np_random)
        else:
            all_obs, all_actions, rewards, all_next_obs, terminals = \
                self.replay_memory.sample(self.batch_size, random_machine=self.np_random)
            n_step_returns = None

        all_obs = torch.from_numpy(all_obs).to(device)
        obs = all_obs[:, self.idx * self.obs_dim:(self.idx + 1) * self.obs_dim]
        # for n, obs in enumerate(all_obs_combined):
        #     all_obs_combined[n] = [obs[i:i + self.obs_dim] for i in range(0, len(obs), self.obs_dim)]

        actions_combined = torch.from_numpy(all_actions).to(device)  # separate actions and parameters
        all_actions = actions_combined[:, :self.act_dim * self.num_agents]
        all_params = actions_combined[:, self.act_dim * self.num_agents:]

        rewards = torch.from_numpy(rewards).to(device)
        all_next_obs = torch.from_numpy(all_next_obs).to(device)

        if self.buffer_next_actions:
            actions_combined = torch.from_numpy(all_next_actions).to(device)  # separate actions and parameters
            all_next_actions = actions_combined[:, :self.act_dim * self.num_agents]
            all_next_params = actions_combined[:, self.act_dim * self.num_agents:]

        terminals = torch.from_numpy(terminals).to(device)
        if self.n_step_returns:
            n_step_returns = torch.from_numpy(n_step_returns).to(device)

        # ---------------------- optimize critic ----------------------
        with torch.no_grad():
            # # predict next actions of each agent based on their observation with their target actor
            # pred_next_actions = []
            # pred_next_params = []
            # all_policies = query_all_policies(self.redis_instance)
            # for nobs, policy, model_to_be_update in zip(all_next_obs, all_policies, self.all_actor_model):
            #     hard_update_target_network(policy, model_to_be_update)
            #     pred_n_actions, pred_n_params = model_to_be_update.forward(nobs)
            #     pred_next_actions.append(pred_n_actions)
            #     pred_next_params.append(pred_n_params)
            # pred_next_actions = np.array(pred_next_actions)
            # pred_next_params = np.array(pred_next_params)

            # use critic_target to predict target value
            off_policy_next_value = \
                self.critic_target.forward(all_next_obs, all_next_actions, all_next_params)
            off_policy_target = rewards + (1 - terminals) * self.gamma * off_policy_next_value

            if self.n_step_returns:
                on_policy_target = n_step_returns
                target = self.beta * on_policy_target + (1. - self.beta) * off_policy_target
            else:
                target = off_policy_target

        y_expected = target
        # use critic to predict actual value
        y_predicted = self.critic.forward(all_obs, all_actions, all_params)
        loss_critic = self.loss_func(y_predicted, y_expected)

        # update critic
        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad)
        self.critic_optimizer.step()

        # ---------------------- optimize actor ----------------------
        # 1 - calculate gradients from critic
        with torch.no_grad():
            # use actor to make action (with no grad)
            actions, action_params = self.actor(obs)
        #     action_params = torch.cat((actions, action_params), dim=1)
        # action_params.requires_grad = True

        # replace self predicted action
        act_start = self.idx * self.act_dim
        act_end = (self.idx + 1) * self.act_dim
        all_actions[:, act_start:act_end] = actions
        all_params[:, self.idx * self.action_parameter_size:(self.idx + 1) * self.action_parameter_size] = action_params

        all_actions = torch.cat((all_actions, all_params), dim=1)
        all_actions.requires_grad = True
        # use critic and compute its gradients
        Q_val = self.critic(all_obs, all_actions[:, :self.act_dim * self.num_agents],
                            all_actions[:, self.act_dim * self.num_agents:]).mean()
        self.critic.zero_grad()
        Q_val.backward()

        from copy import deepcopy
        delta_a = deepcopy(all_actions.grad.data)

        param_start = self.act_dim * self.num_agents + self.idx * self.action_parameter_size
        param_end = self.act_dim * self.num_agents + (self.idx + 1) * self.action_parameter_size
        delta_a = torch.cat((delta_a[:, act_start:act_end], delta_a[:, param_start:param_end]), dim=1)

        # 2 - apply inverting gradients and combine with gradients from actor
        # use actor to make action again (with grad)
        actions, action_params = self.actor(Variable(obs))
        action_params = torch.cat((actions, action_params), dim=1)

        # invert gradients of actions and parameters separately
        delta_a[:, self.act_dim:] = self._invert_gradients(delta_a[:, self.act_dim:].cpu(),
                                                           action_params[:, self.act_dim:].cpu(),
                                                           grad_type="action_parameters", inplace=True)
        delta_a[:, :self.act_dim] = self._invert_gradients(delta_a[:, :self.act_dim].cpu(),
                                                           action_params[:, :self.act_dim].cpu(),
                                                           grad_type="actions", inplace=True)
        out = -torch.mul(delta_a, action_params)
        self.actor.zero_grad()
        out.backward(torch.ones(out.shape).to(device))

        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimizer.step()

        # ---------------- soft update actor and critic ---------------
        soft_update_target_network(self.actor, self.actor_target, self.tau_actor)
        soft_update_target_network(self.critic, self.critic_target, self.tau_critic)
