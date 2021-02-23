import click
import ast
import hfo_py
import numpy as np


class ClickPythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except Exception as e:
            print(e)
            raise click.BadParameter(value)


def configure_agent(player, team, port=6000, feature='low', address='localhost', record_dir=None):
    is_goalie = True if player == 'goalie' else False

    feature_set = hfo_py.LOW_LEVEL_FEATURE_SET if feature == 'low' else hfo_py.HIGH_LEVEL_FEATURE_SET

    agent_args = {'config_dir': hfo_py.get_hfo_path()[:-3] + 'teams/base/config/formations-dt',
                  'feature_set': feature_set,  # High or low level state features
                  'server_port': port,  # port to connect to server on
                  'server_addr': address,  # address of server
                  'team_name': team,  # name of self.team to join
                  'play_goalie': is_goalie,  # is this player the goalie
                  }
    if record_dir:
        agent_args['record_dir'] = record_dir  # record agent's states/actions/rewards to this directory
    return agent_args


def soft_update_target_network(source_network, target_network, tau):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def hard_update_target_network(source_network, target_network):
    for target_param, param in zip(target_network.parameters(), source_network.parameters()):
        target_param.data.copy_(param.data)


def compute_n_step_returns(episode_transitions, gamma):
    n = len(episode_transitions)
    n_step_returns = np.zeros((n,))
    n_step_returns[n - 1] = episode_transitions[n - 1][2]  # Q-value is just the final reward
    for i in range(n - 2, 0, -1):
        reward = episode_transitions[i][2]
        target = n_step_returns[i + 1]
        n_step_returns[i] = reward + gamma * target
    return n_step_returns


class OrnsteinUhlenbeckActionNoise(object):
    """
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    Source: https://github.com/vy007vikas/PyTorch-ActorCriticRL/blob/master/utils.py
    """

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2, random_machine=np.random):
        super(OrnsteinUhlenbeckActionNoise, self).__init__()
        self.random = random_machine
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * self.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X
