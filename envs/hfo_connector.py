import os
import signal
import socket
import subprocess
import time
from contextlib import closing

from gym import error

try:
    import hfo_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (Try 'pip install -e .' to install HFO dependencies.')".format(e))

import logging

logger = logging.getLogger(__name__)


def find_free_port():
    """Find a random free port. Does not guarantee that the port will still be free after return.
    Note: HFO takes three consecutive port numbers, this only checks one.

    Source: https://github.com/crowdAI/marLo/blob/master/marlo/utils.py

    :rtype:  `int`
    """

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class HFOConnector(object):
    def __init__(self):
        self.viewer = None
        self.server_process = None
        self.server_port = 6000
        self.hfo_path = hfo_py.get_hfo_path()
        print('HFO path: ', self.hfo_path)

    def __del__(self):
        os.kill(self.server_process.pid, signal.SIGINT)
        if self.viewer is not None:
            os.kill(self.viewer.pid, signal.SIGKILL)

    def start_hfo_server(self, frames_per_trial=1000,
                         untouched_time=100, start_viewer=True,
                         offense_agents=1,
                         defense_agents=0, offense_npcs=0,
                         defense_npcs=0, sync_mode=True, port=None,
                         offense_on_ball=0, fullstate=True, seed=-1,
                         ball_x_min=0.0, ball_x_max=0.2,
                         verbose=False, log_game=False,
                         agent_play_goalie=True,
                         log_dir="log"):
        """
        Starts the Half-Field-Offense server.
        frames_per_trial: Episodes end after this many steps.
        untouched_time: Episodes end if the ball is untouched for this many steps.
        offense_agents: Number of user-controlled offensive players.
        defense_agents: Number of user-controlled defenders.
        offense_npcs: Number of offensive bots.
        defense_npcs: Number of defense bots.
        sync_mode: Disabling sync mode runs server in real time (SLOW!).
        port: Port to start the server on.
        offense_on_ball: Player to give the ball to at beginning of episode.
        fullstate: Enable noise-free perception.
        seed: Seed the starting positions of the players and ball.
        ball_x_[min/max]: Initialize the ball this far downfield: [0,1]
        verbose: Verbose server messages.
        log_game: Enable game logging. Logs can be used for replay + visualization.
        log_dir: Directory to place game logs (*.rcg).
        """
        if port is None:
            port = find_free_port()
        self.server_port = port
        cmd = self.hfo_path + " --frames-per-trial %i" \
                              " --offense-npcs %i --defense-npcs %i --port %i --offense-on-ball %i --seed %i" \
                              " --ball-x-min %f --ball-x-max %f --log-dir %s" \
                              % (frames_per_trial, offense_npcs, defense_npcs,
                                 port, offense_on_ball, seed, ball_x_min, ball_x_max, log_dir)
        if offense_agents:
            cmd += ' --offense-agents %i' % offense_agents
        if defense_agents:
            cmd += ' --defense-agents %i' % defense_agents
        if not start_viewer:
            cmd += ' --headless'
        if not sync_mode:
            cmd += " --no-sync"
        if fullstate:
            cmd += " --fullstate"
        if verbose:
            cmd += " --verbose"
        if not log_game:
            cmd += " --no-logging"
        if agent_play_goalie:
            cmd += " --agent-play-goalie"
        print('Starting server with command: %s' % cmd)
        # return cmd
        self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
        print('HFO Server Connected')
        self.server_process.wait()
        # time.sleep(5)  # Wait for server to startup before connecting a player

    def _start_viewer(self):
        """
        Starts the SoccerWindow visualizer. Note the viewer may also be
        used with a *.rcg logfile to replay a game. See details at
        https://github.com/LARG/HFO/blob/master/doc/manual.pdf.
        """
        cmd = hfo_py.get_viewer_path() + " --connect --port %d" % (self.server_port)
        self.viewer = subprocess.Popen(cmd.split(' '), shell=False)

    def _render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        if close:
            if self.viewer is not None:
                os.kill(self.viewer.pid, signal.SIGKILL)
        else:
            if self.viewer is None:
                self._start_viewer()

    def close(self):
        if self.server_process is not None:
            try:
                os.kill(self.server_process.pid, signal.SIGKILL)
            except Exception:
                pass


# class SoccerMultiAgentEnv:
#     def __init__(self, num_offense_agents=1, num_defense_agents=1, defense_goalie=True, share_reward=True):
#         self.t = {}
#         self.observations = {}
#         self.rewards = {}
#         self.actions = {}
#
#     def register(self, agent_name, agent_type):
#         self.t[agent_name] = agent_type
#         self.observations[agent_name] = []
#         self.actions[agent_name] = []
#         self.rewards[agent_name] = None
#
#     def collect_obs(self, agent_name, features):
#         self.observations[agent_name] = features
#
#     def get_action(self, agent_name):
#         return self.actions[agent_name]
#
#     def get_reward(self, agent_name, features):
#         return self.rewards[agent_name]
#
#     def reset(self):
#         pass
#         # obs_n = []
#         # for agent in self.t:
#         #     obs_n.append(agent.env.getState())
#         #     """ Repeats NO-OP action until a new episode begins. """
#         #     while agent.status == hfo_py.IN_GAME:
#         #         agent.env.act(hfo_py.NOOP)
#         #         agent.status = agent.env.step()
#         #     while agent.status != hfo_py.IN_GAME:
#         #         agent.env.act(hfo_py.NOOP)
#         #         agent.status = agent.env.step()
#         #         # prevent infinite output when server dies
#         #         if agent.status == hfo_py.SERVER_DOWN:
#         #             raise ServerDownException("HFO server down!")
#         # return obs_n

