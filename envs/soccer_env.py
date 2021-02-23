import os, subprocess, time, signal
import numpy as np
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import socket
from contextlib import closing

try:
    import hfo_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you can install HFO dependencies with 'pip install gym[soccer].')".format(e))

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


class SoccerEnv(object):
    metadata = {'render.modes': ['human']}

    def __init__(self, hfo):
        super(SoccerEnv, self).__init__()
        self.hfo = hfo
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.hfo.getStateSize(),), dtype=np.float32)

        self.status = hfo_py.IN_GAME
        self._seed = -1

        self.unum = self.hfo.getUnum()  # uniform number (identifier) of our lone agent
        print("UNUM =", self.unum)

    # def __del__(self):
    #     try:
    #         self.hfo.act(hfo_py.QUIT)
    #         self.hfo.step()
    #     finally:
    #         pass

    def step(self, action):
        self._take_action(action)
        self.status = self.hfo.step()
        reward = self._get_reward()
        ob = self.hfo.getState()
        episode_over = self.status
        return ob, reward, episode_over, {'status': STATUS_LOOKUP[self.status]}

    def _take_action(self, action):
        raise NotImplementedError

    def _get_reward(self):
        """ Reward is given for scoring a goal. """
        if self.status == hfo_py.GOAL:
            return 1
        else:
            return 0

    def reset(self):
        """ Repeats NO-OP action until a new episode begins. """
        while self.status == hfo_py.IN_GAME:
            self.hfo.act(hfo_py.NOOP)
            self.status = self.hfo.step()
        while self.status != hfo_py.IN_GAME:
            self.hfo.act(hfo_py.NOOP)
            self.status = self.hfo.step()
            # prevent infinite output when server dies
            if self.status == hfo_py.SERVER_DOWN:
                raise ServerDownException("HFO server down!")
        return self.hfo.getState()



class ServerDownException(Exception):
    """
    Custom error so models can catch it and exit cleanly if the server dies unexpectedly.
    """
    pass


LOW_LEVEL_ACTION_LOOKUP = {
    0: hfo_py.DASH,
    1: hfo_py.TURN,
    2: hfo_py.KICK,
    3: hfo_py.TACKLE,  # Used on defense to slide tackle the ball
    4: hfo_py.CATCH,  # Used only by goalie to catch the ball
}

MID_LEVEL_OFFENSE_ACTION_LOOKUP = {
    0: hfo_py.KICK_TO,
    1: hfo_py.MOVE_TO,
    2: hfo_py.DRIBBLE_TO,
    3: hfo_py.SHOOT,
}

MID_LEVEL_GOALIE_ACTION_LOOKUP = {
    0: hfo_py.KICK_TO,
    1: hfo_py.MOVE_TO,
    2: hfo_py.DRIBBLE_TO,
    3: hfo_py.CATCH,  # Used only by goalie to catch the ball
}

HIGH_LEVEL_ACTION_LOOKUP = {
    0: hfo_py.MOVE,
    1: hfo_py.SHOOT,
    2: hfo_py.DRIBBLE,
    # 3 : hfo_py.CATCH,  # Used only by goalie to catch the ball
}

STATUS_LOOKUP = {
    hfo_py.IN_GAME: 'IN_GAME',
    hfo_py.SERVER_DOWN: 'SERVER_DOWN',
    hfo_py.GOAL: 'GOAL',
    hfo_py.OUT_OF_BOUNDS: 'OUT_OF_BOUNDS',
    hfo_py.OUT_OF_TIME: 'OUT_OF_TIME',
    hfo_py.CAPTURED_BY_DEFENSE: 'CAPTURED_BY_DEFENSE',
}
