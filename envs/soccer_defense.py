import logging
import math
import numpy as np
from gym import error, spaces
from envs.soccer_offense import SoccerOffenseEnv
from envs.soccer_env import MID_LEVEL_GOALIE_ACTION_LOOKUP, STATUS_LOOKUP
import hfo_py

logger = logging.getLogger(__name__)


class SoccerDefenseEnv(SoccerOffenseEnv):
    """
    SoccerScoreGoal is the same task as SoccerEmptyGoal, which tasks the 
    agent with approaching the ball, dribbling, and scoring a goal. Rewards 
    are given as the agent nears the ball, kicks the ball towards the goal, 
    and scores a goal.

    The difference is that the reward structure is altered to be consistent
    with the Hausknecht paper: "Deep Reinforcement Learning with Parameterised
    Action Spaces".

    """

    def __init__(self, hfo):
        super(SoccerDefenseEnv, self).__init__(hfo)

        # mid level action space
        self.action_space = spaces.Tuple((spaces.Discrete(3),
                                          spaces.Box(low=self.low0, high=self.high0, dtype=np.float32),
                                          spaces.Box(low=self.low1, high=self.high1, dtype=np.float32),
                                          spaces.Box(low=self.low2, high=self.high2, dtype=np.float32)))

    # take mid level actions
    def _take_action(self, action):
        """ Converts the action space into an HFO action. """
        action_type = MID_LEVEL_GOALIE_ACTION_LOOKUP[action[0]]
        if action_type == hfo_py.KICK_TO:
            np.clip(action[1:4], self.low0, self.high0, out=action[1:4])
            self.hfo.act(action_type, action[1], action[2], action[3])
        elif action_type == hfo_py.MOVE_TO:
            np.clip(action[4:6], self.low1, self.high1, out=action[4:6])
            self.hfo.act(action_type, action[4], action[5])
        elif action_type == hfo_py.DRIBBLE_TO:
            np.clip(action[6:8], self.low2, self.high2, out=action[6:8])
            self.hfo.act(action_type, action[6], action[7])
        else:
            print('Unrecognized action %d' % action_type)
            self.hfo.act(hfo_py.NOOP)
        # print('\rTaking action:', 'type:', action_type, 'params:', action[1:], end='')

    def _get_reward(self):
        """
        Agent is rewarded for minimizing the distance between itself and
        the ball, minimizing the distance between the ball and the goal,
        and scoring a goal.
        """
        current_state = self.hfo.getState()
        ball_proximity = current_state[53]
        goal_proximity = current_state[15]
        ball_dist = 1.0 - ball_proximity
        goal_dist = 1.0 - goal_proximity
        kickable = current_state[12]
        ball_ang_sin_rad = current_state[51]
        ball_ang_cos_rad = current_state[52]
        ball_ang_rad = math.acos(ball_ang_cos_rad)
        if ball_ang_sin_rad < 0:
            ball_ang_rad *= -1.
        goal_ang_sin_rad = current_state[13]
        goal_ang_cos_rad = current_state[14]
        goal_ang_rad = math.acos(goal_ang_cos_rad)
        if goal_ang_sin_rad < 0:
            goal_ang_rad *= -1.
        alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad, goal_ang_rad)
        ball_dist_goal = math.sqrt(ball_dist * ball_dist + goal_dist * goal_dist -
                                   2. * ball_dist * goal_dist * math.cos(alpha))
        # Compute the difference in ball proximity from the last step
        if not self.first_step:
            ball_prox_delta = ball_proximity - self.old_ball_prox
            kickable_delta = kickable - self.old_kickable
            ball_dist_goal_delta = ball_dist_goal - self.old_ball_dist_goal
        self.old_ball_prox = ball_proximity
        self.old_kickable = kickable
        self.old_ball_dist_goal = ball_dist_goal
        reward = 0
        if not self.first_step:
            mtb = self.__move_to_ball_reward(kickable_delta, ball_prox_delta)
            ktg = 3. * self.__kick_to_goal_reward(ball_dist_goal_delta)
            eot = self.__EOT_reward()
            reward = mtb + ktg + eot

        self.first_step = False
        return reward

    def __move_to_ball_reward(self, kickable_delta, ball_prox_delta):
        reward = 0.
        if self.hfo.playerOnBall().unum < 0 or self.hfo.playerOnBall().unum == self.unum:
            reward += ball_prox_delta
        if kickable_delta >= 1 and not self.got_kickable_reward:
            reward += 1.
            self.got_kickable_reward = True
        return reward

    def __kick_to_goal_reward(self, ball_dist_goal_delta):
        if self.hfo.playerOnBall().unum == self.unum:
            return -ball_dist_goal_delta
        elif self.got_kickable_reward:
            return 0.2 * -ball_dist_goal_delta
        return 0.

    def __EOT_reward(self):
        if self.status == hfo_py.GOAL:
            return -1.
        elif self.status == hfo_py.CAPTURED_BY_DEFENSE:
            return 5.
        return 0.

    def reset(self):
        self.old_ball_prox = 0
        self.old_kickable = 0
        self.old_ball_dist_goal = 0
        self.got_kickable_reward = False
        self.first_step = True
        return super(SoccerDefenseEnv, self).reset()
