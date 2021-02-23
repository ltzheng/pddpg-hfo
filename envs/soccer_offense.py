import logging
import math
import numpy as np
from gym import error, spaces
from envs.soccer_env import SoccerEnv
from envs.soccer_env import MID_LEVEL_OFFENSE_ACTION_LOOKUP, STATUS_LOOKUP
import hfo_py

logger = logging.getLogger(__name__)


class SoccerOffenseEnv(SoccerEnv):
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
        super(SoccerOffenseEnv, self).__init__(hfo)
        self.old_ball_prox = 0
        self.old_kickable = 0
        self.old_ball_dist_goal = 0
        self.got_kickable_reward = False
        self.first_step = True

        # mid level action space
        self.low0 = np.array([-1, -1, 0], dtype=np.float32)
        self.high0 = np.array([1, 1, 3], dtype=np.float32)
        self.low1 = np.array([-1, -1], dtype=np.float32)
        self.high1 = np.array([1, 1], dtype=np.float32)
        self.low2 = np.array([-1, -1], dtype=np.float32)
        self.high2 = np.array([1, 1], dtype=np.float32)
        self.action_space = spaces.Tuple((spaces.Discrete(4),
                                          spaces.Box(low=self.low0, high=self.high0, dtype=np.float32),
                                          spaces.Box(low=self.low1, high=self.high1, dtype=np.float32),
                                          spaces.Box(low=self.low2, high=self.high2, dtype=np.float32)))

        # # low level action space
        # # omits the Tackle/Catch actions, which are useful on defense
        # self.low0 = np.array([0, -180], dtype=np.float32)
        # self.high0 = np.array([100, 180], dtype=np.float32)
        # self.low1 = np.array([-180], dtype=np.float32)
        # self.high1 = np.array([180], dtype=np.float32)
        # self.low2 = np.array([0, -180], dtype=np.float32)
        # self.high2 = np.array([100, 180], dtype=np.float32)
        # self.low3 = np.array([-180], dtype=np.float32)
        # self.high3 = np.array([180], dtype=np.float32)
        # # dash, turn, kick
        # self.action_space = spaces.Tuple((spaces.Discrete(3),
        #                                   spaces.Box(low=self.low0, high=self.high0, dtype=np.float32),
        #                                   spaces.Box(low=self.low1, high=self.high1, dtype=np.float32),
        #                                   spaces.Box(low=self.low2, high=self.high2, dtype=np.float32)))

    # take mid level actions
    def _take_action(self, action):
        """ Converts the action space into an HFO action. """
        action_type = MID_LEVEL_OFFENSE_ACTION_LOOKUP[action[0]]
        if action_type == hfo_py.KICK_TO:
            np.clip(action[1:4], self.low0, self.high0, out=action[1:4])
            self.hfo.act(action_type, action[1], action[2], action[3])
        elif action_type == hfo_py.MOVE_TO:
            np.clip(action[4:6], self.low1, self.high1, out=action[4:6])
            self.hfo.act(action_type, action[4], action[5])
        elif action_type == hfo_py.DRIBBLE_TO:
            np.clip(action[6:8], self.low2, self.high2, out=action[6:8])
            self.hfo.act(action_type, action[6], action[7])
        elif action_type == hfo_py.SHOOT:
            self.hfo.act(action_type)
        else:
            print('Unrecognized action %d' % action_type)
            self.hfo.act(hfo_py.NOOP)
        # print('\rTaking action:', 'type:', action_type, 'params:', action[1:], end='')

    # # take low level actions
    # def _take_action(self, action):
    #     """ Converts the action space into an HFO action. """
    #     action_type = ACTION_LOOKUP[action[0]]
    #     if action_type == hfo_py.DASH:
    #         self.env.act(action_type, action[1], action[2])
    #     elif action_type == hfo_py.TURN:
    #         self.env.act(action_type, action[3])
    #     elif action_type == hfo_py.KICK:
    #         self.env.act(action_type, action[4], action[5])
    #     elif action_type == hfo_py.TACKLE:
    #         self.env.act(action_type, action[6])
    #     else:
    #         print('Unrecognized action %d' % action_type)
    #         self.env.act(hfo_py.NOOP)

    # def _get_reward(self):
    #     """
    #     Agent is rewarded for minimizing the distance between itself and
    #     the ball, minimizing the distance between the ball and the goal,
    #     and scoring a goal.
    #     """
    #     current_state = self.hfo.getState()
    #     # print("State =",current_state)
    #     # print("len State =",len(current_state))
    #     ball_proximity = current_state[53]
    #     goal_proximity = current_state[15]
    #     ball_dist = 1.0 - ball_proximity
    #     goal_dist = 1.0 - goal_proximity
    #     kickable = current_state[12]
    #     ball_ang_sin_rad = current_state[51]
    #     ball_ang_cos_rad = current_state[52]
    #     ball_ang_rad = math.acos(ball_ang_cos_rad)
    #     if ball_ang_sin_rad < 0:
    #         ball_ang_rad *= -1.
    #     goal_ang_sin_rad = current_state[13]
    #     goal_ang_cos_rad = current_state[14]
    #     goal_ang_rad = math.acos(goal_ang_cos_rad)
    #     if goal_ang_sin_rad < 0:
    #         goal_ang_rad *= -1.
    #     alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad, goal_ang_rad)
    #     ball_dist_goal = math.sqrt(ball_dist * ball_dist + goal_dist * goal_dist -
    #                                2. * ball_dist * goal_dist * math.cos(alpha))
    #     # Compute the difference in ball proximity from the last step
    #     if not self.first_step:
    #         ball_prox_delta = ball_proximity - self.old_ball_prox
    #         kickable_delta = kickable - self.old_kickable
    #         ball_dist_goal_delta = ball_dist_goal - self.old_ball_dist_goal
    #     self.old_ball_prox = ball_proximity
    #     self.old_kickable = kickable
    #     self.old_ball_dist_goal = ball_dist_goal
    #     # print(self.env.playerOnBall())
    #     # print(self.env.playerOnBall().unum)
    #     # print(self.env.getUnum())
    #     reward = 0
    #     if not self.first_step:
    #         '''# Reward the agent for moving towards the ball
    #         reward += ball_prox_delta
    #         if kickable_delta > 0 and not self.got_kickable_reward:
    #             reward += 1.
    #             self.got_kickable_reward = True
    #         # Reward the agent for kicking towards the goal
    #         reward += 0.6 * -ball_dist_goal_delta
    #         # Reward the agent for scoring
    #         if self.status == hfo_py.GOAL:
    #             reward += 5.0'''
    #         '''reward = self.__move_to_ball_reward(kickable_delta, ball_prox_delta) + \
    #                 3. * self.__kick_to_goal_reward(ball_dist_goal_delta) + \
    #                 self.__EOT_reward();'''
    #         mtb = self.__move_to_ball_reward(kickable_delta, ball_prox_delta)
    #         ktg = 3. * self.__kick_to_goal_reward(ball_dist_goal_delta)
    #         eot = self.__EOT_reward()
    #         reward = mtb + ktg + eot
    #         # print("mtb: %.06f ktg: %.06f eot: %.06f"%(mtb,ktg,eot))
    #
    #     self.first_step = False
    #     # print("r =",reward)
    #     return reward

    def __move_to_ball_reward(self, kickable_delta, ball_prox_delta):
        reward = 0.
        if self.hfo.playerOnBall().unum < 0 or self.hfo.playerOnBall().unum == self.unum:
            reward += ball_prox_delta
        if kickable_delta >= 1 and not self.got_kickable_reward:
            reward += 1.
            self.got_kickable_reward = True
        return reward

    def __kick_to_goal_reward(self, ball_dist_goal_delta):
        if (self.hfo.playerOnBall().unum == self.unum):
            return -ball_dist_goal_delta
        elif self.got_kickable_reward == True:
            return 0.2 * -ball_dist_goal_delta
        return 0.

    def __EOT_reward(self):
        if self.status == hfo_py.GOAL:
            return 5.
        elif self.status == hfo_py.CAPTURED_BY_DEFENSE:
            return -1.
        return 0.

    def reset(self):
        self.old_ball_prox = 0
        self.old_kickable = 0
        self.old_ball_dist_goal = 0
        self.got_kickable_reward = False
        self.first_step = True
        return super(SoccerOffenseEnv, self).reset()
