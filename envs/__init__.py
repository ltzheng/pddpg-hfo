import numpy as np


# mid level
def offense_mid_action(act, act_param):
    action = np.zeros((8,))
    action[0] = act
    if act == 0:
        action[[1, 2, 3]] = act_param
    elif act == 1:
        action[[4, 5]] = act_param
    elif act == 2:
        action[[6, 7]] = act_param
    elif act == 3:
        pass
    else:
        raise ValueError("Unknown action index '{}'".format(act))
    return action


def goalie_mid_action(act, act_param):
    action = np.zeros((8,))
    action[0] = act
    if act == 0:
        action[[1, 2, 3]] = act_param
    elif act == 1:
        action[[4, 5]] = act_param
    elif act == 2:
        action[[6, 7]] = act_param
    elif act == 3:
        pass
    else:
        raise ValueError("Unknown action index '{}'".format(act))
    return action


def defense_mid_action(act, act_param):
    action = np.zeros((8,))
    action[0] = act
    if act == 0:
        action[[1, 2, 3]] = act_param
    elif act == 1:
        action[[4, 5]] = act_param
    elif act == 2:
        action[[6, 7]] = act_param
    else:
        raise ValueError("Unknown action index '{}'".format(act))
    return action


# low level
def low_action(act, act_param):
    action = np.zeros((7,))
    action[0] = act
    if act == 0:  # DASH
        action[[1, 2]] = act_param
    elif act == 1:  # TURN
        action[3] = act_param
    elif act == 2:  # KICK
        action[[4, 5]] = act_param
    # elif act == 3:
    #     action[[6]] = act_param
    else:
        raise ValueError("Unknown action index '{}'".format(act))
    return action
