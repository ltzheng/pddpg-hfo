class BaseAgent(object):

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_dim = self.observation_space.shape[0]  # dimension for single agent
        self.act_dim = self.action_space.spaces[0].n

    def act(self, state):
        """
        Returns action with parameters to take in given state.
        """
        raise NotImplementedError

    # def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1):
    #     """
    #     Performs a learning step given a (s,a,r,s',a') sample.
    #
    #     :param state: previous observed state (s)
    #     :param action: action taken in previous state (a)
    #     :param reward: reward for the transition (r)
    #     :param next_state: the resulting observed state (s')
    #     :param next_action: action taken in next state (a')
    #     :param terminal: whether the episode is over
    #     :param time_steps: number of time steps the action took to execute (default=1)
    #     :return:
    #     """
    #     raise NotImplementedError