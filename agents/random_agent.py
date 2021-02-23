from agents.base_agent import BaseAgent

class RandomAgent(BaseAgent):
    """
    A random agent that acts uniformly randomly.
    """
    def __init__(self, observation_space, action_space):
        super(RandomAgent, self).__init__(observation_space, action_space)

    def act(self, state):
        # print('self.action_space.sample():', self.action_space.sample())
        random_action = self.action_space.sample()
        return random_action[0], random_action[1:]

    # def step(self, state, action, reward, next_state, next_action, terminal, time_steps=1):
    #     pass
