import numpy as np # type: ignore

class RandomAgent:
    def __init__(self):
        pass

    def act(self, observation):
        legal_actions = np.where(observation['action_mask'] == 1)[0]
        action = np.random.choice(legal_actions)
        return action
