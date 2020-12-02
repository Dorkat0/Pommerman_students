'''An agent that preforms a random action each step'''
from pommerman.agents import BaseAgent


class RandomNoBombAgent(BaseAgent):
    """The Random Agent that returns random actions given an action_space."""

    def act(self, obs, action_space):
        action = action_space.sample()
        while action == 5:
            action = action_space.sample()
        return action
