import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('last_state', 'last_action', 'reward', 'current_state', 'terminal'))


class ReplayMemory(object):
    """
    This class saves transitions that are used for optimization.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # save a transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # sample a batch uniformly from memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
