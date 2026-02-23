import random
from collections import namedtuple, deque


class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.transition_tuple: namedtuple = namedtuple(typename='Transition', field_names=(
        'current_state', 'agent_action', 'reward', 'next_state', 'is_done'))

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition_tuple(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
