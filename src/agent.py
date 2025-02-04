from memory import SimpleMemory
from entities import Experience


class Agent:
    def __init__(self, memory):
        self.target = lambda x: 0.0
        self.policy = lambda x: 0.0
        self.memory = memory
