import numpy as np
from entities import Experience


class SimpleMemory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.memory = []

    def add(self, experience: Experience):
        if len(self.memory) >= self.max_size:
            self.memory.pop(0)
        self.memory.append(experience)

    def discount_rewards(self, rewards, gamma=0.99):
        discounted_rewards = []
        running_add = 0.0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_rewards.append(running_add)
        return discounted_rewards

    def add_episode(self, experiences: list[Experience]):
        rewards = [exp.reward for exp in experiences]
        discounted_rewards = self.discount_rewards(rewards)
        for i, exp in enumerate(experiences):
            self.add(
                Experience(
                    exp.state,
                    exp.next_state,
                    exp.action,
                    discounted_rewards[i],
                    exp.done,
                )
            )

    def sample(self, batch_size):
        choices = np.random.choice(len(self.memory), batch_size)
        return [self.memory[i] for i in choices]
