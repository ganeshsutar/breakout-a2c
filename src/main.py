import gym
from gym.wrappers import frame_stack
from rich.progress import track

from entities import Experience, Stat
from memory import SimpleMemory

env = gym.make("Breakout-v4")
env = frame_stack.FrameStack(env, num_stack=4)

memory = SimpleMemory(100_000)

obs, info = env.reset(seed=42)
stats = []
total_reward = 0.0
steps = 0
episode = 1

for _ in track(list(range(100_000))):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    memory.add(Experience(obs, next_obs, action, reward, terminated))
    total_reward += reward
    steps += 1

    if terminated or truncated:
        stats.append(Stat(episode, total_reward, steps))
        obs, info = env.reset(seed=42)
        episode_values = []
        total_reward = 0.0
        steps = 0
        episode += 1

    obs = next_obs

env.close()

for stat in stats:
    print(stat)
