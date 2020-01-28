import gym
import os
import time
import lib.utils as utils
from agents.breakout_agent import BreakoutAgent
import torch.multiprocessing as mp


num_episodes = 10000
max_timesteps = 200
log_interval = 100
save_interval = 100
gamma = 0.99
eps = 0.0001
checkpointFile = 'checkpoints/breakout-cp.pt'


def trainModel(model):
    env = gym.make('Breakout-v0')

    start_time = time.time()
    agent = BreakoutAgent(model)
    stats = utils.Stats()
    last_lives = 5.0

    agent.train()
    for episode in range(1, num_episodes+1):
        current_state = env.reset()
        ep_reward = 0.0

        for t in range(1, max_timesteps+1):
            probs, action = agent.getAction(env, current_state)
            current_state, reward, done, info = env.step(action)

            if last_lives != info['ale.lives']:
                reward = -1.0
                last_lives = info['ale.lives']

            if done:
                reward = -1.0

            agent.feedback(reward)
            ep_reward += reward

            if done:
                break

        loss = agent.learn()
        stats.update(ep_reward, t, loss)

        if (episode % log_interval) == 0:
            print(probs)
            stats.printStats()
            stats.reset()

        if (episode % save_interval) == 0:
            agent.save(checkpointFile)

    env.close()


if __name__ == '__main__':
    agent = BreakoutAgent()
    
    if os.path.exists(checkpointFile):
        agent.load(checkpointFile)

    trainModel(agent.model)

