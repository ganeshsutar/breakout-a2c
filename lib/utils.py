import time
import torch as T
import torch.nn.functional as F
import numpy as np
import gym
import torch.nn.utils as tutils


eps = np.finfo(np.float32).eps.item()


def getTimePassed(start_time):
    delta = int(time.time() - start_time)
    secs = int(delta % 60)
    delta /= 60
    mins = int(delta % 60)
    hrs = int(delta / 60)

    return '{:02d} h {:02d} m {:02d} secs'.format(hrs, mins, secs)


def getDiscountedReturns(rewards, gamma):
    R = 0.0
    returns = []
    for r in rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def writeLineToFile(filename, line):
    with open(filename, 'a') as fp:
        fp.write(line + '\n')


def train_agent(agent, gamma, device=None):
    returns = getDiscountedReturns(agent.rewards, gamma)
    # writeLineToFile('returns.txt', str(returns))
    # writeLineToFile('values.txt', str([a.item() for a in agent.values]))

    returns = T.tensor(returns)
    log_probs = T.stack(agent.log_probs)
    values = T.stack(agent.values)
    entropy = T.stack(agent.entropy)

    if device is not None:
        returns = returns.to(device)
        log_probs = log_probs.to(device)
        values = values.to(device)

    advantage = returns - values
    actor_loss = (-log_probs * advantage.detach()).mean()
    #critic_loss = advantage.pow(2.0).mean()
    critic_loss = F.smooth_l1_loss(values, returns)
    entropy_loss = entropy.mean()
    loss = actor_loss + critic_loss - 0.01 * entropy_loss

    agent.optimizer.zero_grad()
    loss.backward()
    tutils.clip_grad_norm_(agent.model.parameters(), 40.0)
    agent.optimizer.step()
    
    return loss.item()


class Stats:
    def __init__(self):
        self.mean_rewards = 0.0
        self.mean_timesteps = 0.0
        self.mean_loss = 0.0
        self.start_time = time.time()
        self.episodes = 0
        self.last_reward = 0.0
        self.last_timesteps = 0.0
        self.last_loss = 0.0
        self.max_rewards = 0.0

    def update(self, reward, timesteps, loss):
        self.episodes += 1
        self.mean_rewards += (reward - self.mean_rewards) / self.episodes
        self.mean_timesteps += (timesteps - self.mean_timesteps) / self.episodes
        self.mean_loss += (loss - self.mean_loss) / self.episodes
        self.last_reward = reward
        self.last_timesteps = timesteps
        self.last_loss = loss
        
        if self.max_rewards < reward:
            self.max_rewards = reward
    
    def reset(self):
        self.mean_rewards = 0.0
        self.mean_timesteps = 0.0
        self.mean_loss = 0.0
        #self.start_time = time.time()
        self.episodes = 0
        self.last_reward = 0.0
        self.last_timesteps = 0.0
        self.last_loss = 0.0
        self.max_rewards = 0.0

    def printStats(self):
        print('{}, Total Episodes: {}'.format(getTimePassed(self.start_time), self.episodes))
        print('Max Reward: {}'.format(self.max_rewards))
        print('Mean Reward:    {:10.03f}, Last Reward:    {:10.03f}'.format(self.mean_rewards, self.last_reward))
        print('Mean Timesteps: {:10.03f}, Last Timesteps: {:10.03f}'.format(self.mean_timesteps, self.last_timesteps))
        print('Mean Loss:      {:10.03f}, Last Loss:      {:10.03f}'.format(self.mean_loss, self.last_loss))



