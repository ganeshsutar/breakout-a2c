import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
from skimage.transform import resize
import gym
import os
import time


num_episodes = 10000
max_timesteps = 200
log_interval = 100
save_interval = 100
gamma = 0.99
checkpointFile = 'checkpoints/breakout-cp.pt'
eps = np.finfo(np.float32).eps.item()


device = T.device('cpu')
if T.cuda.is_available():
    device = T.device('cuda:0')
    print('Using cuda:0')


def preprocess(state):
    return resize(state[35:195].mean(2), (80, 80), mode='reflect') / 255.0


class Policy(nn.Module):
    """Policy Network"""
    def __init__(self):
        super(Policy, self).__init__()
        self.flat_dim = 32 * 5 * 5
        self.hidden_dim = 512
        
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.rnn = nn.GRU(self.flat_dim, self.hidden_dim, batch_first=True)
        
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 200),
            nn.ReLU(),
            nn.Linear(200, 4)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )
    
    def forward(self, x, hx = None):
        x = x.view(-1, 1, 80, 80)
        x = self.cnn(x)
        x = x.view(-1, 1, self.flat_dim)
        x, hx = self.rnn(x, hx)
        
        action_probs = F.log_softmax(self.action_head(hx).squeeze(), dim=0)
        state_value = self.value_head(hx).squeeze()

        return action_probs, state_value, hx


class BreakoutAgent:
    """Agent for training and keeping current episodes info"""
    def __init__(self, model = None):
        self.model = model if model is not None else Policy()
        self.optimizer = optim.SGD(self.model.parameters(), lr=3e-4)

        self.log_probs = []
        self.entropy = []
        self.values = []
        self.rewards = []
        self.hx = None
        
        self.gamma = 0.99
        self.model.to(device)

    def getAction(self, env, state):
        state = preprocess(state)
        state = T.from_numpy(state).float().to(device)
        probs, state_value, self.hx = self.model(state, self.hx)
        
        m = Categorical(logits=probs.squeeze())
        action = m.sample()
        log_prob = m.log_prob(action)

        self.log_probs.append(log_prob)
        self.entropy.append(m.entropy())
        self.values.append(state_value)

        return probs, action.item()

    def feedback(self, reward):
        self.rewards.append(reward)

    def learn(self):
        loss = train_agent(self, self.gamma, device)

        del self.log_probs[:]
        del self.rewards[:]
        del self.values[:]
        del self.entropy[:]
        self.hx = None
        
        return loss

    def load(self, filename):
        self.model.load_state_dict(T.load(filename))

    def save(self, filename):
        T.save(self.model.state_dict(), filename)
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()


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


def train_agent(agent, gamma, device=None):
    returns = getDiscountedReturns(agent.rewards, gamma)

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
    critic_loss = advantage.pow(2.0).mean()
    entropy_loss = entropy.mean()
    loss = actor_loss + critic_loss - 0.01 * entropy_loss

    agent.optimizer.zero_grad()
    loss.backward()
    tutils.clip_grad_norm_(agent.model.parameters(), 40.0)
    agent.optimizer.step()
    
    return loss.item()


class Stats:
    """To keeping the stats across episodes"""
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




