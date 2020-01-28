import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
from lib.utils import train_agent
import numpy as np
from skimage.transform import resize


device = T.device('cpu')
if T.cuda.is_available():
    device = T.device('cuda:0')
    print('Using cuda:0')


def preprocess(state):
    return resize(state[35:195].mean(2), (80, 80), mode='reflect') / 255.0


class Policy(nn.Module):
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

        #print(log_prob)

        self.log_probs.append(log_prob)
        self.entropy.append(m.entropy())
        self.values.append(state_value)

        return probs, action.item()

    def getPlayAction(self, env, state):
        state = preprocess(state)
        state = T.from_numpy(state).float().to(device)
        probs, state_value, self.hx = self.model(state, self.hx)

        m = Categorical(T.exp(probs.squeeze()))
        action = m.sample()
        
        return action.item()

        #return T.argmax(T.exp(probs.squeeze())).item()

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

