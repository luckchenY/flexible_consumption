import torch
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ReplayBufferQue:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
    def push(self,transitions):
        '''_summary_
        Args:
            trainsitions (tuple): _description_
        '''
        self.buffer.append(transitions)
    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential: # sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
    def clear(self):
        self.buffer.clear()
    def __len__(self):
        return len(self.buffer)

class PGReplay(ReplayBufferQue):
    '''replay buffer for policy gradient based methods, each time these methods will sample all transitions
    Args:
        ReplayBufferQue (_type_): _description_
    '''
    def __init__(self):
        self.buffer = deque()
    def sample(self):
        ''' sample all the transitions
        '''
        batch = list(self.buffer)
        return zip(*batch)

class ActorSoftmax(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(ActorSoftmax, self).__init__()
        init_w=3e-3
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self,state):
        dist = F.relu(self.fc1(state))
        dist = F.relu(self.fc2(dist))
        dist = F.softmax(self.fc3(dist),dim=1)
        return dist

class Critic(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=256):
        super(Critic,self).__init__()
        init_w=3e-3
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
    def forward(self,state):
        value = F.relu(self.fc1(state))
        value = F.relu(self.fc2(value)) 
        value = self.fc3(value)
        return value
        
class A2C:
    def __init__(self,n_states,n_actions,cfg):
        self.n_actions = n_actions
        self.gamma = cfg.gamma 
        self.device = cfg.device
        self.memory = PGReplay()
        self.actor = ActorSoftmax(n_states, n_actions, hidden_dim=cfg.actor_hidden_dim).to(self.device)
        self.critic = Critic(n_states, 1, hidden_dim=cfg.critic_hidden_dim).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
    def sample_action(self,state):
        #state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        dist = self.actor(state)
        value = self.critic(state) # note that 'dist' need require_grad=True
        value = value.detach().cpu().numpy().squeeze(0)[0]
        action = np.random.choice(self.n_actions, p=dist.detach().cpu().numpy().squeeze(0)) # shape(p=(n_actions,1)
        return action,value,dist 
    # def predict_action(self,state):
    #     state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
    #     dist = self.actor(state)
    #     value = self.critic(state) # note that 'dist' need require_grad=True
    #     value = value.detach().cpu().numpy().squeeze(0)[0]
    #     action = np.random.choice(self.n_actions, p=dist.detach().numpy().squeeze(0)) # shape(p=(n_actions,1)
    #     return action,value,dist
    def update(self,next_state,entropy):
        value_pool,log_prob_pool,reward_pool = self.memory.sample()
        next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        next_value = self.critic(next_state)
        returns = np.zeros_like(reward_pool)
        for t in reversed(range(len(reward_pool))):
            next_value = reward_pool[t] + self.gamma * next_value # G(s_{t},a{t}) = r_{t+1} + gamma * V(s_{t+1})
            returns[t] = next_value
        returns = torch.tensor(returns, device=self.device)
        value_pool = torch.tensor(value_pool, device=self.device)
        advantages = returns - value_pool
        log_prob_pool = torch.stack(log_prob_pool)
        actor_loss = (-log_prob_pool * advantages).mean()
        critic_loss = 0.5 * advantages.pow(2).mean()
        tot_loss = actor_loss + critic_loss + 0.001 * entropy
        self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        tot_loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()
        self.memory.clear() #onpolicy算法 数据使用过后就清除不复用
    def save_model(self, path):
        from pathlib import Path
        # create path
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{path}/actor_checkpoint.pt")
        torch.save(self.critic.state_dict(), f"{path}/critic_checkpoint.pt")

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(f"{path}/actor_checkpoint.pt"))
        self.critic.load_state_dict(torch.load(f"{path}/critic_checkpoint.pt"))