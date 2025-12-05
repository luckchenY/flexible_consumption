import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
class Actor(nn.Module):
    """策略网络，交互时仅存储原始数据，更新时重新计算log_prob和entropy"""
    def __init__(self, state_dim, action_dim, cfg,consumptionflag):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.mean_layer = nn.Linear(cfg.hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(cfg.hidden_dim, action_dim)
        self.log_std_min = -10   # 更稳健
        self.log_std_max = 1     # 避免过大标准差

        
        # 超参数
        self.gamma = cfg.gamma
        self.entropy_coef = cfg.entropy_coef
        self.device = cfg.device
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=cfg.actor_lr)
        
        # 交互时仅存储原始数据（不存储log_prob和entropy）
        self.states = []  # 存储状态
        self.actions = []  # 存储已执行的动作
        self.rewards = []  # 存储奖励
        self.next_states = []  # 存储下一个状态
        self.dones = []  # 存储终止标志
        
        self.to(cfg.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample_action(self, state):
        """根据当前状态采样一个动作，并返回动作、对数概率和熵"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        action = torch.tanh(x_t)  # 将动作映射到[-1, 1]
        
        # 计算对数概率
        log_prob = normal.log_prob(x_t)
        # 修正tanh带来的概率变化
        log_prob -= torch.log(torch.clamp(1 - action.pow(2), min=1e-6))
        log_prob = log_prob.sum(-1, keepdim=True)
        
        # 计算熵
        entropy = normal.entropy().sum(-1, keepdim=True)
        
        return action, log_prob, entropy

    def select_action(self, state):
        """选择动作"""
        # 明确指定dtype为torch.float32，确保类型一致性
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.sample_action(state)
        return action.cpu().numpy().flatten()
    
    def store_transition(self, state, action, reward, next_state, done):
        """仅存储原始交互数据"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def update(self, critic):
        """更新时重新计算log_prob和entropy"""
        if not self.states:  # 没有数据则不更新
            return 0.0
            
        # 将存储的原始数据转换为张量
        states = torch.tensor(self.states, dtype=torch.float32).to(self.device)  # (T, state_dim)
        actions = torch.tensor(self.actions, dtype=torch.float32).to(self.device)  # (T, action_dim)
        rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device).unsqueeze(1)  # (T, 1)
        next_states = torch.tensor(self.next_states, dtype=torch.float32).to(self.device)  # (T, state_dim)
        dones = torch.tensor(self.dones, dtype=torch.float32).to(self.device).unsqueeze(1)  # (T, 1)
        
        # 清空存储的原始数据
        self.clear_memory()
        
        # -------------------------- 关键：重新计算log_prob和entropy --------------------------
        _, log_probs, entropies = self.sample_action(states)
        
        # -------------------------- 计算损失并更新 --------------------------
        # 使用共享critic计算价值
        values = critic(states)  # (T, 1)
        with torch.no_grad():
            next_values = critic(next_states)  # (T, 1)
            targets = rewards + (1 - dones) * self.gamma * next_values  # (T, 1)
        
        # 计算优势
        advantages = targets - values  # (T, 1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # 计算策略损失
        actor_loss = -(log_probs * advantages.detach()).mean() - self.entropy_coef * entropies.mean()
        
        # 更新策略网络
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        
        return actor_loss.item()
    
    def clear_memory(self):
        """清空存储的原始数据"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

class SharedCritic(nn.Module):
    """共享价值网络，负责所有状态的价值估计和自身更新"""
    def __init__(self, state_dim, cfg):
        super(SharedCritic, self).__init__()
        self.fc1 = nn.Linear(state_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.fc3 = nn.Linear(cfg.hidden_dim, 1)  # 输出状态价值
        
        # 超参数
        self.gamma = cfg.gamma
        self.device = cfg.device
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=cfg.critic_lr)
        
        # 存储所有需要用于更新的数据
        self.all_states = []
        self.all_next_states=[]
        self.all_rewards= []
        self.all_dones=[]
        
        self.to(cfg.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value
    
    def collect_data(self, state, next_state, reward, done):
        """收集单个样本并计算目标值"""
        self.all_states.append(state)
        self.all_next_states.append(next_state)
        self.all_rewards.append(reward)
        self.all_dones.append(done)
    
    def update(self):
        """使用所有收集的数据更新自身参数"""
        if not self.all_states:
            return 0.0
        states = torch.tensor(self.all_states, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(self.all_next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(self.all_rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        dones = torch.tensor(self.all_dones, dtype=torch.float32).to(self.device).unsqueeze(1)
        
        # 4. 用最新Critic参数计算目标值（与独立Critic一致）
        with torch.no_grad():
            next_values = self(next_states)  # 关键：用当前最新参数算next_value
            targets = rewards + (1 - dones) * self.gamma * next_values
        
        values = self(states)
        critic_loss = F.mse_loss(values, targets)
        
        # 更新价值网络
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()
        
        # 清空数据
        self.clear_memory()
        
        return critic_loss.item()
    
    def clear_memory(self):
        self.all_states = []
        self.all_next_states = []
        self.all_rewards = []
        self.all_dones = []