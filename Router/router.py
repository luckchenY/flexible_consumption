import torch.nn as nn
import numpy as np
from collections import deque
import torch

class RouterNet(nn.Module):
    def __init__(self, state_dim, batch_size=128, gamma=0.99):
        super().__init__()
        # 适合单样本推理的分类网络结构
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 分类头
        self.classifier = nn.Linear(32, 2)  # 输出2类，0: policyA, 1: policyB

        # 只初始化最后一层，类似DDPG Actor的做法
        init_w = 3e-3
        self.classifier.weight.data.uniform_(-init_w, init_w)
        self.classifier.bias.data.uniform_(-init_w, init_w)
                
        # 使用更低的学习率，防止过快收敛
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-6, weight_decay=1e-6)
        
        # REINFORCE参数
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Router专用经验池：存储（状态，选择的policy，长期累积奖励）
        self.router_replay_buffer = deque(maxlen=1000)  # 容量更大，保留更久数据
        
        # 临时存储当前episode的数据，用于计算累积奖励
        self.current_episode_data = []
        
    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        # 分类
        logits = self.classifier(features)
        logits = torch.clamp(logits, -5, 5)

        return logits

    def store_step_data(self, state, policy_idx, reward):
        """临时存储单步数据，用于episode结束时计算累积奖励"""
        self.current_episode_data.append((state.copy(), policy_idx, reward))
    
    def finish_episode(self):
        """episode结束时，计算累积奖励并存入经验池"""
        if not self.current_episode_data:
            return
            
        # 为当前episode的每一步计算累积折扣奖励
        for i, (state, policy_idx, reward) in enumerate(self.current_episode_data):
            # 计算从当前步开始的累积折扣奖励
            discounted_reward = 0
            for j in range(i, len(self.current_episode_data)):
                _, _, future_reward = self.current_episode_data[j]
                discounted_reward += future_reward * (self.gamma ** (j - i))
            
            # 存入经验回放缓冲区
            self.router_replay_buffer.append((state, policy_idx, discounted_reward))
        
        # 清空当前episode数据
        self.current_episode_data = []
    
    def update(self):
        """基于REINFORCE算法和经验回放的混合更新机制"""
        # 当经验不足一个批量时，不更新
        if len(self.router_replay_buffer) < self.batch_size:
            return False
        
        # 随机采样一个批量的经验（包含累积奖励）
        import random
        batch_experiences = random.sample(list(self.router_replay_buffer), self.batch_size)
        
        # 解压批量数据
        states, policy_indices, discounted_rewards = zip(*batch_experiences)
        
        # 转换为tensor
        states = torch.FloatTensor(np.array(states))
        policy_indices = torch.LongTensor(policy_indices)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        
        # 标准化累积奖励（提高训练稳定性）
        if discounted_rewards.std() > 1e-9:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)
        
        # 重新计算当前网络的logits和log_probs（确保梯度流正确）
        logits = self.forward(states)  # [batch_size, 2]
        probs = torch.softmax(logits, dim=1)  # [batch_size, 2]
        
        # 计算每个样本选择的policy的log_prob
        log_probs = torch.log(probs.gather(1, policy_indices.unsqueeze(1)).squeeze(1) + 1e-8)
        
        # REINFORCE核心：策略梯度损失
        policy_loss = -(log_probs * discounted_rewards).mean()
        
        # 添加熵正则化，鼓励保持探索
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        entropy_weight = 0.01  # 降低熵权重，因为已有累积奖励提供稳定性
        policy_loss = policy_loss - entropy_weight * entropy
        
        # 执行反向传播
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return True
