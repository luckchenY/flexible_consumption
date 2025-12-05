import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class RouterNet(nn.Module):
    """
    RouterNet (Dual-Phase Training)
    ======================================
    Phase 1 (Warm-up): imitation + critic only
    Phase 2 (Co-training): actor-critic optimization
    """
    def __init__(self, state_dim, batch_size=128, gamma=0.99, lr=5e-5,
                 entropy_coef=0.01, imitation_coef=0.05, switch_ratio=0.3):
        super().__init__()

        self.gamma = gamma
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.imitation_coef = imitation_coef
        self.switch_ratio = switch_ratio  # phase switch percentage

        # Actor network
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.actor_head = nn.Linear(64, 2)

        # Critic network
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

        # init
        init_w = 3e-3
        self.actor_head.weight.data.uniform_(-init_w, init_w)
        self.actor_head.bias.data.uniform_(-init_w, init_w)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

        # Replay buffer: (state, policy, rule_policy, Gt)
        self.replay_buffer = deque(maxlen=2000)
        self.current_episode_data = []

        # internal counter for phase switching
        self.total_episodes = 0
        self.phase = 'imitation'

    def forward(self, x):
        logits = self.actor_head(self.actor_net(x))
        return torch.clamp(logits, -5, 5)

    def value(self, x):
        return self.critic_net(x)

    def store_step_data(self, state, policy_idx, reward, rule_policy_idx=None):
        self.current_episode_data.append((state.copy(), policy_idx, reward, rule_policy_idx))

    def finish_episode(self):
        if not self.current_episode_data:
            return
        for i, (state, policy_idx, reward, rule_policy_idx) in enumerate(self.current_episode_data):
            Gt = 0
            for j in range(i, len(self.current_episode_data)):
                _, _, r_future, _ = self.current_episode_data[j]
                Gt += (self.gamma ** (j - i)) * r_future
            self.replay_buffer.append((state, policy_idx, rule_policy_idx, Gt))
        self.current_episode_data = []
        self.total_episodes += 1

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return False

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, policy_indices, rule_indices, returns = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        policy_indices = torch.LongTensor(policy_indices)
        returns = torch.FloatTensor(returns)

        values = self.value(states).squeeze(1)
        advantages = returns - values.detach()

        logits = self.forward(states)
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs.gather(1, policy_indices.unsqueeze(1)).squeeze(1) + 1e-8)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)

        # imitation term
        imitation_loss = 0.0
        valid_mask = [i for i, r in enumerate(rule_indices) if r is not None]
        if len(valid_mask) > 0:
            rule_targets = torch.LongTensor([rule_indices[i] for i in valid_mask])
            rule_logits = logits[valid_mask]
            imitation_loss = F.cross_entropy(rule_logits, rule_targets)

        # determine phase
        ratio = self.total_episodes / max(1, self.total_episodes + 1)
        if ratio < self.switch_ratio:
            self.phase = 'imitation'
        else:
            self.phase = 'co_train'

        # select loss based on phase
        if self.phase == 'imitation':
            total_loss = 0.5 * value_loss + self.imitation_coef * imitation_loss
        else:
            total_loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()

        return {
            'phase': self.phase,
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'imitation_loss': imitation_loss if isinstance(imitation_loss, float) else imitation_loss.item(),
            'entropy': entropy.item()
        }
