import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
from algo.ac import Actor, Critic

class HierarchicalActorCritic(nn.Module):
    """
    支持不同 expert 动作维度的 HAC（Router + Experts + Critic）
    动作维度不一致时采用 padding，在 update() 中分 expert 处理。
    """
    def __init__(self, state_dim, action_dims, cfg):
        super().__init__()

        self.device = cfg.device
        self.gamma = cfg.gamma
        self.entropy_coef = cfg.entropy_coef

        self.action_dims = action_dims              # [3, 2] 等
        self.max_action_dim = max(action_dims)
        self.num_experts = len(action_dims)

        # ============== Router ==============
        self.router_backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.router_head = nn.Linear(64, self.num_experts)

        
        self.experts = nn.ModuleList([
            Actor(state_dim, act_dim, cfg) for act_dim in action_dims
        ])

        # 统一 critic
        self.critic = Critic(state_dim, cfg)

        # ============== Optimizers ==============
        self.optimizer_actor = optim.Adam(
            list(self.router_backbone.parameters()) +
            list(self.router_head.parameters()) +
            list(self.experts.parameters()),
            lr=cfg.actor_lr
        )
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        # ============== Memory ==============
        self.states = []
        self.actions = []         # padding 后的动作 [max_action_dim]
        self.expert_indices = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self.to(self.device)
    # ============================================================
    #   Router Forward
    # ============================================================
    def _router_forward(self, state_tensor):
        h = self.router_backbone(state_tensor)
        logits = self.router_head(h)
        dist = Categorical(logits=logits)
        return logits, dist

    @torch.no_grad()
    def select_expert(self, state):
        """router 选择 expert"""
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        _, dist = self._router_forward(s)
        z = dist.sample()
        return int(z.item())

    @torch.no_grad()
    def select_action_forced(self, state, expert_idx, context, t):
        """强制使用 expert_idx 采样动作"""
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        expert = self.experts[expert_idx]

        action_tensor, _, _ = expert.sample_action(s)
        action = action_tensor.cpu().numpy().flatten()

        # 调用环境约束 clip
        power_input, ess_change, PG, action_clipped = expert.clip_action(
            context, expert_idx, action, t
        )
        return power_input, ess_change, PG, action_clipped

    # ============================================================
    #   Memory
    # ============================================================
    def store_transition(self, state, expert_idx, action, reward, next_state, done):

        # ---- 状态 ----
        self.states.append(np.asarray(state, dtype=np.float32))
        self.expert_indices.append(int(expert_idx))
        self.rewards.append(float(reward))
        self.next_states.append(np.asarray(next_state, dtype=np.float32))
        self.dones.append(float(done))

        # ---- 动作 padding ----
        a = np.asarray(action, dtype=np.float32)
        pad = np.zeros(self.max_action_dim, dtype=np.float32)
        pad[: a.shape[-1]] = a
        self.actions.append(pad)

    def _clear_memory(self):
        self.states = []
        self.actions = []
        self.expert_indices = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    # ============================================================
    #   Evaluate Joint Log Probabilities
    # ============================================================
    def _evaluate_expert_logprob(self, states, actions, expert_idx):
        """
        给定 expert_index，只计算该 expert 的 log_prob 和 entropy。
        """
        act_dim = self.action_dims[expert_idx]
        expert_actions = actions[:, :act_dim]
        return self.experts[expert_idx].evaluate_action(states, expert_actions)

    # ============================================================
    #   Update: 分 expert 更新版本（支持不同动作维度）
    # ============================================================
    def update(self, policy_update=True):
        if len(self.states) == 0:
            return None, None

        # ========== 1) 转换数据到 Tensor ==========
        states = torch.tensor(np.stack(self.states), device=self.device, dtype=torch.float32)
        actions = torch.tensor(np.stack(self.actions), device=self.device, dtype=torch.float32)
        expert_indices = torch.tensor(self.expert_indices, device=self.device, dtype=torch.long)
        rewards = torch.tensor(self.rewards, device=self.device, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.stack(self.next_states), device=self.device, dtype=torch.float32)
        dones = torch.tensor(self.dones, device=self.device, dtype=torch.float32).unsqueeze(1)

        self._clear_memory()

        # ========== 2) Critic 全量更新 ==========
        values = self.critic(states)
        with torch.no_grad():
            next_values = self.critic(next_states)
            targets = rewards + (1 - dones) * self.gamma * next_values

        advantages = targets - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ========== 3) Router 全量更新 ==========
        _, router_dist = self._router_forward(states)
        logp_router = router_dist.log_prob(expert_indices)
        entropy_router = router_dist.entropy()

        router_loss = -(logp_router * advantages.detach().squeeze(1)).mean() \
                      - self.entropy_coef * entropy_router.mean()
        print(f"Router loss: {router_loss.item():.4f}")
        # ========== 4) Experts 分 expert 更新 ==========
        expert_loss = 0

        for k in range(self.num_experts):
            mask = (expert_indices == k)
            if mask.sum() == 0:
                continue

            states_k = states[mask]
            advantages_k = advantages[mask].detach()
            actions_k = actions[mask]

            logp_k, ent_k = self._evaluate_expert_logprob(states_k, actions_k, k)

            loss_k = -(logp_k * advantages_k).mean() - self.entropy_coef * ent_k.mean()
            expert_loss += loss_k

        # ========== 5) 合并并更新 ==========
        actor_loss = router_loss + expert_loss
        if policy_update:
            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()
        critic_loss = F.mse_loss(values, targets)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        return actor_loss.item(), critic_loss.item()