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
        self.old_logp_router = []
        self.old_logp_expert = []

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
        log_val, _ = self.experts[expert_idx].evaluate_action(state=torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0),
                                                              action=torch.tensor(a, dtype=torch.float32, device=self.device).unsqueeze(0)) 
        self.old_logp_expert.append(log_val.item())

        _, router_dist = self._router_forward(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
        logp_router = router_dist.log_prob(torch.tensor(expert_idx, dtype=torch.long, device=self.device).unsqueeze(0))
        self.old_logp_router.append(logp_router.item())
       
    def _clear_memory(self):
        self.states = []
        self.actions = []
        self.expert_indices = []
        self.rewards = []
        self.next_states = []
        self.old_logp_router = []
        self.old_logp_expert = []
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

    def update(self, policy_update=True, ppo_epochs=3, batch_size=1024, clip_eps=0.2):
        if len(self.states) == 0:
            return None, None

        # ===== 1) 整理数据到 Tensor =====
        states = torch.tensor(np.stack(self.states), device=self.device, dtype=torch.float32)
        actions = torch.tensor(np.stack(self.actions), device=self.device, dtype=torch.float32)
        expert_indices = torch.tensor(self.expert_indices, device=self.device, dtype=torch.long)
        rewards = torch.tensor(self.rewards, device=self.device, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.stack(self.next_states), device=self.device, dtype=torch.float32)
        dones = torch.tensor(self.dones, device=self.device, dtype=torch.float32).unsqueeze(1)

        old_logp_router = torch.tensor(self.old_logp_router, device=self.device, dtype=torch.float32)
        old_logp_expert = torch.tensor(self.old_logp_expert, device=self.device, dtype=torch.float32)

        self._clear_memory()

        # ===== 2) Compute advantage =====
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            targets = rewards + (1 - dones) * self.gamma * next_values
            advantages = targets - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            advantages = advantages.squeeze(1) #使用critic可能会多给出一个batch的维度就是1， 把这个去掉之后变为一维。
            advantages = torch.clamp(advantages, -10, 10)

        # ===== 3) PPO epoch & minibatch =====
        N = states.size(0)
        idxs = np.arange(N)

        for _ in range(ppo_epochs):

            np.random.shuffle(idxs)
            for start in range(0, N, batch_size):
                end = start + batch_size
                mb_idx = idxs[start:end]

                mb_states      = states[mb_idx]
                mb_actions     = actions[mb_idx]
                mb_adv         = advantages[mb_idx]
                mb_targets     = targets[mb_idx]
                mb_expert_idx  = expert_indices[mb_idx]

                mb_old_logp_router = old_logp_router[mb_idx]
                mb_old_logp_expert = old_logp_expert[mb_idx]

                # ===== Router PPO =====
                _, router_dist = self._router_forward(mb_states)
                new_logp_router = router_dist.log_prob(mb_expert_idx)
                ratio_router = torch.clamp(new_logp_router - mb_old_logp_router, -3, 3)
                ratio_router = torch.exp(ratio_router)

                surr1 = ratio_router * mb_adv
                surr2 = torch.clamp(ratio_router, 1-clip_eps, 1+clip_eps) * mb_adv
                router_loss = -torch.min(surr1, surr2).mean()

                entropy_router = router_dist.entropy().mean()
                router_loss -= self.entropy_coef * entropy_router

                # ===== Experts PPO（分 expert 更新）=====
                expert_loss = 0
                for k in range(self.num_experts):
                    mask = (mb_expert_idx == k)
                    if mask.sum() == 0:
                        continue

                    st_k = mb_states[mask]
                    act_k = mb_actions[mask]
                    adv_k = mb_adv[mask]
                    oldp_k = mb_old_logp_expert[mask]

                    new_logp_k, ent_k = self._evaluate_expert_logprob(st_k, act_k, k)
                    
                    new_logp_k = new_logp_k.view(-1)      # [Nk]
                    ent_k      = ent_k.view(-1)           # [Nk]
                    adv_k      = adv_k.view(-1)           # [Nk]
                    oldp_k     = oldp_k.view(-1)          # [Nk]

                    assert torch.isfinite(new_logp_k).all(), "expert new_logp NaN"
                    assert torch.isfinite(oldp_k).all(), "expert old_logp NaN"
                    # assert torch.isfinite(diff_k).all(), "expert logp diff NaN"

                    ratio_k = torch.clamp(new_logp_k - oldp_k, -3, 3)
                    ratio_k = torch.exp(ratio_k)
                    assert torch.isfinite(ratio_k).all(), "expert ratio_k NaN"

                    s1 = ratio_k * adv_k
                    s2 = torch.clamp(ratio_k, 1-clip_eps, 1+clip_eps) * adv_k
                    loss_k = -torch.min(s1, s2).mean() - self.entropy_coef * ent_k.mean()

                    expert_loss += loss_k

                actor_loss = router_loss + expert_loss

                # ===== Critic loss with clipping =====
                new_values = self.critic(mb_states)
                value_clipped = values[mb_idx] + (new_values - values[mb_idx]).clamp(-clip_eps, clip_eps)
                critic_loss1 = (new_values - mb_targets).pow(2)
                critic_loss2 = (value_clipped - mb_targets).pow(2)
                critic_loss = 0.5 * torch.max(critic_loss1, critic_loss2).mean()
                
                if policy_update:
                    self.optimizer_actor.zero_grad()
                    actor_loss.backward()
                    self.optimizer_actor.step()
                    
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                self.optimizer_critic.step()

        return actor_loss.item(), critic_loss.item()
    
