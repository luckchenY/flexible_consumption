import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

# 设置随机种子，保证结果可复现
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

class Actor(nn.Module):
    """策略网络，输出连续动作的均值和标准差"""
    def __init__(self, state_dim, action_dim, cfg):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.mean_layer = nn.Linear(cfg.hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(cfg.hidden_dim, action_dim)
        
        # 限制标准差的范围
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, state):
        if self.check_nan("state", state):
            raise ValueError("State has NaN!")
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        if self.check_nan("mean", mean):
            raise ValueError("Mean contains NaN!")
        if self.check_nan("log_std", log_std):
            raise ValueError("log_std contains NaN!")
        return mean, log_std
    
    
    @staticmethod
    def clip_action(context, consumptionflag, action, t):
        action = action.astype(np.float64) if isinstance(action, np.ndarray) else np.array(action, dtype=np.float64)
        ES_capacity = context["ES_capacity"]
        max_ES_capacity = context["max_ES_capacity"]
        ES_rateddischare = context["ES_rateddischare"]
        ES_ratedcharge = context["ES_ratedcharge"]
        PV_GEN = context["PV_GEN"]
        WT_GEN = context["WT_GEN"]
       
        household_ele = context["household_ele"]
        maxEVpower = context["maxEVpower"]
        available_EV = context["available_EV"]
        fundamental = context["fundamental"]
        PG_rated = context["PG_rated"]

        # 基本检查
        assert 0 <= ES_capacity <= max_ES_capacity
        assert sum(fundamental) >= 0
        
        if consumptionflag > 0:  # 消费模式
            action[0] = (action[0] + 1) / 2 * ES_rateddischare
            action[1] = (action[1] + 1) / 2 * PG_rated *2 # 电网功率
            
            ESSoutputcapacity =  min(max(0,ES_capacity - 0.1 * max_ES_capacity), ES_rateddischare)
            if ESSoutputcapacity < action[0]:
                action[0] = ESSoutputcapacity

            PG = action[1]
            power_discharge = action[0]
            power_input = PV_GEN + WT_GEN + PG + power_discharge

            # # ----------- 不满足的需求使用penalty-----------
            # if power_input < demand:
            #     penalty = demand - power_input

            # ----------- 消纳能力限制 -----------
            max_demand = household_ele + maxEVpower * available_EV
            
            if power_input > max_demand:
                # 计算当前超出量
                excess = power_input - max_demand
                PG_new = max(PG - excess, 0)
               
                delta_PG = PG - PG_new
                PG = PG_new
                power_input -= delta_PG
                excess = power_input - max_demand
               
                if excess > 0:
                    power_discharge_new = max(power_discharge - excess, 0)
                    delta_discharge = power_discharge - power_discharge_new
                    power_discharge = power_discharge_new
                    power_input -= delta_discharge
            eps=1e-6
            # =================== 自动容错 + clip ===================
            if not (0 - eps <= action[0] <= ESSoutputcapacity + eps):
                print(f"[WARNING] action[0]={action[0]}, ESSoutputcapacity={ESSoutputcapacity} (auto-clipped)")
                if action[0] < 0 - eps or action[0] > ESSoutputcapacity + eps:
                    raise ValueError(f"action[0] out of safe range: {action[0]} > {ESSoutputcapacity}")
                action[0] = np.clip(action[0], 0.0, ESSoutputcapacity)

            if not (0 - eps <= action[1] <= PG_rated * 2 + eps):
                print(f"[WARNING] action[1]={action[1]}, PG_rated={PG_rated} (auto-clipped)")
                if action[1] < 0 - eps or action[1] > PG_rated * 2 + eps:
                    raise ValueError(f"action[1] out of safe range: {action[1]} > {PG_rated*2}")
                action[1] = np.clip(action[1], 0.0, PG_rated * 2)
                
            action[0] = 2 * (action[0] / ES_rateddischare) - 1
            action[1] = (action[1] / PG_rated) - 1
            return power_input, power_discharge, PG,action

        else:  #储能充电阶段
            if PV_GEN + WT_GEN >= household_ele + maxEVpower * available_EV + min(max(0, 0.9 * max_ES_capacity - ES_capacity), ES_ratedcharge):
                # print("必须要发生弃电", t % 96)
                power_storage =min(max(0, 0.9 * max_ES_capacity - ES_capacity), ES_ratedcharge)
                power_input = household_ele + maxEVpower * available_EV
                PG = 0
                
                action[0] = power_storage/(PV_GEN+WT_GEN)
                action[1] = 0.0
                action[2] = 0.0
                return power_input, power_storage, PG,action
        
            action[0] = (action[0] + 1) / 2
            PVS = PV_GEN * action[0]
            WTS = WT_GEN * action[0]
            action[1] = (action[1] + 1) / 2 * PG_rated # 电网→储能
            action[2] = (action[2] + 1) / 2 * PG_rated   # 电网→用户

            power_storage = PVS + WTS + action[1]
            power_input = PV_GEN - PVS + WT_GEN - WTS + action[2]
            PG = action[1] + action[2]

            storage_capability = min(max(0, 0.9 * max_ES_capacity - ES_capacity), ES_ratedcharge)
            max_demand = household_ele + maxEVpower * available_EV
            
            # =========================================================
            # 动态约束检查
            # =========================================================
            violate_user = power_input > max_demand
            violate_storage = power_storage > storage_capability

            def fix_user_violation():
                nonlocal action, PG, power_input, power_storage, PVS, WTS
                excess = power_input - max_demand
                # print('超出',excess)
                # Step 1: 优先减少电网→用户功率
                reducible = min(excess, action[2])
                action[2] -= reducible
                PG -= reducible
                power_input -= reducible
                excess -= reducible

                # Step 2: 若仍然超出，则增加储能比例
                if excess > 0:
                    total_new_energy = PV_GEN + WT_GEN
                    if total_new_energy > 0:
                        ratio_increase = excess / total_new_energy
                        action[0] = min(1.0, action[0] + ratio_increase)
                        # 重新计算功率
                        PVS = PV_GEN * action[0]
                        WTS = WT_GEN * action[0]
                        power_input = PV_GEN - PVS + WT_GEN - WTS + action[2]
                        power_storage = PVS + WTS + action[1]
                    else:
                        action[0] = 0.0

            def fix_storage_violation():
                nonlocal action, PG, power_input, power_storage, PVS, WTS
                excess = power_storage - storage_capability

                # Step 1: 优先减少电网→储能功率
                reducible = min(excess, action[1])
                action[1] -= reducible
                PG -= reducible
                power_storage -= reducible
                excess -= reducible

                # Step 2: 若仍然超出，则减少新能源入储能比例
                if excess > 0:
                    total_new_energy = PV_GEN + WT_GEN
                    if total_new_energy > 0:
                        ratio_reduce = excess / total_new_energy
                        action[0] = max(0.0, action[0] - ratio_reduce)
                        # 重新计算功率
                        PVS = PV_GEN * action[0]
                        WTS = WT_GEN * action[0]
                        power_storage = PVS + WTS + action[1]
                        power_input = PV_GEN - PVS + WT_GEN - WTS + action[2]
                    else:
                        action[0] = 0.0

            # =========================================================
            # 动态调度逻辑
            # =========================================================
            if violate_user:
                fix_user_violation()
                violate_storage = power_storage > storage_capability
                if violate_storage:
                    fix_storage_violation()  # 修正后可能影响储能约束

            elif violate_storage and not violate_user:
                # 仅违反储能约束
                fix_storage_violation()
                if violate_user:
                    fix_user_violation()  # 修正后可能影响用户约束
            
            # =================== 自动容错 + clip ===================
            eps=1e-6
            if not (0 - eps <= action[0] <= 1 + eps):
                print(f"[WARNING] action[0]={action[0]} 超出 [0,1] 范围 (auto-clipped)")
                if action[0] < 0 - eps or action[0] > 1 + eps:
                    raise ValueError(f"action[0] 超出安全范围: {action[0]}")
                action[0] = np.clip(action[0], 0.0, 1.0)

            for i in [1, 2]:
                if not (0 - eps <= action[i] <= PG_rated + eps):
                    print(f"[WARNING] action[{i}]={action[i]} 超出 [0,PG_rated] 范围 (auto-clipped)")
                    if action[i] < 0 - eps or action[i] > PG_rated + eps:
                        raise ValueError(f"action[{i}] 超出安全范围: {action[i]}")
                    action[i] = np.clip(action[i], 0.0, PG_rated)

            action[0]=  2*action[0]-1    
            action[1] = 2*(action[1] / PG_rated) - 1
            action[2]=  2*action[2]/PG_rated -1    
            # if power_input < demand:
            #     penalty = demand - power_input
            
            return power_input, power_storage, PG,action


    def check_nan(self,name, tensor):
        if torch.isnan(tensor).any():
            print(f"[NaN DETECTED] {name} contains NaN values!")
            print(f"  tensor: {tensor.detach().cpu().numpy()}")
            return True
        return False

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
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        # 计算熵
        entropy = normal.entropy().sum(-1, keepdim=True)

        return action, log_prob, entropy
    
    def evaluate_action(self, state, action):
        # print(action)
        """根据给定动作计算log_prob和熵"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)

        # 反算 tanh 之前的原始动作
        raw_action = 0.5 * (torch.log1p(action + 1e-6) - torch.log1p(-action + 1e-6))
        log_prob =(dist.log_prob(raw_action)- torch.log(1 - action.pow(2) + 1e-6)).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy

class Critic(nn.Module):
    """价值网络 V(s)"""
    def __init__(self, state_dim, cfg):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.fc3 = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value
