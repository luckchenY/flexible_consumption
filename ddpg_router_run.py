import torch
import torch.nn as nn
import numpy as np
import datetime
from DDPG.ddpg import DDPG
from common.utils import plot_rewards, save_results
import argparse
from envs.standard_env import StandardEnv
from envs.fix_demand_env import FixDemandEnv
from collections import deque
from Router.router import RouterNet

def get_args():
    """ Hyperparameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='DDPG_router',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='standard',type=str,help="name of environment")
    parser.add_argument('--season',default='summer')
    parser.add_argument('--train_eps',default=1000,type=int,help="episodes of training")
    parser.add_argument('--time_eps',default=672,type=int,help="interaction steps of each episode")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
    parser.add_argument('--critic_lr',default=1e-4,type=float,help="learning rate of critic")
    parser.add_argument('--actor_lr',default=1e-4,type=float,help="learning rate of actor")
    parser.add_argument('--memory_capacity',default=3000,type=int,help="memory capacity")
    parser.add_argument('--batch_size',default=256,type=int)
    parser.add_argument('--target_update',default=2,type=int)
    parser.add_argument('--soft_tau',default=1e-2,type=float)
    parser.add_argument('--hidden_dim',default=256,type=int)
    parser.add_argument('--exploration_std',default=0.9,type=float,help="exploration noise std")
    parser.add_argument('--exploration_std_initial',default=0.9,type=float)
    parser.add_argument('--exploration_decay',default=1000,type=int,help="exploration decay steps")
    parser.add_argument('--exploration_std_end',default=0.1,type=float,help="exploration noise std end")
    parser.add_argument('--device',default='cuda',type=str,help="cpu or cuda") 
    parser.add_argument('--result_path',default="outputs/" + parser.parse_args().env_name +'/'+parser.parse_args().algo_name+'/results/' + curr_time +'/')
    parser.add_argument('--model_path',default= "outputs/" + parser.parse_args().env_name +'/'+parser.parse_args().algo_name+'/models/' + curr_time +'/') # path to save models
    parser.add_argument('--save',default=True,type=bool,help="if save figure or not")   
    args = parser.parse_args()
    
    return args

def update_exploration_std(cfg, epoch):
    """
    根据当前epoch更新探索噪声标准差
    衰减逻辑：前exploration_decay个epoch内从初始值线性衰减到最终值，之后保持不变
    """
    # 修正条件：当epoch < 衰减总步数时，执行衰减；否则保持最终值
    if epoch < cfg.exploration_decay:
        # 计算衰减比例（0~1）：epoch越大，比例越接近1，探索率越接近最终值
        decay_ratio = (epoch + 1) / cfg.exploration_decay  # +1避免epoch=0时比例为0
        # 线性插值：初始值 -> 最终值
        cfg.exploration_std = cfg.exploration_std_end + (cfg.exploration_std_initial - cfg.exploration_std_end) * (1 - decay_ratio)
    else:
        # 超过衰减步数后，固定为最终值
        cfg.exploration_std = cfg.exploration_std_end

def train(cfg, env, policyA, policyB, router):
    print("env",cfg.env_name)
    print("algorithm",cfg.algo_name)
    rewards, ma_rewards = [], []
    for i_ep in range(cfg.train_eps):
        state = env.reset(cfg.season)
        ep_reward = 0
    
        for time_step in range(cfg.time_eps):
            force_policy_index = env.env_knowledge(time_step,cfg.season)  # 获取当前时间步的规则策略指令
            
            if force_policy_index:  # 强制使用特定策略
                policy_idx = 0 if force_policy_index==1 else 1 #1是储能2是放电
            else: 
                # 路由决策
                # 在每步训练时，记录 router 的 log_prob
                logits = router(torch.FloatTensor(state).unsqueeze(0))
                # print("state", state)
                # print("logits:", logits)
                prob = torch.softmax(logits, dim=1)
                m = torch.distributions.Categorical(prob)
                policy_idx = m.sample().item()
                log_prob = m.log_prob(torch.tensor(policy_idx))
            
            if policy_idx == 0: #进行储能
                action = policyA.choose_action(state)
                action = action + np.random.normal(0, cfg.exploration_std,size=action.shape)
                action = np.clip(action, -1.0, 1.0)  # 确保action在[-1, 1]范围内
            else:
                action = policyB.choose_action(state)
                action = action + np.random.normal(0, cfg.exploration_std, size=action.shape)
                action = np.clip(action, -1.0, 1.0)  # 确保action在[-1, 1]范围内
            next_state, reward, done, price_f, price_c, purchase_fee, energyfee = env.step(time_step,policy_idx,action)
            ep_reward += reward
            # policy训练
            if policy_idx == 0:
                policyA.memory.push(state, action, reward, next_state, done)
                policyA.update()
            else:
                policyB.memory.push(state, action, reward, next_state, done)
                policyB.update()
            
            if not force_policy_index:
                # 存储Router单步数据（用于后续计算累积奖励）
                router.store_step_data(state, policy_idx, reward)

            state = next_state
        
        # Episode结束，计算累积奖励并存入经验池，然后进行更新
        router.finish_episode()
        router.update()
        
        update_exploration_std(cfg, i_ep)
        rewards.append(ep_reward)
        
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)

        # if (i_ep+1)%10 == 0:
        print(f'Episode:{i_ep+1}, Reward:{ep_reward:.2f}')
    return {'rewards': rewards, 'ma_rewards': ma_rewards}

def test(cfg, env, policyA, policyB, router):
    rewards, ma_rewards = [], []
    for i_ep in range(cfg.test_eps):
        state = env.reset()
        ep_reward = 0
        for time_step in range(cfg.time_eps):
            logits = router(torch.FloatTensor(state).unsqueeze(0))
            prob = torch.softmax(logits, dim=1)
            policy_idx = torch.argmax(prob, dim=1).item()  # 直接选最大概率，不采样
            if policy_idx == 0:
                action = policyA.choose_action(state)
            else:
                action = policyB.choose_action(state)
            next_state, reward, done, _ , _ = env.step(time_step, policy_idx, action)
            ep_reward += reward
            state = next_state
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep+1) % 10 == 0:
            print(f'Test Episode:{i_ep+1}, Reward:{ep_reward:.2f}')
    return {'rewards': rewards, 'ma_rewards': ma_rewards}

if __name__ == "__main__":
    # 假设cfg已定义
    cfg = get_args()
    # training
    if cfg.env_name == 'standard':
        env = StandardEnv()  # 使用StandardEnv替代gym环境
    elif cfg.env_name == 'fix_demand':
        env = FixDemandEnv()  # 使用FixDemandEnv替代gym环境
    
    action_dimA = 3
    action_dimB= 2
    state_dim = 5  # 根据实际情况调整状态维度
    policyA = DDPG(state_dim, action_dimA, cfg)
    policyB = DDPG(state_dim, action_dimB, cfg)
    router = RouterNet(state_dim, cfg.batch_size, cfg.gamma)

    res_dic = train(cfg, env, policyA, policyB, router)
    plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], cfg, tag="train")
    save_results(res_dic['rewards'], res_dic['ma_rewards'], tag='train', path=cfg.result_path)  # save results
    # save_models()