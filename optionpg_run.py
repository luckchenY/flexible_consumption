import torch
import torch.nn as nn
import numpy as np
import datetime
from optionpolicy.policy_gradient import PolicyGradient
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
    parser.add_argument('--env_name',default='fix_demand',type=str,help="name of environment")
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
    parser.add_argument('--exploration_decay',default=1000,type=int,help="exploration decay steps")
    parser.add_argument('--exploration_std_end',default=0.1,type=float,help="exploration noise std end")
    parser.add_argument('--device',default='cuda',type=str,help="cpu or cuda") 
    parser.add_argument('--result_path',default="outputs/" + parser.parse_args().env_name +'/'+parser.parse_args().algo_name+'/results/' + curr_time +'/')
    parser.add_argument('--model_path',default= "outputs/" + parser.parse_args().env_name +'/'+parser.parse_args().algo_name+'/models/' + curr_time +'/') # path to save models
    parser.add_argument('--save',default=True,type=bool,help="if save figure or not")   
    args = parser.parse_args()
    
    return args

def update_exploration_std(cfg, epoch):
    # 根据衰减公式更新探索噪声标准差
    if epoch-cfg.exploration_decay > 0:
        cfg.exploration_std = ( cfg.exploration_std_end+ (cfg.exploration_std - cfg.exploration_std_end)* (1 - (epoch + 1) / cfg.exploration_decay))
    else:
        cfg.exploration_std = cfg.exploration_std_end

def train(cfg, env, policyA, policyB, router):
    print("env",cfg.env_name)
    print("algorithm",cfg.algo_name)
    rewards, ma_rewards = [], []
    for i_ep in range(cfg.train_eps):
        state = env.reset()
        ep_reward = 0
    
        for time_step in range(cfg.time_eps):
            force_policy_index = env.env_knowledge(time_step)
            
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
                
            else:
                action = policyB.choose_action(state)
                
            next_state, reward, done = env.step(time_step,policy_idx,action)
            ep_reward += reward

            #联合训练
            


            state = next_state
        
        update_exploration_std(cfg, i_ep)
        rewards.append(ep_reward)
        
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)

        # if (i_ep+1)%10 == 0:
        print(f'Episode:{i_ep+1}, Reward:{ep_reward:.2f}')
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