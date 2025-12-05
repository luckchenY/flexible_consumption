import torch
import torch.nn as nn
import numpy as np
import datetime
from common.utils import plot_rewards, plot_losses,save_results,save_result
import argparse
from envs.standard_env import StandardEnv   
from envs.fix_demand_env import FixDemandEnv
from envs.router_rule_env import RouterRuleEnv
from envs.fix_demand_rule_env import FixDemandRuleEnv   
from algo.a2c import HierarchicalActorCritic 

def get_args():
    """ Hyperparameters
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='A2C_router',type=str,help="name of algorithm")
    parser.add_argument('--season',default='summer')
    parser.add_argument('--env_name',default='router_rule',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=1000,type=int,help="episodes of training")
    parser.add_argument('--time_eps',default=672,type=int,help="interaction steps of each episode")
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing")
    parser.add_argument('--gamma',default=1,type=float,help="discounted factor")
    parser.add_argument('--critic_lr',default=1e-5,type=float,help="learning rate of critic")
    parser.add_argument('--actor_lr',default=1e-6,type=float,help="learning rate of actor")
    parser.add_argument('--actor_hidden_dim',default=256,type=int,help="hidden dimension of actor")
    parser.add_argument('--critic_hidden_dim',default=256,type=int,help="hidden dimension of critic")
    parser.add_argument('--entropy_coef',default=1,type=float,help="entropy regularization coefficient")
    parser.add_argument('--memory_capacity',default=3000,type=int,help="memory capacity")
    parser.add_argument('--freeze_actor_epochs',default=100,type=int,help="number of epochs to freeze actor during training")
    parser.add_argument('--batch_size',default=256,type=int)
    parser.add_argument('--target_update',default=2,type=int)
    parser.add_argument('--soft_tau',default=1e-2,type=float)
    parser.add_argument('--hidden_dim',default=256,type=int)
    parser.add_argument('--exploration_std',default=0.9,type=float,help="exploration noise std")
    parser.add_argument('--exploration_decay',default=1000,type=int,help="exploration decay steps")
    parser.add_argument('--exploration_std_end',default=0.1,type=float,help="exploration noise std end")
    parser.add_argument('--device',default='cuda:2',type=str,help="cpu or cuda") 
    parser.add_argument('--result_path',default="outputs/" + parser.parse_args().env_name + '/'+parser.parse_args().algo_name +'/results/' + curr_time +'/')
    parser.add_argument('--model_path',default= "outputs/" + parser.parse_args().env_name + '/'+parser.parse_args().algo_name + '/models/' + curr_time +'/') # path to save models
    parser.add_argument('--save',default=True,type=bool,help="if save figure or not")   
    args = parser.parse_args()
    
    return args

def train(cfg, env, hac: HierarchicalActorCritic):
    print("env", cfg.env_name)
    print("algorithm", cfg.algo_name)

    rewards, ma_rewards = [], []
    actor_losses, critic_losses = [], []

    policy_choice_count = [0 for _ in range(hac.num_experts)]
    forced_policy_count = 0
    router_choice_count = 0

    # price / fee 记录
    price_f_records, price_c_records = [], []
    purchase_fee_totals, energyfee_totals, ep_penaltys = [], [], []

    for i_ep in range(cfg.train_eps):
        state = env.reset(cfg.season)

        ep_reward = 0
        ep_purchase_fee = 0.0
        ep_energyfee = 0.0
        ep_penalty_sum = 0.0

        for time_step in range(cfg.time_eps):

            # ===== 1. 环境强制策略 =====
            force_policy_index = env.env_knowledge(time_step, cfg.season)
            if force_policy_index:
                # 环境约定的策略编号映射到 expert_idx
                forced_policy_count += 1
                if force_policy_index == 2:
                    expert_idx = 1
                else:
                    expert_idx = 0
            else:
                # ===== 2. Router 决策 =====
                router_choice_count += 1
                expert_idx = hac.select_expert(state) 
            

            # ===== 3. 根据 expert_idx 采样动作 =====
            context = {
                "ES_capacity": env.ES_capacity,
                "max_ES_capacity": env.max_ES_capacity,
                "ES_rateddischare": env.ES_rateddischare,
                "ES_ratedcharge": env.ES_ratedcharge,
                "PV_GEN": env.PV_GEN[time_step],
                "WT_GEN": env.WT_GEN[time_step],
                "PV_F": env.PV_F[time_step],
                "WT_F": env.WT_F[time_step],
                "household_ele": env.household_ele[time_step],
                "maxEVpower": env.maxEVpower,
                "available_EV": env.available_EV,
                "fundamental": env.fundamental,
                "PG_rated": env.PG_rated,
            }

            # 用 HAC 的接口采样（会调用对应 expert）
            power_input, ess_change, PG, action = hac.select_action_forced(
                state, expert_idx, context, time_step
            )

            policy_choice_count[expert_idx] += 1

            # ===== 4. 环境交互 =====
            next_state, reward, done, price_f, price_c, purchase_fee, energyfee, penalty = \
                env.step(time_step, expert_idx, power_input, ess_change, PG)

            ep_reward += reward
            ep_purchase_fee += float(purchase_fee)
            ep_energyfee += float(energyfee)
            ep_penalty_sum += float(penalty)

            # ===== 5. 记忆 =====
            hac.store_transition(state, expert_idx, action, reward, next_state, done)

            state = next_state

        # ===== Episode 完成 =====
        price_f_records.append(price_f)
        price_c_records.append(price_c)
        purchase_fee_totals.append(ep_purchase_fee)
        energyfee_totals.append(ep_energyfee)
        ep_penaltys.append(ep_penalty_sum)

        # ===== 6. 更新 (一个 HAC 更新所有：router + experts + critic) =====
        actor_loss, critic_loss = hac.update(i_ep > cfg.freeze_actor_epochs)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        rewards.append(ep_reward)
        ma_rewards.append(ep_reward if not ma_rewards else 0.9 * ma_rewards[-1] + 0.1 * ep_reward)

        if (i_ep + 1) % 10 == 0:
            print(f"Episode {i_ep+1}, Reward {ep_reward:.2f}")
            print(f"Router choice ratio: {router_choice_count / (router_choice_count + forced_policy_count + 1e-6):.2f}")
            print(f"Policy choices: {policy_choice_count}")

    # ===== 最终统计 =====
    print("\n=== Final Statistics ===")
    print(f"Router total choices: {router_choice_count}")
    print(f"Forced policy total: {forced_policy_count}")
    print(f"Policy choice counts: {policy_choice_count}")

    return {
        'rewards': rewards,
        'ma_rewards': ma_rewards,
        'price_f_records': price_f_records,
        'price_c_records': price_c_records,
        'purchase_fee_totals': purchase_fee_totals,
        'energyfee_totals': energyfee_totals,
        'actorloss': actor_losses,
        'criticloss': critic_losses,
        'ep_penaltys': ep_penaltys,
    }

if __name__ == "__main__":
    # 假设cfg已定义
    cfg = get_args()
    # training
    if cfg.env_name == 'standard':
        env = StandardEnv()  # 使用StandardEnv
    elif cfg.env_name == 'fix_demand':
        env = FixDemandEnv()  # 使用FixDemandEnv
    elif cfg.env_name == 'router_rule':
        env = RouterRuleEnv()  # 使用RouterRuleEnv
    elif cfg.env_name == 'fix_demand_rule':
        env = FixDemandRuleEnv()  # 使用FixDemandRuleEnv
    
    action_dimA = 3
    action_dimB = 2
    state_dim = 6  # 根据实际情况调整状态维度
    hac = HierarchicalActorCritic(state_dim, action_dims=[action_dimA,action_dimB], cfg=cfg)
    res_dic = train(cfg, env, hac)
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    result_path="outputs/" + cfg.env_name + '/'+cfg.algo_name +'/'+ cfg.season+ curr_time +'/results/' 
    model_path= "outputs/" + cfg.env_name + '/'+cfg.algo_name +'/'+ cfg.season+ curr_time +'/models/hac_model.pth'   

    import os
  
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], result_path=result_path, tag="train")
    plot_losses(res_dic['actorloss'], algo="Actor", save=True, path=result_path + "actorA")
    plot_losses(res_dic['criticloss'], algo="Critic", save=True, path=result_path + "criticA")
    plot_losses(res_dic['ep_penaltys'], algo="Penalty", save=True, path=result_path + "penalty_")
    save_results(res_dic['rewards'], res_dic['ma_rewards'], tag='train', path=result_path)  # save results
    save_result(res_dic['price_f_records'], tag='price_f', path=result_path)
    save_result(res_dic['price_c_records'], tag='price_c', path=result_path)
    save_result(res_dic['purchase_fee_totals'], tag='purchase_fee', path=result_path)
    save_result(res_dic['energyfee_totals'], tag='energyfee', path=result_path) 
    save_result(res_dic['ep_penaltys'], tag='penalty', path=result_path)

    try:
        torch.save(hac.state_dict(), model_path)
        print(f"🎉 模型成功保存！文件大小: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"❌ 保存过程中出现错误: {e}")

    # # 测试
    # test_res = test(cfg, env, policyA, policyB, router)
    # plot_rewards(test_res['rewards'], test_res['ma_rewards'], cfg, tag="test")
    # save_results(test_res['rewards'], test_res['ma_rewards'], tag='test', path=cfg.result_path)
    