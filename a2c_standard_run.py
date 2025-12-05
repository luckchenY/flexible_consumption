import torch
import torch.nn as nn
import numpy as np
import datetime
from A2C.a2c import ActorCritic, RouterActorCritic
from common.utils import plot_rewards, save_results, save_result
import argparse
from envs.standard_env import StandardEnv   
from envs.fix_demand_env import FixDemandEnv
from envs.router_rule_env import RouterRuleEnv


def get_args():
    """ Hyperparameters """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='A2C_router',type=str)
    parser.add_argument('--season',default='summer')
    parser.add_argument('--env_name',default='router_rule',type=str)
    parser.add_argument('--train_eps',default=1000,type=int)
    parser.add_argument('--time_eps',default=671,type=int)
    parser.add_argument('--test_eps',default=20,type=int)
    parser.add_argument('--gamma',default=0.99,type=float)
    parser.add_argument('--critic_lr',default=1e-3,type=float)
    parser.add_argument('--actor_lr',default=1e-3,type=float)
    parser.add_argument('--actor_hidden_dim',default=256,type=int)
    parser.add_argument('--critic_hidden_dim',default=256,type=int)
    parser.add_argument('--entropy_coef',default=1e-3,type=float)
    parser.add_argument('--memory_capacity',default=3000,type=int)
    parser.add_argument('--batch_size',default=256,type=int)
    parser.add_argument('--target_update',default=2,type=int)
    parser.add_argument('--soft_tau',default=1e-2,type=float)
    parser.add_argument('--hidden_dim',default=256,type=int)
    parser.add_argument('--exploration_std',default=0.9,type=float)
    parser.add_argument('--exploration_decay',default=1000,type=int)
    parser.add_argument('--exploration_std_end',default=0.1,type=float)
    parser.add_argument('--device',default='cuda',type=str) 
    parser.add_argument('--result_path',default="outputs/router_rule/A2C_router/results/"+curr_time+'/')
    parser.add_argument('--model_path',default="outputs/router_rule/A2C_router/models/"+curr_time+'/')
    parser.add_argument('--save',default=True,type=bool)
    args = parser.parse_args()
    return args


def train(cfg, env, policyA, policyB, router):
    print("env:", cfg.env_name)
    print("algorithm:", cfg.algo_name)
    rewards, ma_rewards = [], []

    # 统计计数器
    total_router_choices = 0
    total_forced_policy_uses = 0
    policy_0_choices = 0
    policy_1_choices = 0
    price_f_records, price_c_records = [], []
    purchase_fee_totals, energyfee_totals = [], []

    for i_ep in range(cfg.train_eps):
        state = env.reset(cfg.season)
        ep_reward = 0
        ep_purchase_fee_sum = 0.0
        ep_energyfee_sum = 0.0

        for time_step in range(cfg.time_eps):
            force_policy_index = env.env_knowledge(time_step, cfg.season)

            # ============= Router Decision =============
            if force_policy_index:  
                total_forced_policy_uses += 1
                if force_policy_index == 2:
                    policy_idx = 1
                else:
                    policy_idx = 0
                log_prob, V_router = None, None
            else:
                total_router_choices += 1
                policy_idx, log_prob, V_router = router.select_policy(state)
            
            # ============= Execute Sub-policy =============
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
            
            if policy_idx == 0:
                power_input, ess_change, PG, action = policyA.select_action(state, context, policy_idx, time_step)
                policy_0_choices += 1
            else:
                power_input, ess_change, PG, action = policyB.select_action(state, context, policy_idx, time_step)
                policy_1_choices += 1

            next_state, reward, done, price_f, price_c, purchase_fee, energyfee = env.step(
                time_step, policy_idx, power_input, ess_change, PG)
            
            ep_reward += reward
            ep_purchase_fee_sum += float(purchase_fee)
            ep_energyfee_sum += float(energyfee)

            # 存储经验
            if policy_idx == 0:
                policyA.store_transition(state, action, reward, next_state, done)
            else:
                policyB.store_transition(state, action, reward, next_state, done)

            if not force_policy_index:
                router.store_transition(state, policy_idx, reward, next_state, done)

            state = next_state

        # ========= 更新阶段 =========
        price_f_records.append(price_f)
        price_c_records.append(price_c)
        purchase_fee_totals.append(ep_purchase_fee_sum)
        energyfee_totals.append(ep_energyfee_sum)

        # 更新 Router 与 Sub-policies
        router.update(policyA, policyB)
        if len(policyA.states) > 0:
            policyA.update(policy_update=True)
        if len(policyB.states) > 0:
            policyB.update(policy_update=True)

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

        if (i_ep + 1) % 10 == 0:
            print(f"Episode {i_ep+1} | Reward {ep_reward:.2f}")
            print(f"  Router decisions: {total_router_choices}, Forced: {total_forced_policy_uses}")
            total_policy_choices = policy_0_choices + policy_1_choices
            if total_policy_choices > 0:
                print(f"  Policy0 {policy_0_choices/total_policy_choices*100:.2f}% | "
                      f"Policy1 {policy_1_choices/total_policy_choices*100:.2f}%")

    # ========= 统计结果 =========
    print("\n=== Training Completed ===")
    print(f"Total Router choices: {total_router_choices}")
    print(f"Total forced decisions: {total_forced_policy_uses}")
    total_policy_choices = policy_0_choices + policy_1_choices
    print(f"Policy0 ratio: {policy_0_choices/total_policy_choices*100:.2f}%")
    print(f"Policy1 ratio: {policy_1_choices/total_policy_choices*100:.2f}%")

    return {
        'rewards': rewards,
        'ma_rewards': ma_rewards,
        'price_f_records': price_f_records,
        'price_c_records': price_c_records,
        'purchase_fee_totals': purchase_fee_totals,
        'energyfee_totals': energyfee_totals
    }


if __name__ == "__main__":
    cfg = get_args()

    # ===== 环境选择 =====
    if cfg.env_name == 'standard':
        env = StandardEnv()
    elif cfg.env_name == 'fix_demand':
        env = FixDemandEnv()
    elif cfg.env_name == 'router_rule':
        env = RouterRuleEnv()
    else:
        raise ValueError("Unknown environment name.")

    state_dim = 5
    action_dimA, action_dimB = 3, 2
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == 'cuda' else "cpu")
    cfg.device = device
    print(f"Using device: {device}")

    # ===== 初始化策略 =====
    policyA = ActorCritic(state_dim, action_dimA, cfg)
    policyB = ActorCritic(state_dim, action_dimB, cfg)
    router = RouterActorCritic(state_dim, cfg)

    # ===== 路径与训练 =====
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    result_path = f"outputs/{cfg.env_name}/{cfg.algo_name}/{cfg.season}_{curr_time}/results/"
    model_path = f"outputs/{cfg.env_name}/{cfg.algo_name}/{cfg.season}_{curr_time}/models/"
    import os
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    res_dic = train(cfg, env, policyA, policyB, router)

    # ===== 结果保存 =====
    plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], result_path=result_path, tag="train")
    save_results(res_dic['rewards'], res_dic['ma_rewards'], tag='train', path=result_path)
    save_result(res_dic['price_f_records'], tag='price_f', path=result_path)
    save_result(res_dic['price_c_records'], tag='price_c', path=result_path)
    save_result(res_dic['purchase_fee_totals'], tag='purchase_fee', path=result_path)
    save_result(res_dic['energyfee_totals'], tag='energyfee', path=result_path)

    torch.save(policyA.actor.state_dict(), model_path + "/policyA_actor.pth")
    torch.save(policyB.actor.state_dict(), model_path + "/policyB_actor.pth")
    torch.save(policyA.critic.state_dict(), model_path + "/policyA_critic.pth")
    torch.save(policyB.critic.state_dict(), model_path + "/policyB_critic.pth")
    torch.save(router.state_dict(), model_path + "/router_actorcritic.pth")

    print("模型保存完成！")
