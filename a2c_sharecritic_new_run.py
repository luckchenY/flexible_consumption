import torch
import torch.nn as nn
import numpy as np
import datetime
from A2C.a2c_sharecritic import SharedCritic,Actor
from common.utils import plot_rewards, plot_losses,save_results,save_result
import argparse
from envs.standard_env import StandardEnv   
from envs.fix_demand_env import FixDemandEnv
from envs.router_rule_env import RouterRuleEnv
from Router.router import RouterNet


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
    parser.add_argument('--gamma',default=0.99,type=float,help="discounted factor")
    parser.add_argument('--critic_lr',default=1e-6,type=float,help="learning rate of critic")
    parser.add_argument('--actor_lr',default=1e-5,type=float,help="learning rate of actor")
    parser.add_argument('--actor_hidden_dim',default=256,type=int,help="hidden dimension of actor")
    parser.add_argument('--critic_hidden_dim',default=256,type=int,help="hidden dimension of critic")
    parser.add_argument('--entropy_coef',default=1e-3,type=float,help="entropy regularization coefficient")
    parser.add_argument('--memory_capacity',default=3000,type=int,help="memory capacity")
    parser.add_argument('--freeze_actor_epochs',default=100,type=int,help="number of epochs to freeze actor during training")
    parser.add_argument('--batch_size',default=256,type=int)
    parser.add_argument('--target_update',default=2,type=int)
    parser.add_argument('--soft_tau',default=1e-2,type=float)
    parser.add_argument('--hidden_dim',default=256,type=int)
    parser.add_argument('--exploration_std',default=0.9,type=float,help="exploration noise std")
    parser.add_argument('--exploration_decay',default=1000,type=int,help="exploration decay steps")
    parser.add_argument('--exploration_std_end',default=0.1,type=float,help="exploration noise std end")
    parser.add_argument('--device',default='cuda',type=str,help="cpu or cuda") 
    parser.add_argument('--result_path',default="outputs/" + parser.parse_args().env_name + '/'+parser.parse_args().algo_name +'/results/' + curr_time +'/')
    parser.add_argument('--model_path',default= "outputs/" + parser.parse_args().env_name + '/'+parser.parse_args().algo_name + '/models/' + curr_time +'/') # path to save models
    parser.add_argument('--save',default=True,type=bool,help="if save figure or not")   
    args = parser.parse_args()
    
    return args


def train(cfg, env, policyA, policyB, critic, router):
    print("env",cfg.env_name)
    print("algorithm",cfg.algo_name)
    rewards, ma_rewards = [], []
    actorlossAs, actorlossBs=[],[]
    sharecriticlosses=[]
    # 添加统计计数器
    total_router_choices = 0  # router选择的总次数
    total_forced_policy_uses = 0  # 强制使用policy的总次数
    policy_0_choices = 0  # 选择policy_idx 0的次数
    policy_1_choices = 0  # 选择policy_idx 1的次数
    policy_2_choices = 0  # 选择policy_idx 2的次数
     # 新增：存储 price 和 fee 的容器（按 episode 存储）
    price_f_records = []       # 每个元素为该 episode 的 price_f 列表
    price_c_records = []       # 每个元素为该 episode 的 price_c 列表
    purchase_fee_totals = []   # 每个元素为该 episode 的 purchase_fee 累加值（标量）
    energyfee_totals = []      # 每个元素为该 episode 的 energyfee 累加值（标量）
    ep_penaltys=[]
    for i_ep in range(cfg.train_eps):
        state = env.reset(cfg.season)
        ep_reward = 0
        ep_entropy = 0
        ep_purchase_fee_average = 0.0
        ep_energyfee_averafe = 0.0
        ep_penalty_sum = 0.0
    
        for time_step in range(cfg.time_eps):
            force_policy_index = env.env_knowledge(time_step,cfg.season)  # 获取环境知识，决定是否强制使用某个policy
            
            if force_policy_index:  # 强制使用特定策略
                total_forced_policy_uses += 1  # 记录强制使用policy的次数
                
                if force_policy_index == 2:
                    policy_idx = 1
                else:
                    policy_idx = 0
                
            else: 
                # 路由决策
                total_router_choices += 1  # 记录router选择的次数
                # 在每步训练时，记录 router 的 log_prob
                logits = router(torch.FloatTensor(state).unsqueeze(0))
                # print("state", state)
                # print("logits:", logits)
                prob = torch.softmax(logits, dim=1)
                m = torch.distributions.Categorical(prob)
                policy_idx = m.sample().item()
                log_prob = m.log_prob(torch.tensor(policy_idx))
            
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
                "fundamental": env.fundamental,   # 当前基础负荷数组
                "PG_rated": env.PG_rated,
                # "log_curve_mapping": env.log_curve_mapping  # 可传入函数引用
            }
            
            if policy_idx == 0: #进行储能
                power_input, ess_change, PG,action= policyA.select_action(state,context,policy_idx,time_step)
                policy_0_choices += 1
            else: 
                power_input, ess_change, PG,action= policyB.select_action(state,context,policy_idx,time_step)
                policy_1_choices += 1
            next_state, reward, done, price_f, price_c, purchase_fee, energyfee ,penalty= env.step(time_step,policy_idx,power_input, ess_change, PG)
            ep_reward += reward
            ep_entropy += 0
            ep_purchase_fee_average += float(purchase_fee)
            ep_energyfee_averafe += float(energyfee)
            ep_penalty_sum += float(penalty)
            if policy_idx == 0:
                # 存储转换到policyA的经验池
                policyA.store_transition(state, action, reward, next_state, done)
            elif policy_idx == 1:
                # 存储转换到policyB的经验池
                policyB.store_transition(state, action, reward, next_state, done)
               
            if not force_policy_index:
                # 存储Router单步数据（用于后续计算累积奖励）
                router.store_step_data(state, policy_idx, reward)
            
            critic.collect_data(state,next_state,reward, done)

            state = next_state

        # Episode结束，计算累积奖励并存入经验池，然后进行更新
        # 将本 episode 的 price/fee 数据保存到全局容器

        price_f_records.append(price_f)
        price_c_records.append(price_c)
        purchase_fee_totals.append(ep_purchase_fee_average/len(ep_purchase_fee_average))
        energyfee_totals.append(ep_energyfee_averafe/len(ep_energyfee_averafe))
        ep_penaltys.append(ep_penalty_sum)
        # Episode结束，计算累积奖励并存入经验池，然后进行更新
        router.finish_episode()
        router.update()
        # 更新策略网络
        # 回合结束后，更新所有actor（自动跳过未使用的actor）
        actorA_loss = policyA.update(critic)  
        actorB_loss = policyB.update(critic)  
    
        critic_loss = critic.update() 
        rewards.append(ep_reward)
        actorlossBs.append(actorB_loss)
        actorlossAs.append(actorA_loss)
        sharecriticlosses.append(critic_loss)

        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)

        if (i_ep+1)%10 == 0:  
            print(f'Episode:{i_ep+1}, Reward:{ep_reward:.2f}')
            print(f'  Router选择次数: {total_router_choices}, 强制使用policy次数: {total_forced_policy_uses}')
            print(f'  Router选择比例: {total_router_choices/(total_router_choices+total_forced_policy_uses)*100:.2f}%')
            total_policy_choices = policy_0_choices + policy_1_choices + policy_2_choices
            if total_policy_choices > 0:
                print(f'  Policy选择: 0({policy_0_choices}, {policy_0_choices/total_policy_choices*100:.2f}%), 1({policy_1_choices}, {policy_1_choices/total_policy_choices*100:.2f}%), 2({policy_2_choices}, {policy_2_choices/total_policy_choices*100:.2f}%)')
    
    # 训练结束后的最终统计
    print(f'\n=== 训练完成统计 ===')
    print(f'总Router选择次数: {total_router_choices}')
    print(f'总强制使用policy次数: {total_forced_policy_uses}')
    print(f'Router选择比例: {total_router_choices/(total_router_choices+total_forced_policy_uses)*100:.2f}%')
    print(f'强制使用policy比例: {total_forced_policy_uses/(total_router_choices+total_forced_policy_uses)*100:.2f}%')
    
    # Policy选择统计
    total_policy_choices = policy_0_choices + policy_1_choices + policy_2_choices
    print(f'\n=== Policy选择统计 ===')
    print(f'Policy 0选择次数: {policy_0_choices} ({policy_0_choices/total_policy_choices*100:.2f}%)')
    print(f'Policy 1选择次数: {policy_1_choices} ({policy_1_choices/total_policy_choices*100:.2f}%)')
    print(f'Policy 2选择次数: {policy_2_choices} ({policy_2_choices/total_policy_choices*100:.2f}%)')
    print(f'总Policy选择次数: {total_policy_choices}')
    
    # 返回包含 price/fee 数据
    return {
        'rewards': rewards,
        'ma_rewards': ma_rewards,
        'price_f_records': price_f_records,
        'price_c_records': price_c_records,
        'purchase_fee_totals': purchase_fee_totals,
        'energyfee_totals': energyfee_totals,
        'actorlossA': actorlossAs,
        'actorlossB': actorlossBs,
        'criticloss': sharecriticlosses,
        'ep_penaltys': ep_penaltys
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
    
    action_dimA = 3
    action_dimB = 2
    state_dim = 5  # 根据实际情况调整状态维度
    
    # 检查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == 'cuda' else "cpu")
    print(f"使用设备: {device}")
    
    # 使用新的ActorCritic类
    policyA = Actor(state_dim, action_dimA, cfg)
    policyB = Actor(state_dim, action_dimB, cfg)
    router = RouterNet(state_dim, cfg.batch_size, cfg.gamma)
    shared_critic=SharedCritic(state_dim,cfg) 

    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # Obtain current time
    result_path="outputs/" + cfg.env_name + '/'+cfg.algo_name +'/'+cfg.season+ curr_time +'/results/' + '/'
    model_path= "outputs/" + cfg.env_name + '/'+cfg.algo_name +'/'+ cfg.season+ curr_time +'/models/' + '/'   

    import os
    res_dic = train(cfg, env, policyA, policyB, shared_critic,router)
    # 确保模型目录存在
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    plot_rewards(res_dic['rewards'], res_dic['ma_rewards'], result_path=result_path, tag="train")
    plot_losses(res_dic['actorlossA'], algo="ActorA", save=True, path=result_path + "actorA_")
    plot_losses(res_dic['actorlossB'], algo="ActorB", save=True, path   =result_path + "actorB_")
    plot_losses(res_dic['criticloss'], algo="SharedCritic", save=True, path=result_path + "sharedcritic_") 
    plot_losses(res_dic['ep_penaltys'], algo="Penalty", save=True, path=result_path + "penalty_")
    save_results(res_dic['rewards'], res_dic['ma_rewards'], tag='train', path=result_path)  # save results
    save_result(res_dic['price_f_records'], tag='price_f', path=result_path)
    save_result(res_dic['price_c_records'], tag='price_c', path=result_path)
    save_result(res_dic['purchase_fee_totals'], tag='purchase_fee', path=result_path)
    save_result(res_dic['energyfee_totals'], tag='energyfee', path=result_path) 
    save_result(res_dic['ep_penaltys'], tag='penalty', path=result_path)
    torch.save(policyA.actor.state_dict(), model_path + "/policyA_actor.pth")
    torch.save(policyB.actor.state_dict(), model_path + "/policyB_actor.pth")
    torch.save(policyA.critic.state_dict(), model_path + "/policyA_critic.pth")
    torch.save(policyB.critic.state_dict(), model_path + "/policyB_critic.pth")

    print("模型保存完成！")


    # # 测试
    # test_res = test(cfg, env, policyA, policyB, router)
    # plot_rewards(test_res['rewards'], test_res['ma_rewards'], cfg, tag="test")
    # save_results(test_res['rewards'], test_res['ma_rewards'], tag='test', path=cfg.result_path)
    
    