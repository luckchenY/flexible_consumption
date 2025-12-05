import torch
import datetime
from envs.environment import EVenv
from PG.policy_gradient import PolicyGradient
from common.utils import save_results, make_dir, save_result
from common.utils import plot_rewards
import os
import sys

curr_path = os.path.dirname(os.path.abspath(__file__))  # absolute path of current directory
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add path to system path

# cuda = torch.device("cuda:" + str(0))
# curr_path = "/home/pc/LIUJIE/RL-Pricing-Charging-Schedule/code"
curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # get current time
algo_name = "Policy_Gradient"   # name of algorithm
env_name = 'Policy_Gradient'  # name of environment

MAX_EP_STEP = 168


class PlotConfig:
    ''' parameters of drawing'''
    def __init__(self) -> None:
        self.algo_name = algo_name  # name of algorithm
        self.env_name = env_name  # name of environment
        self.result_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/results/'  # path to save results
        self.model_path = curr_path + "/outputs/" + self.env_name + \
            '/' + curr_time + '/models/'  # path to save models
        self.save = True  # whether save picture


class PGConfig:
    def __init__(self):
        # self.device = cuda
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # check GPU
        self.train_eps = 1000  # 训练的回合数
        self.batch_size = 10
        self.lr = 0.01  # 学习率
        self.gamma = 0.95
        self.hidden_dim = 128  # dimension of hidden layer


def env_agent_config(cfg, seed):
    evenv = EVenv()  # create environment
    evenv.seed(seed)  # set seed
    state_dim = evenv.state_dim
    agent = PolicyGradient(state_dim, cfg)
    return evenv, agent


def train(cfg, evenv, agent):
    print('Start to train !')
    state_pool = []  # 存放每batch_size个episode的state序列
    action_pool = []
    reward_pool = []
    rewards = []
    ma_rewards = []

    gradient = torch.zeros(128)
    grad_output = torch.zeros(128)

    for i_ep in range(cfg.train_eps):
        real_state, state = evenv.reset()
        ep_reward = 0
        for t in range(MAX_EP_STEP):
            action = agent.choose_action(state)  # 根据当前环境state选择action
            # next_state, reward, done, _ = evenv.step(action)
            reward, real_state_, next_state, charging_num = evenv.step(t, action, real_state)
            ep_reward += reward
            done = False
            if t == MAX_EP_STEP - 1:
                done = True
            state_pool.append(state)
            action_pool.append(float(action))
            reward_pool.append(reward)
            state = next_state
            real_state = real_state_
            if done:
                break
        if i_ep > 0 and i_ep % cfg.batch_size == 0:
            gradient = agent.update(reward_pool, state_pool, action_pool)
            state_pool = []  # 每个episode的state
            action_pool = []
            reward_pool = []
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(
                0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        if (i_ep+1) % 10 == 0:
            print('episode：{}/{}, cumulate reward：{}'.format(i_ep+1, cfg.train_eps, ep_reward))
        if i_ep == 990:
            print("It is valid")
            grad_output = gradient
    print('complete training！')
    return rewards, ma_rewards, grad_output


if __name__ == "__main__":
    cfg = PGConfig()
    plot_cfg = PlotConfig()
    # train
    evenv, agent = env_agent_config(cfg, seed=1)
    rewards, ma_rewards, grad_output = train(cfg, evenv, agent)
    make_dir(plot_cfg.result_path, plot_cfg.model_path)
    agent.save(path=plot_cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=plot_cfg.result_path)  # save results
    save_result(grad_output, tag='gradient', path=plot_cfg.result_path)
    plot_rewards(rewards, ma_rewards, plot_cfg, tag="train")  # draw the figure of rewards
    torch.cuda.empty_cache()
    sys.exit()
