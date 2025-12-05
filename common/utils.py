import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set(color_codes=True)

# from matplotlib.font_manager import FontProperties  # 导入字体模块

def plot_chargingnum(charging_nums, plot_cfg):
    plt.figure()
    x = np.arange(1, 51)
    plt.xlabel('Time')
    plt.ylabel('Number of EVs being charged')
    plt.bar(x, charging_nums, color="#87CEFA", label='# of EVs')
    plt.legend()
    if plot_cfg.save:
        # 确保目录存在
        make_dir(os.path.dirname(plot_cfg.result_path))
        plt.savefig(plot_cfg.result_path + "{}_curve".format('# of EVs'))
    plt.show()

def plot_rewards(rewards, ma_rewards, result_path, tag):
    # sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("{} curve".format(tag))     # "{}环境下{}算法的学习曲线"
    plt.xlabel('Episodes')       # 回合数
    plt.plot(rewards, label='rewards')      # 奖励
    plt.plot(ma_rewards, label='ma rewards')    # 滑动平均奖励
    plt.legend()
    
    plt.savefig(result_path+"rewards_curve.png")
    plt.show()

def plot_price(plotterm, plot_cfg, tag ):
    plt.figure()
    plt.xlabel('Times')  # 回合数
    plt.ylabel('Price(RMB)')
    plt.xticks(np.arange(0, 48, 3))
    plt.plot(plotterm, color="steelblue", label=tag)  # 原电价
    plt.legend()
    if plot_cfg.save:
        # 确保目录存在
        make_dir(os.path.dirname(plot_cfg.result_path))
        plt.savefig(plot_cfg.result_path + "{}_curve".format(tag))
    plt.show()

def plot_losses(losses, algo, save=True, path='./results'):
    # sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('Episodes')
    plt.plot(losses, label='losses')
    plt.legend()
    if save:
        # 确保目录存在
        make_dir(os.path.dirname(path))
        plt.savefig(path+"losses_curve")
    plt.show()


def save_results(rewards, ma_rewards, tag='train', path='./results'):
    ''' 保存奖励
    '''
    # 确保目录存在
    make_dir(path)
    np.save(path+'{}_rewards.npy'.format(tag), rewards)
    np.save(path+'{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('Results saving completed！')

def save_result(saveterm, tag, path='./results'):
    # 确保目录存在
    make_dir(path)
    np.save(path+'{}.npy'.format(tag), saveterm)

def make_dir(*paths):
    ''' 创建文件夹
    '''
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def del_empty_dir(*paths):
    ''' 删除目录下所有空文件夹
    '''
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))
