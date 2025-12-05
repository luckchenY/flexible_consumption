import argparse
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 字体与字号设置：Times New Roman（新罗马），字号 18
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
mpl.rcParams['font.size'] = 18

def load_array(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"找不到文件: {p}")
    if p.suffix.lower() == '.npz':
        z = np.load(p)
        # 取第一个数组
        for k in z.files:
            return z[k]
        raise ValueError(f"{p} 中不包含数组")
    return np.load(p)

def main():
    parser = argparse.ArgumentParser(description="Plot rewards and ma_rewards (Times New Roman, fontsize 18)")
    parser.add_argument("--rewards", "-r", default="/home/chenyang2/flexible_consumption/outputs/standard/ppo/summer20251129-150716/results/train_rewards.npy", help="rewards numpy file (.npy/.npz)")
    parser.add_argument("--ma", "-m", default="/home/chenyang2/flexible_consumption/outputs/standard/ppo/summer20251129-150716/results/train_ma_rewards.npy", help="ma_rewards numpy file (.npy/.npz)")
    parser.add_argument("--out", "-o", default="figures/rewards_compareS.png", help="output image path")
    parser.add_argument("--title", "-t", default=None, help="图表标题")
    args = parser.parse_args()

    rewards = np.asarray(load_array(Path(args.rewards))).ravel()
    ma_rewards = np.asarray(load_array(Path(args.ma))).ravel()

    plt.figure(figsize=(8, 5))
    plt.plot(rewards[0:300], label='rewards', linewidth=1.5)
    plt.plot(ma_rewards[0:300], label='ma_rewards', linewidth=1.5)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    if args.title:
        plt.title(args.title)
    plt.legend()
    plt.grid(alpha=0.3)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outp, dpi=300)
    plt.show()

if __name__ == "__main__":
    main()