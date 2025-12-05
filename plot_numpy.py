import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 字体与字号设置：Times New Roman（新罗马），字号 10
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']  # fallback
mpl.rcParams['font.size'] = 10

def load_numpy_file(path: Path):
    ext = path.suffix.lower()
    if ext == '.npy':
        return np.load(path)
    if ext == '.npz':
        z = np.load(path)
        # 取第一个数组
        for k in z.files:
            return z[k]
        raise ValueError("npz 中不包含数组")
    if ext in ('.csv', '.txt'):
        try:
            return np.loadtxt(path, delimiter=',')
        except Exception:
            return np.loadtxt(path)  # 尝试自动分隔
    # 最后一招：尝试 numpy.load
    try:
        return np.load(path)
    except Exception as e:
        raise ValueError(f"无法读取文件 {path}: {e}")

def plot_data(data: np.ndarray, out_path: Path, title: str = None):
    plt.figure(figsize=(6, 4))
    if data.ndim == 1:
        plt.plot(data, marker='.', linewidth=1)
        plt.xlabel('Index')
        plt.ylabel('Value')
    elif data.ndim == 2:
        rows, cols = data.shape
        if cols == 2:
            plt.plot(data[:, 0], data[:, 1], marker='.', linewidth=1)
            plt.xlabel('X')
            plt.ylabel('Y')
        else:
            for i in range(cols):
                plt.plot(data[:, i], label=f'col{i}', linewidth=1)
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.legend(fontsize=8)
    else:
        raise ValueError("仅支持一维或二维数组绘图")

    if title:
        plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="读入 numpy 数据并画图")
    parser.add_argument("--file", default="outputs/router_rule/A2C_router/summer20251104-024425/results/penalty.npy")
    parser.add_argument("--out",default='figures/output.png' )
    parser.add_argument("--title", help="图表标题", default=None)
    args = parser.parse_args()

    p = Path(args.file)
    if not p.exists():
        raise FileNotFoundError(f"找不到文件: {p}")

    data = load_numpy_file(p)
    out = Path(args.out) if args.out else p.with_name(p.stem + "_plot.png")
    plot_data(data, out, title=args.title)

if __name__ == "__main__":
    main()