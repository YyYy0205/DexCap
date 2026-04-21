#!/usr/bin/env python3
"""
Tracker 数据可视化工具
用法：
  python visualize_tracker.py            # 绘制全部帧
  python visualize_tracker.py --frames 300   # 只看前300帧
  python visualize_tracker.py --arm right    # 只看右臂
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_all_frames
import yaml


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def plot_tracker(demo_dir, n_frames=None, arm="both"):
    print(f"Loading frames from {demo_dir} ...")
    rp, rq, lp, lq, _, _, _, _ = load_all_frames(str(demo_dir))

    if n_frames:
        rp, rq, lp, lq = rp[:n_frames], rq[:n_frames], lp[:n_frames], lq[:n_frames]

    n = len(rp)
    t = np.arange(n)

    # 相对 frame_0 的位移
    dr = rp - rp[0]
    dl = lp - lp[0]

    # yaw 提取
    def yaw(quat):
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))

    yr = np.degrees(yaw(rq))
    yl = np.degrees(yaw(lq))
    yr -= yr[0]
    yl -= yl[0]

    show_right = arm in ("both", "right")
    show_left  = arm in ("both", "left")
    n_plots = (2 if show_right else 0) + (2 if show_left else 0)

    fig = plt.figure(figsize=(14, 3.5 * n_plots))
    fig.suptitle(f"Tracker 数据  ({n} 帧)", fontsize=13, y=1.01)
    gs = gridspec.GridSpec(n_plots, 1, hspace=0.55)

    row = 0
    colors = ["#e74c3c", "#27ae60", "#2980b9"]
    labels = ["tracker[0]=X(左右)", "tracker[1]=Y(上下)", "tracker[2]=Z(前后)"]

    def _pos_plot(ax, data, title):
        for i in range(3):
            ax.plot(t, data[:, i], color=colors[i], lw=1.2, label=labels[i])
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("帧")
        ax.set_ylabel("Δ位移 (m)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    def _yaw_plot(ax, data, title):
        ax.plot(t, data, color="#8e44ad", lw=1.2, label="yaw (deg)")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("帧")
        ax.set_ylabel("Δyaw (°)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    if show_right:
        _pos_plot(fig.add_subplot(gs[row]), dr, "右臂 Tracker — 各轴位移（相对frame_0）")
        row += 1
        _yaw_plot(fig.add_subplot(gs[row]), yr, "右臂 Tracker — Yaw 变化")
        row += 1

    if show_left:
        _pos_plot(fig.add_subplot(gs[row]), dl, "左臂 Tracker — 各轴位移（相对frame_0）")
        row += 1
        _yaw_plot(fig.add_subplot(gs[row]), yl, "左臂 Tracker — Yaw 变化")
        row += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--frames", type=int, default=None, help="只看前N帧")
    ap.add_argument("--arm", choices=["both", "right", "left"], default="both")
    args = ap.parse_args()

    cfg = load_config(args.config)
    demo_dir = Path(__file__).parent / cfg["data"]["demo_dir"]
    plot_tracker(str(demo_dir), args.frames, args.arm)
