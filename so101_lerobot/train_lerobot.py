#!/usr/bin/env python3
"""
在 so101_lerobot 数据集上训练 LeRobot 策略（ACT / Diffusion Policy）。

用法：
  conda activate lerobot
  cd so101_lerobot

  # 先生成数据集
  python build_lerobot_dataset.py

  # 训练
  python train_lerobot.py --policy act                 # ACT (推荐双臂/长时序任务)
  python train_lerobot.py --policy diffusion           # Diffusion Policy (推荐精细操作)
  python train_lerobot.py --policy act --steps 50000 --batch 8
  python train_lerobot.py --policy act --device cuda   # GPU 训练

  # 单 demo 数据量小，先跑短步数验证：
  python train_lerobot.py --policy act --steps 2000 --batch 4 --save-freq 500

训练结果保存至 so101_lerobot/outputs/<job_name>/
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()


def build_cli(args):
    """把简化参数翻译成 lerobot_train 的 draccus CLI 参数。"""
    repo_id = args.repo_id
    ds_root = (HERE / args.dataset_root / repo_id).resolve()
    out_dir = (HERE / "outputs" / args.job_name).resolve()

    cli = [
        sys.executable, "-m", "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={ds_root}",
        f"--dataset.video_backend=pyav",
        f"--policy.type={args.policy}",
        f"--policy.device={args.device}",
        f"--policy.push_to_hub=false",
        f"--output_dir={out_dir}",
        f"--job_name={args.job_name}",
        f"--steps={args.steps}",
        f"--batch_size={args.batch}",
        f"--num_workers={args.num_workers}",
        f"--log_freq={args.log_freq}",
        f"--save_freq={args.save_freq}",
        f"--eval_freq=0",                  # 实机环境下没有仿真 eval，关掉
        f"--wandb.enable=false",
    ]

    # ── policy 专属参数 ──────────────────────────────────────────
    if args.policy == "act":
        # ACT 用 action chunk；chunk_size 控制一次预测多少帧
        cli += [
            f"--policy.chunk_size={args.chunk_size}",
            f"--policy.n_action_steps={args.chunk_size}",
        ]
    elif args.policy == "diffusion":
        # Diffusion 有 horizon / n_action_steps；crop_shape 需关掉或设匹配图像
        cli += [
            f"--policy.horizon={args.horizon}",
            f"--policy.n_action_steps={args.n_action_steps}",
            f"--policy.crop_shape=null",   # 240x320 图像直接用，不裁剪
        ]

    return cli


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy",  default="act", choices=["act", "diffusion"])
    ap.add_argument("--device",  default="auto", help="cpu / cuda / mps / auto")
    ap.add_argument("--steps",   type=int, default=20000, help="训练总步数")
    ap.add_argument("--batch",   type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--log-freq",    type=int, default=100)
    ap.add_argument("--save-freq",   type=int, default=2000)
    ap.add_argument("--repo-id",     default="local/so101_dexcap")
    ap.add_argument("--dataset-root", default="./dataset",
                    help="数据集根目录（相对本目录）")
    ap.add_argument("--job-name",    default=None, help="输出子目录名，默认 {policy}_so101")

    # ACT 参数
    ap.add_argument("--chunk-size", type=int, default=32,
                    help="ACT action chunk 长度（默认 32，数据少时用小值）")
    # Diffusion 参数
    ap.add_argument("--horizon",        type=int, default=16)
    ap.add_argument("--n-action-steps", type=int, default=8)

    args = ap.parse_args()

    if args.job_name is None:
        args.job_name = f"{args.policy}_so101"

    # auto device
    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    cli = build_cli(args)
    print("=" * 60)
    print("Command:")
    for i, a in enumerate(cli):
        print(f"  {a}")
    print("=" * 60)
    print(f"Device={args.device}  Steps={args.steps}  Batch={args.batch}")
    print(f"Output: {HERE / 'outputs' / args.job_name}")
    print()

    env = os.environ.copy()
    env["HF_LEROBOT_HOME"] = str((HERE / args.dataset_root).resolve())
    ret = subprocess.run(cli, env=env).returncode
    sys.exit(ret)


if __name__ == "__main__":
    main()
