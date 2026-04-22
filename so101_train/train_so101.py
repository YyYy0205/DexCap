#!/usr/bin/env python3
"""
SO-101 双臂模仿学习训练入口（robomimic BC / BC-RNN）

用法：
  conda activate lerobot
  cd so101_train

  python train_so101.py                        # BC-MLP，500 epoch
  python train_so101.py --algo rnn             # BC-RNN，600 epoch（推荐）
  python train_so101.py --epochs 200 --batch 32
  python train_so101.py --no-image             # 纯 low-dim，跑速快，用于快速验证

训练结果保存至 so101_train/trained_models/<run_name>/<timestamp>/
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent.resolve()


def patch_paths(cfg_dict, here: Path):
    """把相对路径替换为绝对路径，避免 robomimic 用 package 目录作基准。"""
    data_path = Path(cfg_dict["train"]["data"])
    if not data_path.is_absolute():
        cfg_dict["train"]["data"] = str(here / data_path)

    out_path = Path(cfg_dict["train"]["output_dir"])
    if not out_path.is_absolute():
        cfg_dict["train"]["output_dir"] = str(here / out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo",     default="bc",   choices=["bc", "rnn"],
                    help="bc = BC-MLP  |  rnn = BC-RNN")
    ap.add_argument("--epochs",   type=int, default=None, help="覆盖 num_epochs")
    ap.add_argument("--batch",    type=int, default=None, help="覆盖 batch_size")
    ap.add_argument("--lr",       type=float, default=None, help="覆盖 learning rate")
    ap.add_argument("--no-image", action="store_true",
                    help="不使用图像观测（纯低维）")
    ap.add_argument("--device",   default="auto",
                    help="训练设备：cpu / cuda / auto（有GPU则用cuda）")
    args = ap.parse_args()

    # ── 选择配置文件 ──────────────────────────────────────────────
    cfg_file = HERE / ("bc_rnn_so101.json" if args.algo == "rnn" else "bc_so101.json")
    print(f"Config: {cfg_file}")

    with open(cfg_file) as f:
        cfg_dict = json.load(f)

    # ── 修补路径 ──────────────────────────────────────────────────
    patch_paths(cfg_dict, HERE)

    # ── 覆盖超参数 ────────────────────────────────────────────────
    if args.epochs is not None:
        cfg_dict["train"]["num_epochs"] = args.epochs
    if args.batch is not None:
        cfg_dict["train"]["batch_size"] = args.batch
    if args.lr is not None:
        cfg_dict["algo"]["optim_params"]["policy"]["learning_rate"]["initial"] = args.lr
    if args.no_image:
        cfg_dict["observation"]["modalities"]["obs"]["rgb"] = []
        print("  [INFO] Image observations disabled.")

    # ── 构建 robomimic config ─────────────────────────────────────
    import robomimic.config as Config
    cfg = Config.config_factory(algo_name=cfg_dict["algo_name"])
    cfg.update(cfg_dict)

    # ── 确定训练设备 ──────────────────────────────────────────────
    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    cfg.unlock()
    cfg.train.cuda = (device == "cuda")
    cfg.lock()
    print(f"Device: {device}  |  epochs={cfg.train.num_epochs}  "
          f"batch={cfg.train.batch_size}  lr={cfg.algo.optim_params.policy.learning_rate.initial}")

    # ── 开始训练 ──────────────────────────────────────────────────
    import robomimic.utils.train_utils as TrainUtils
    import robomimic.scripts.train as train_script

    # 已有同名实验目录时自动覆盖，避免非交互环境下的 input() 报错
    _orig_get_exp_dir = TrainUtils.get_exp_dir
    TrainUtils.get_exp_dir = lambda config, **kw: _orig_get_exp_dir(config, auto_remove_exp_dir=True)

    train_script.train(cfg, device=device)


if __name__ == "__main__":
    main()
