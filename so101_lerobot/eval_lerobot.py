#!/usr/bin/env python3
"""
LeRobot 策略推理：加载 ACT / Diffusion checkpoint 在真实 SO-101 上执行。

用法：
  conda activate lerobot
  cd so101_lerobot

  # 干跑（不连机器人，只打印动作）：
  python eval_lerobot.py --checkpoint outputs/act_so101/checkpoints/last/pretrained_model --dry-run

  # 实机：
  python eval_lerobot.py --checkpoint outputs/act_so101/checkpoints/last/pretrained_model

  # 多 episode：
  python eval_lerobot.py --checkpoint ... --episodes 3 --horizon 500

依赖：
  - 训练好的 checkpoint（训练后默认保存到 outputs/<job_name>/checkpoints/last/pretrained_model）
  - so101_train/env_real_so101.py（提供与真实机器人的通讯）
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent.resolve()

# 复用 so101_train 的真机环境封装（IK, 机器人接口）
sys.path.insert(0, str(HERE.parent / "so101_train"))
sys.path.insert(0, str(HERE.parent / "so101_replay"))

from env_real_so101 import EnvRealSO101

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig


def load_policy(ckpt_dir: Path, device: str):
    """
    加载 LeRobot 训练 checkpoint。
    ckpt_dir 应该是 pretrained_model/ 子目录（含 config.json, model.safetensors 等）。
    """
    cfg = PreTrainedConfig.from_pretrained(ckpt_dir)
    cfg.device = device

    PolicyCls = get_policy_class(cfg.type)
    policy = PolicyCls.from_pretrained(ckpt_dir, config=cfg)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=ckpt_dir,
    )
    return policy, preprocessor, postprocessor


def obs_to_batch(obs: dict, device: str) -> dict:
    """
    把 env.get_observation() 返回的 dict 转成 policy 能接收的 batch 格式。
    LeRobot policy 期望:
      observation.state          (B, state_dim)
      observation.images.<name>  (B, C, H, W) float32 in [0,1]
    """
    # 把双臂的 eef_pos/quat 拼成 14D state
    state = np.concatenate([
        obs["robot0_eef_pos"],   # (3,)
        obs["robot0_eef_quat"],  # (4,)
        obs["robot1_eef_pos"],
        obs["robot1_eef_quat"],
    ]).astype(np.float32)

    batch = {
        "observation.state": torch.from_numpy(state).unsqueeze(0).to(device),
    }

    if "agentview_image" in obs:
        img = obs["agentview_image"]     # (H, W, 3) uint8
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW
        batch["observation.images.top"] = torch.from_numpy(img).unsqueeze(0).to(device)

    return batch


def run_episode(policy, pre, post, env, horizon, dry_run, device):
    obs = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    # 推断策略期望的图像尺寸（如训练带图像而推理时没相机，用黑图占位）
    needs_image = any(k.startswith("observation.images.") for k in policy.config.input_features)

    for step in range(horizon):
        t0 = time.time()
        if needs_image and "agentview_image" not in obs:
            img_shape = next(v.shape for k, v in policy.config.input_features.items()
                             if k.startswith("observation.images."))
            c, h, w = img_shape
            obs["agentview_image"] = np.zeros((h, w, c), dtype=np.uint8)
        batch = obs_to_batch(obs, device)
        batch = pre(batch)

        with torch.no_grad():
            action = policy.select_action(batch)

        action_out = post(action)
        action_np = action_out.squeeze(0).cpu().numpy()

        if dry_run:
            if step % 10 == 0:
                print(f"[{step:04d}] action={np.round(action_np, 3)}")
            # 干跑时用零观测假装环境仍在运行
            obs = {k: v.copy() if hasattr(v, 'copy') else v for k, v in obs.items()}
        else:
            obs, _, done, _ = env.step(action_np)
            if done:
                print(f"  Episode done at step {step}")
                return step

        elapsed = time.time() - t0
        if step % 20 == 0:
            print(f"  step {step}: {elapsed*1000:.1f}ms")

    return horizon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="pretrained_model 目录路径")
    ap.add_argument("--config",     default="../so101_train/config_train.yaml",
                    help="EnvRealSO101 的配置文件")
    ap.add_argument("--horizon",    type=int, default=400)
    ap.add_argument("--episodes",   type=int, default=1)
    ap.add_argument("--dry-run",    action="store_true", help="不连机器人")
    ap.add_argument("--device",     default="auto",
                    help="cpu / cuda / mps / auto")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = ("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {args.device}")

    ckpt_dir = Path(args.checkpoint).resolve()
    print(f"Loading policy from: {ckpt_dir}")
    policy, pre, post = load_policy(ckpt_dir, args.device)
    print(f"Policy type: {type(policy).__name__}")

    print(f"Creating env from: {args.config}")
    env = EnvRealSO101.from_config(args.config)
    if not args.dry_run:
        env.connect()
        env.connect_camera()

    try:
        for ep in range(args.episodes):
            print(f"\n=== Episode {ep + 1}/{args.episodes} ===")
            steps = run_episode(policy, pre, post, env, args.horizon,
                                args.dry_run, args.device)
            print(f"  Finished in {steps} steps")
    finally:
        env.close()


if __name__ == "__main__":
    main()
