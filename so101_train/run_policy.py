#!/usr/bin/env python3
"""
用训练好的 robomimic 策略在真实 SO-101 上执行推理。

用法：
  cd so101_train
  python run_policy.py --checkpoint trained_models/bc_so101/models/model_epoch_500.pth
  python run_policy.py --checkpoint ... --dry-run   # 不连接机器人，只打印动作
  python run_policy.py --checkpoint ... --horizon 300
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "so101_replay"))

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils


def load_policy(ckpt_path, device):
    """加载 robomimic 训练好的策略。"""
    policy, _ = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path,
        device=device,
        verbose=True,
    )
    policy.start_episode()
    return policy


def run_rollout(policy, env, horizon, dry_run):
    """
    执行一次 rollout，返回 (步数, 是否成功)。
    """
    obs = env.reset()
    policy.start_episode()

    total_reward = 0.0
    for step in range(horizon):
        # algo.__call__ 内部调用 _prepare_observation：to_tensor + to_batch + to_device
        # 直接传原始 numpy obs dict（无 batch dim），由框架自动处理
        with torch.no_grad():
            action = policy(ob=obs)   # returns numpy array already

        action = np.array(action).flatten()

        if dry_run:
            if step % 15 == 0:
                print(f"[{step:04d}] action={np.round(action, 3)}")
            obs_next = {k: (v + np.random.randn(*v.shape) * 1e-4).astype(np.float32)
                        for k, v in obs.items()}
            obs = obs_next
        else:
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                print(f"  Episode done at step {step}")
                return step, True

    return horizon, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",  required=True, help="robomimic .pth checkpoint 路径")
    ap.add_argument("--config",      default="config_train.yaml", help="训练配置文件")
    ap.add_argument("--horizon",     type=int, default=600, help="最大执行步数")
    ap.add_argument("--dry-run",     action="store_true", help="只打印动作，不连机器人")
    ap.add_argument("--episodes",    type=int, default=1, help="执行次数")
    ap.add_argument("--device",      default="cpu", help="推理设备 (cpu/cuda)")
    args = ap.parse_args()

    device = TorchUtils.get_torch_device(try_to_use_cuda=(args.device == "cuda"))
    print(f"Device: {device}")

    # 加载策略
    print(f"Loading checkpoint: {args.checkpoint}")
    policy = load_policy(args.checkpoint, device)
    print("Policy loaded.")

    # 创建环境（不连接机器人时用 dry_run 模式绕过）
    from env_real_so101 import EnvRealSO101
    env = EnvRealSO101.from_config(args.config)

    if not args.dry_run:
        env.connect()
        env.connect_camera()

    try:
        for ep in range(args.episodes):
            print(f"\n=== Episode {ep + 1}/{args.episodes} ===")
            steps, success = run_rollout(policy, env, args.horizon, args.dry_run)
            print(f"  Steps={steps}  Success={success}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
