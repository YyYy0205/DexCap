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
import rerun as rr

HERE = Path(__file__).parent.resolve()

# 复用 so101_train 的真机环境封装（IK, 机器人接口）
sys.path.insert(0, str(HERE.parent / "so101_train"))
sys.path.insert(0, str(HERE.parent / "so101_replay"))

from env_real_so101 import EnvRealSO101

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.configs.policies import PreTrainedConfig


# ─────────────────────────────────────────────────────────────────────────────
# 自定义 ACT (.pth) 的 LeRobot 兼容包装层
# ─────────────────────────────────────────────────────────────────────────────

class CustomACTWrapper:
    """
    把 train_act_so101.py 训练出的 .pth checkpoint 包装成
    LeRobot policy 接口（select_action / config.input_features）。
    内置动作 buffer：每 chunk_size 步推理一次，中间直接取 buffer。
    """

    class _Config:
        def __init__(self, img_shape=(3, 240, 320)):
            class _F:
                def __init__(self, shape): self.shape = shape
            self.input_features = {
                "observation.state":      _F((14,)),
                "observation.images.top": _F(img_shape),
            }

    def __init__(self, ckpt_path: str, device: str):
        from train_act_so101 import ACTPolicy

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sa   = ckpt["args"]
        stats = ckpt["stats"]
        d_model = sa.get("d_model", 256)

        self._policy = ACTPolicy(
            state_dim=14, action_dim=10,
            chunk_size=sa["chunk_size"],
            latent_dim=32,
            d_model=d_model,
            n_heads=4 if d_model <= 256 else 8,
            n_enc=4,
            n_dec=sa.get("n_dec", 4),
            use_image=not sa.get("no_image", False),
            kl_weight=sa.get("kl_weight", 10.0),
        ).to(device)
        self._policy.load_state_dict(ckpt["model"])
        self._policy.eval()

        dev = torch.device(device)
        self._a_mean = torch.tensor(stats["a_mean"], device=dev)
        self._a_std  = torch.tensor(stats["a_std"],  device=dev)
        self._s_mean = torch.tensor(stats["s_mean"], device=dev)
        self._s_std  = torch.tensor(stats["s_std"],  device=dev)
        self._use_image = not sa.get("no_image", False)
        self._buffer: list = []
        self.config = self._Config(img_shape=(3, 240, 320))

        print(f"CustomACT loaded: chunk={sa['chunk_size']}  "
              f"d_model={d_model}  image={self._use_image}")

    def reset(self):
        self._buffer.clear()

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        """每步调用一次，buffer 空时重新推理整个 chunk。"""
        if not self._buffer:
            state = batch["observation.state"]           # (B, 14)
            state_n = (state - self._s_mean) / self._s_std

            image = None
            if self._use_image and "observation.images.top" in batch:
                image = batch["observation.images.top"]  # (B, 3, H, W) float [0,1]

            chunk = self._policy.predict(state_n, image)        # (B, T, 10)
            chunk = chunk * self._a_std + self._a_mean          # 反归一化
            for k in range(chunk.shape[1]):
                self._buffer.append(chunk[:, k, :])             # 存 (B, 10)

        return self._buffer.pop(0)   # (B, 10)


def load_policy(ckpt_path: str, device: str):
    """
    自动检测 checkpoint 类型：
      - .pth 文件  → 自定义 ACT（CustomACTWrapper）
      - 目录       → LeRobot 格式（safetensors）
    返回 (policy, pre_fn, post_fn)，自定义模型的 pre/post 为恒等映射。
    """
    path = Path(ckpt_path)
    identity = lambda x: x   # noqa: E731

    if path.suffix == ".pth":
        policy = CustomACTWrapper(str(path), device)
        return policy, identity, identity

    # LeRobot checkpoint 目录
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    PolicyCls = get_policy_class(cfg.type)
    policy = PolicyCls.from_pretrained(path, config=cfg)
    policy.to(device)
    policy.eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=path)
    return policy, pre, post


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


def run_episode(policy, pre, post, env, horizon, dry_run, device, display: bool = False):
    obs = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()

    # 推断策略期望的图像尺寸（如训练带图像而推理时没相机，用黑图占位）
    needs_image = any(k.startswith("observation.images.") for k in policy.config.input_features)

    action_names = [
        "r_dx", "r_dy", "r_dz", "r_dyaw", "r_grip",
        "l_dx", "l_dy", "l_dz", "l_dyaw", "l_grip",
    ]

    for step in range(horizon):
        t0 = time.time()
        if needs_image and "agentview_image" not in obs:
            img_shape = next(v.shape for k, v in policy.config.input_features.items()
                             if k.startswith("observation.images."))
            c, h, w = img_shape
            obs["agentview_image"] = np.zeros((h, w, c), dtype=np.uint8)
        batch = obs_to_batch(obs, device)
        batch = pre(batch)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        with torch.no_grad():
            action = policy.select_action(batch)

        action_out = post(action)
        action_np = action_out.squeeze(0).cpu().numpy()

        # ── Rerun 可视化 ────────────────────────────────────────
        if display:
            rr.set_time("step", sequence=step)
            if "agentview_image" in obs and obs["agentview_image"] is not None:
                rr.log("camera/top", rr.Image(obs["agentview_image"]))
            for i, (name, val) in enumerate(zip(action_names, action_np)):
                rr.log(f"action/{name}", rr.Scalars(float(val)))

        if dry_run:
            if step % 10 == 0:
                print(f"[{step:04d}] action={np.round(action_np, 3)}")
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
    ap.add_argument("--display",    action="store_true",
                    help="用 Rerun 实时显示摄像头画面和动作曲线")
    ap.add_argument("--device",     default="auto",
                    help="cpu / cuda / mps / auto")
    args = ap.parse_args()

    if args.device == "auto":
        args.device = ("cuda" if torch.cuda.is_available() else
                       "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {args.device}")

    # ── 初始化 Rerun ─────────────────────────────────────────────
    if args.display:
        rr.init("so101_eval", spawn=True)   # 自动弹出 Rerun 窗口
        print("Rerun viewer launched.")

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
                                args.dry_run, args.device,
                                display=args.display)
            print(f"  Finished in {steps} steps")
    finally:
        env.close()


if __name__ == "__main__":
    main()
