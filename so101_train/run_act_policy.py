#!/usr/bin/env python3
"""
ACT 策略推理脚本 — 在真实 SO-101 双臂上部署

动作格式(10D): [right_dpos(3), right_dyaw(1), right_grip(1),
                left_dpos(3),  left_dyaw(1),  left_grip(1)]

用法：
  cd so101_train

  # 干跑（不连机器人/相机，只打印动作）
  python run_act_policy.py --checkpoint trained_models/act_so101/act_epoch_2000.pth --dry-run

  # 实机（需连接 RealSense + 机器人）
  python run_act_policy.py --checkpoint trained_models/act_so101/act_epoch_2000.pth

  # 调整重规划频率：每 5 步重推理一次（默认 10 步）
  python run_act_policy.py --checkpoint ... --exec-steps 5
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "so101_replay"))

from train_act_so101 import ACTPolicy


# ─────────────────────────────────────────────────────────────────────────────
# 加载 checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_act_checkpoint(ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt["args"]
    stats      = ckpt["stats"]

    use_image = not saved_args.get("no_image", False)
    d_model   = saved_args.get("d_model", 256)

    policy = ACTPolicy(
        state_dim=14,
        action_dim=10,
        chunk_size=saved_args["chunk_size"],
        latent_dim=32,
        d_model=d_model,
        n_heads=4 if d_model <= 256 else 8,
        n_enc=4,
        n_dec=saved_args.get("n_dec", 4),
        use_image=use_image,
        kl_weight=saved_args.get("kl_weight", 10.0),
    ).to(device)

    policy.load_state_dict(ckpt["model"])
    policy.eval()

    print(f"ACT loaded: chunk={saved_args['chunk_size']}  "
          f"d_model={d_model}  image={use_image}")
    return policy, stats, saved_args


# ─────────────────────────────────────────────────────────────────────────────
# 观测预处理
# ─────────────────────────────────────────────────────────────────────────────

def make_state_tensor(obs: dict, s_mean, s_std, device):
    """obs dict → normalized state tensor (1, 14)"""
    state = np.concatenate([
        obs["robot0_eef_pos"],    # (3,)
        obs["robot0_eef_quat"],   # (4,)
        obs["robot1_eef_pos"],    # (3,)
        obs["robot1_eef_quat"],   # (4,)
    ])
    state = (state - s_mean) / s_std
    return torch.from_numpy(state).float().unsqueeze(0).to(device)


def make_image_tensor(obs: dict, img_size, device):
    """obs dict → image tensor (1, 3, H, W)"""
    if "agentview_image" not in obs or obs["agentview_image"] is None:
        return None
    img = obs["agentview_image"]   # (H, W, 3) uint8
    if img_size:
        img = cv2.resize(img, (img_size[1], img_size[0]))
    img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return img_t.unsqueeze(0).to(device)


# ─────────────────────────────────────────────────────────────────────────────
# 单次 rollout
# ─────────────────────────────────────────────────────────────────────────────

def run_rollout(policy, env, stats, train_args, img_size, device,
                horizon: int, exec_steps: int, dry_run: bool, show_camera: bool = False):
    a_mean = np.array(stats["a_mean"])
    a_std  = np.array(stats["a_std"])
    s_mean = np.array(stats["s_mean"])
    s_std  = np.array(stats["s_std"])
    chunk  = train_args["chunk_size"]
    use_image = not train_args.get("no_image", False)

    obs  = env.reset()
    step = 0
    t_start = time.time()

    print(f"Rollout start (horizon={horizon}, exec_steps={exec_steps}, "
          f"chunk_size={chunk})")

    while step < horizon:
        # ── 推理 ────────────────────────────────────────────────
        state_t = make_state_tensor(obs, s_mean, s_std, device)
        image_t = make_image_tensor(obs, img_size, device) if use_image else None

        t_infer = time.time()
        with torch.no_grad():
            action_chunk = policy.predict(state_t, image_t)   # (1, chunk, 10)
        infer_ms = (time.time() - t_infer) * 1000

        action_chunk = action_chunk.squeeze(0).cpu().numpy()  # (chunk, 10)

        # ── 执行 exec_steps 步 ──────────────────────────────────
        n_exec = min(exec_steps, chunk, horizon - step)
        for k in range(n_exec):
            # 反归一化 → 真实物理单位
            action = action_chunk[k] * a_std + a_mean   # (10,)

            if dry_run:
                if step % 30 == 0:
                    print(f"  [{step:04d}] infer={infer_ms:.0f}ms  "
                          f"dpos_r={np.round(action[:3],4)}  "
                          f"grip_r={action[4]:.2f}  "
                          f"dpos_l={np.round(action[5:8],4)}  "
                          f"grip_l={action[9]:.2f}")
                obs, _, _, _ = env.step(action)
            else:
                obs, _, done, _ = env.step(action)
                if done:
                    print(f"  Episode done at step {step}.")
                    if show_camera:
                        cv2.destroyAllWindows()
                    return step

            # ── 实时画面预览（按 Q 停止）────────────────────────
            if show_camera:
                img = obs.get("agentview_image")
                if img is not None:
                    disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.putText(disp,
                                f"step={step} grip_r={action[4]:.2f} grip_l={action[9]:.2f}",
                                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
                    cv2.imshow("ACT Camera View", disp)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        print("  [Q pressed] stopping.")
                        return step

            step += 1

        elapsed = time.time() - t_start
        fps = step / elapsed if elapsed > 0 else 0
        print(f"  Step {step:4d}/{horizon}  re-planning...  "
              f"({fps:.1f} steps/s  infer={infer_ms:.0f}ms)")

    return step


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",  required=True,
                    help="act_epoch_XXXX.pth 路径")
    ap.add_argument("--config",      default="config_train.yaml",
                    help="训练配置文件（含 replay_config / image_size 等）")
    ap.add_argument("--horizon",     type=int, default=500,
                    help="每个 episode 最大执行步数")
    ap.add_argument("--exec-steps",  type=int, default=10,
                    help="每次推理后执行几步再重推理（越小越频繁）")
    ap.add_argument("--episodes",    type=int, default=1)
    ap.add_argument("--device",      default="cpu",
                    help="推理设备（cpu / cuda）")
    ap.add_argument("--dry-run",     action="store_true",
                    help="不连接机器人和相机，只打印动作")
    ap.add_argument("--show-camera", action="store_true",
                    help="弹出窗口实时显示摄像头画面（按 Q 停止）")
    args = ap.parse_args()

    device = torch.device(args.device)

    # ── 加载模型 ──────────────────────────────────────────────
    policy, stats, train_args = load_act_checkpoint(args.checkpoint, device)

    # ── 加载配置 ──────────────────────────────────────────────
    cfg_path = HERE / args.config
    with open(cfg_path) as f:
        tcfg = yaml.safe_load(f)
    img_size = tcfg.get("image_size")   # [H, W] or None

    # ── 创建环境 ──────────────────────────────────────────────
    from env_real_so101 import EnvRealSO101
    env = EnvRealSO101.from_config(str(cfg_path))

    if not args.dry_run:
        env.connect()
        env.connect_camera()
    else:
        print("[dry-run] Robot and camera skipped.")

    if args.show_camera and args.dry_run:
        print("[WARN] --show-camera 在 dry-run 模式下无图像输入，画面为空。")

    # ── Rollout 循环 ───────────────────────────────────────────
    try:
        for ep in range(args.episodes):
            print(f"\n{'='*50}")
            print(f"Episode {ep + 1} / {args.episodes}")
            print(f"{'='*50}")
            steps = run_rollout(
                policy, env, stats, train_args, img_size, device,
                horizon=args.horizon,
                exec_steps=args.exec_steps,
                dry_run=args.dry_run,
                show_camera=args.show_camera,
            )
            print(f"Episode {ep+1} finished after {steps} steps.")
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    main()
