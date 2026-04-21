#!/usr/bin/env python3
"""
DexCap demo → SO-101 双臂回放
用法：
  conda activate lerobot
  cd so101_replay

  # 1. 先干跑，确认坐标和IK输出合理
  python replay_demo_so101.py --dry-run

  # 2. 统计夹爪标定值
  python replay_demo_so101.py --calibrate

  # 3. 慢速实机（config.yaml 中 dry_run: false, speed_scale: 0.3）
  python replay_demo_so101.py

  # 4. 全速
  python replay_demo_so101.py --speed 1.0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from data_loader     import load_all_frames, lowpass
from transform_utils import compute_target
from so101_ik        import SO101IK
from gripper_utils   import GripperMapper, calibrate_gripper


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def build_action(joints_r, joints_l, g_r, g_l):
    """
    BiSOFollower.send_action 的 key 格式：{side}_{motor}.pos
    SOFollower.send_action 只处理以 .pos 结尾的 key。
    度数（use_degrees=True），gripper 0-100。
    """
    action = {}
    for name, val in zip(_JOINT_NAMES, joints_r):
        action[f"right_{name}.pos"] = float(np.degrees(val))
    for name, val in zip(_JOINT_NAMES, joints_l):
        action[f"left_{name}.pos"] = float(np.degrees(val))
    action["right_gripper.pos"] = float(g_r * 100.0)   # [0,1] → [0,100]
    action["left_gripper.pos"]  = float(g_l * 100.0)
    return action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",    default="config.yaml")
    ap.add_argument("--dry-run",   action="store_true", help="只打印，不发控制命令")
    ap.add_argument("--calibrate", action="store_true", help="统计夹爪标定值后退出")
    ap.add_argument("--speed",     type=float, default=None, help="覆盖 speed_scale")
    ap.add_argument("--frames",    type=int,   default=None, help="只回放前N帧（调试用）")
    args = ap.parse_args()

    cfg       = load_config(args.config)
    dry_run   = args.dry_run or cfg["robot"]["dry_run"]
    speed     = args.speed   if args.speed is not None else cfg["robot"]["speed_scale"]
    demo_dir  = Path(__file__).parent / cfg["data"]["demo_dir"]
    tcfg      = cfg["transform"]
    fcfg      = cfg["filter"]

    # ── 加载数据 ──────────────────────────────────────────────────
    print(f"Loading frames from {demo_dir} ...")
    rp, rq, lp, lq, rj, lj, cp, _ = load_all_frames(str(demo_dir))
    n_frames = len(rp) if args.frames is None else min(args.frames, len(rp))
    rp, rq, lp, lq = rp[:n_frames], rq[:n_frames], lp[:n_frames], lq[:n_frames]
    rj, lj = rj[:n_frames], lj[:n_frames]
    cp = cp[:n_frames]

    # 胸前tracker是否有效（非全零则启用相对坐标模式）
    use_chest = np.any(cp != 0)
    if use_chest:
        print(f"  Chest tracker detected → using relative coordinate mode.")
    else:
        print(f"  No chest tracker → using absolute displacement mode.")
    print(f"  {n_frames} frames loaded.")

    # ── 夹爪标定 ──────────────────────────────────────────────────
    if args.calibrate:
        print("\n=== 右手夹爪 ===")
        calibrate_gripper(rj)
        print("\n=== 左手夹爪 ===")
        calibrate_gripper(lj)
        print("\n将以上建议值填入 config.yaml 的 gripper 字段后再运行回放。")
        return

    # ── 低通滤波 ──────────────────────────────────────────────────
    hz = cfg["data"]["fps"]
    co = fcfg["cutoff_hz"]
    od = fcfg["order"]
    rp = lowpass(rp, co, hz, od)
    lp = lowpass(lp, co, hz, od)
    print(f"  Low-pass filter applied ({co}Hz cutoff).")

    # ── 初始化 IK 和夹爪 ─────────────────────────────────────────
    urdf = Path(__file__).parent / cfg["ik"]["urdf_path"]
    print(f"Loading IK from {urdf} ...")
    ik_r = SO101IK(str(urdf))
    ik_l = SO101IK(str(urdf))

    gripper = GripperMapper(
        cfg["gripper"]["dist_open"],
        cfg["gripper"]["dist_closed"],
    )

    pos0_r, quat0_r = rp[0], rq[0]
    pos0_l, quat0_l = lp[0], lq[0]
    chest0 = cp[0] if use_chest else None
    home_r = cfg["ik"]["home_eef_right"]
    home_l = cfg["ik"]["home_eef_left"]

    # ── 连接机器人 ────────────────────────────────────────────────
    robot = None
    if not dry_run:
        from lerobot.robots.bi_so_follower import BiSOFollower
        from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
        from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

        rcfg = cfg["robot"]
        robot = BiSOFollower(BiSOFollowerConfig(
            id="so101_dual",   # 使左右臂分别存 so101_dual_right.json / so101_dual_left.json
            right_arm_config=SOFollowerConfig(port=rcfg["port_right"]),
            left_arm_config=SOFollowerConfig(port=rcfg["port_left"]),
        ))
        robot.connect()
        print("Robot connected.")

    # ── 回放主循环 ────────────────────────────────────────────────
    dt = 1.0 / hz / speed
    mode = "DRY-RUN" if dry_run else f"LIVE (speed={speed}x)"
    print(f"\n{'='*50}")
    print(f"Replaying {n_frames} frames  [{mode}]")
    print(f"Press Ctrl+C to stop.\n")

    try:
        for i in range(n_frames):
            t0 = time.time()

            chest_i = cp[i] if use_chest else None
            tgt_r, yaw_r = compute_target(
                rp[i], rq[i], pos0_r, quat0_r, home_r,
                tcfg["scale"], tcfg["right_axis_remap"], tcfg["right_axis_sign"],
                chest_t=chest_i, chest_0=chest0,
            )
            tgt_l, yaw_l = compute_target(
                lp[i], lq[i], pos0_l, quat0_l, home_l,
                tcfg["scale"], tcfg["left_axis_remap"], tcfg["left_axis_sign"],
                chest_t=chest_i, chest_0=chest0,
            )

            joints_r = ik_r.solve(tgt_r, yaw_r)
            joints_l = ik_l.solve(tgt_l, yaw_l)

            g_r = gripper.compute(rj[i])
            g_l = gripper.compute(lj[i])

            if dry_run:
                if i % 30 == 0:
                    raw_r = rp[i] - pos0_r   # 原始tracker Δpos，用于标定axis_remap
                    raw_l = lp[i] - pos0_l
                    print(f"[{i:04d}] "
                          f"R_raw=({raw_r[0]:+.3f},{raw_r[1]:+.3f},{raw_r[2]:+.3f}) "
                          f"R_eef={np.round(tgt_r, 3)} grip={g_r:.2f} | "
                          f"L_raw=({raw_l[0]:+.3f},{raw_l[1]:+.3f},{raw_l[2]:+.3f}) "
                          f"L_eef={np.round(tgt_l, 3)} grip={g_l:.2f}")
            else:
                action = build_action(joints_r, joints_l, g_r, g_l)
                robot.send_action(action)

            elapsed = time.time() - t0
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception as e:
        import traceback
        print(f"\n[ERROR] Frame loop crashed: {e}")
        traceback.print_exc()

    finally:
        if robot is not None:
            try:
                robot.disconnect()
                print("Robot disconnected.")
            except Exception as e:
                print(f"[WARN] disconnect error (ignored): {e}")
        print(f"Done. Replayed {n_frames} frames.")


if __name__ == "__main__":
    main()
