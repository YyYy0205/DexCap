#!/usr/bin/env python3
"""
读取 SO-101 双臂当前末端位置，输出可直接填入 config.yaml 的 home_eef 值。
用法：
  1. 将双臂摆到期望的起始姿态
  2. python get_home_eef.py
"""

import sys
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from so101_ik import SO101IK


def joints_obs_to_array(obs, side):
    """从 LeRobot obs dict 提取单臂5个关节角（度→弧度）"""
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    return np.array([np.radians(obs[f"{side}_{n}.pos"]) for n in names])


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    urdf = Path(__file__).parent / cfg["ik"]["urdf_path"]

    print(f"Loading IK from {urdf} ...")
    ik = SO101IK(str(urdf))

    # ── 连接机器人读取关节角 ──────────────────────────────────────
    from lerobot.robots.bi_so_follower import BiSOFollower
    from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

    rcfg = cfg["robot"]
    robot = BiSOFollower(BiSOFollowerConfig(
        id="so101_dual",
        right_arm_config=SOFollowerConfig(port=rcfg["port_right"]),
        left_arm_config=SOFollowerConfig(port=rcfg["port_left"]),
    ))
    robot.connect()
    print("Robot connected.\n")

    try:
        obs = robot.get_observation()

        joints_r = joints_obs_to_array(obs, "right")
        joints_l = joints_obs_to_array(obs, "left")

        fk_r = ik.forward(joints_r)
        fk_l = ik.forward(joints_l)

        pos_r = fk_r[:3, 3].tolist()
        pos_l = fk_l[:3, 3].tolist()

        print("当前关节角 (度):")
        for side, joints in [("right", joints_r), ("left", joints_l)]:
            names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
            vals  = {n: round(float(np.degrees(j)), 2) for n, j in zip(names, joints)}
            print(f"  {side}: {vals}")

        print("\n末端位置 (m):")
        print(f"  right: {[round(v, 4) for v in pos_r]}")
        print(f"  left:  {[round(v, 4) for v in pos_l]}")

        print("\n── 复制以下内容到 config.yaml 的 ik 字段 ──")
        print(f"  home_eef_right: {[round(v, 4) for v in pos_r]}")
        print(f"  home_eef_left:  {[round(v, 4) for v in pos_l]}")

    finally:
        try:
            robot.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
