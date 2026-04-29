#!/usr/bin/env python3
# Windows AppLocker/WDAC 拦截 pyarrow DLL。
# 机器人控制不需要 pyarrow，提前注入 auto-mock 截断整条加载链。
import sys as _sys, types as _types

if "pyarrow" not in _sys.modules:
    class _AM(_types.ModuleType):
        """Auto-mock：任意属性访问返回自身，调用返回 None。
        大写开头属性（类型名如 DataType/Array）返回真正的 type，
        使 isinstance(x, pa.DataType) 不抛 TypeError。
        """
        def __getattr__(self, n):
            if n and n[0].isupper():
                # 类型名：返回真正的 class 供 isinstance() 使用
                c = type(n, (), {"__init__": lambda s, *a, **kw: None,
                                 "__module__": self.__name__})
            else:
                c = _AM(f"{self.__name__}.{n}")
            object.__setattr__(self, n, c)
            return c
        def __call__(self, *a, **kw): return None
        def __iter__(self): return iter([])
        def __getitem__(self, k): return _AM(f"{self.__name__}[{k}]")

    _pa = _AM("pyarrow")
    object.__setattr__(_pa, "__version__", "17.0.0")  # pandas版本检查需要字符串
    _sys.modules["pyarrow"] = _pa
    for _s in ["lib", "compute", "fs", "ipc", "json", "parquet",
               "dataset", "types", "array", "chunked_array"]:
        _m = _AM(f"pyarrow.{_s}")
        object.__setattr__(_pa, _s, _m)
        _sys.modules[f"pyarrow.{_s}"] = _m

"""
SO-101 双臂手套遥操作

数据流：
  Rokoko 手套 → Redis → GloveReader  ─┐
  Vive Tracker → OpenXR → TrackerReader ─┤→ IK → BiSOFollower → SO-101 双臂
                                       └─ IIRFilter（低通）

用法：
  conda activate dexcap
  cd so101_teleop

  # 终端1：启动手套数据服务（NUC上）
  python ../STEP1_collect_data_202408updates/../STEP1_collect_data/redis_glove_server.py

  # 终端2：干跑（不连机器人，只打印）
  python teleop_so101.py --dry-run

  # 终端2：不用手套（夹爪保持半开）
  python teleop_so101.py --no-glove --dry-run

  # 终端2：实机
  python teleop_so101.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE.parent / "so101_replay"))

from so101_ik import SO101IK
from gripper_utils import GripperMapper
from transform_utils import compute_target

from iir_filter import IIRFilter
from tracker_reader import TrackerReader
from glove_reader import GloveReader


_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def _build_robot_action(joints_r, joints_l, g_r: float, g_l: float) -> dict:
    action = {}
    for name, val in zip(_JOINT_NAMES, joints_r):
        action[f"right_{name}.pos"] = float(np.degrees(val))
    for name, val in zip(_JOINT_NAMES, joints_l):
        action[f"left_{name}.pos"] = float(np.degrees(val))
    action["right_gripper.pos"] = float(g_r * 100.0)
    action["left_gripper.pos"]  = float(g_l * 100.0)
    return action


def _clamp_target(pos, workspace):
    """把末端目标位置限制在安全工作空间内。"""
    lo = [workspace["x"][0], workspace["y"][0], workspace["z"][0]]
    hi = [workspace["x"][1], workspace["y"][1], workspace["z"][1]]
    return np.clip(pos, lo, hi)


def _wait_for_trackers(tracker: TrackerReader,
                       required=("right_elbow", "left_elbow", "chest")):
    """阻塞直到所有必需 tracker 都读到有效位姿。"""
    print(f"Waiting for trackers {list(required)} ...")
    while True:
        poses = tracker.read()
        missing = [r for r in required if r not in poses]
        if not missing:
            print("  All required trackers found.")
            return poses
        print(f"  Still missing: {missing}")
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",   default="config.yaml")
    ap.add_argument("--dry-run",  action="store_true", help="只打印，不连机器人")
    ap.add_argument("--no-glove", action="store_true", help="不使用手套（夹爪固定 0.5）")
    args = ap.parse_args()

    cfg      = yaml.safe_load(open(HERE / args.config, encoding="utf-8"))
    icfg     = cfg["ik"]
    tcfg     = cfg["transform"]
    gcfg     = cfg["gripper"]
    fcfg     = cfg["filter"]
    ws       = cfg.get("workspace", {})
    rcfg     = cfg["robot"]
    hz       = cfg.get("control_freq", 15)
    dt       = 1.0 / hz

    # ── IK ──────────────────────────────────────────────────────
    urdf = (HERE.parent / icfg["urdf_path"]).resolve()
    ik_r = SO101IK(str(urdf))
    ik_l = SO101IK(str(urdf))

    home_r = np.array(icfg["home_eef_right"], dtype=float)
    home_l = np.array(icfg["home_eef_left"],  dtype=float)

    # ── Gripper ─────────────────────────────────────────────────
    gripper = GripperMapper(gcfg["dist_open"], gcfg["dist_closed"])

    # ── 在线低通滤波（每轴独立） ─────────────────────────────────
    filt_r = IIRFilter(fcfg["cutoff_hz"], hz, order=fcfg.get("order", 2), n_dim=3)
    filt_l = IIRFilter(fcfg["cutoff_hz"], hz, order=fcfg.get("order", 2), n_dim=3)

    # ── Tracker ─────────────────────────────────────────────────
    tracker = TrackerReader()

    # ── Glove ───────────────────────────────────────────────────
    glove = None
    if not args.no_glove:
        glove = GloveReader(**cfg.get("redis", {}))
        if glove.is_available():
            print("Glove Redis: connected.")
        else:
            print("[WARN] Redis not reachable — glove disabled. "
                  "Run redis_glove_server.py first, or use --no-glove.")
            glove = None

    # ── Robot ───────────────────────────────────────────────────
    robot = None
    if not args.dry_run:
        from lerobot.robots.bi_so_follower import BiSOFollower
        from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
        from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

        robot = BiSOFollower(BiSOFollowerConfig(
            id="so101_dual",
            right_arm_config=SOFollowerConfig(port=rcfg["port_right"]),
            left_arm_config=SOFollowerConfig(port=rcfg["port_left"]),
        ))
        robot.connect()
        print("Robot connected.")

    try:
        # ── 连接 Tracker，等待所有设备就绪 ─────────────────────
        print("Connecting to Vive Trackers (OpenXR headless)...")
        tracker.connect()

        poses0 = _wait_for_trackers(tracker)

        # 记录起始帧（参考帧）
        pos0_r,  quat0_r  = poses0["right_elbow"]
        pos0_l,  quat0_l  = poses0["left_elbow"]
        chest0             = poses0.get("chest", (None, None))[0]

        # 用第一帧初始化滤波器（消除启动阶跳）
        filt_r.step(pos0_r)
        filt_l.step(pos0_l)

        print(f"\nTeleop running at {hz} Hz. Ctrl+C to stop.\n")
        step = 0

        while True:
            t0 = time.time()

            # ── 读取 Tracker ──────────────────────────────────
            poses   = tracker.read()
            if "right_elbow" not in poses or "left_elbow" not in poses:
                time.sleep(dt)
                continue

            pos_r, quat_r = poses["right_elbow"]
            pos_l, quat_l = poses["left_elbow"]
            chest_t        = poses.get("chest", (None, None))[0]

            # ── 低通滤波 ──────────────────────────────────────
            pos_r_f = filt_r.step(pos_r)
            pos_l_f = filt_l.step(pos_l)

            # ── 坐标变换 → 末端目标 ───────────────────────────
            tgt_r, yaw_r = compute_target(
                pos_r_f, quat_r, pos0_r, quat0_r, home_r,
                tcfg["scale"],
                tcfg["right_axis_remap"], tcfg["right_axis_sign"],
                chest_t=chest_t, chest_0=chest0,
            )
            tgt_l, yaw_l = compute_target(
                pos_l_f, quat_l, pos0_l, quat0_l, home_l,
                tcfg["scale"],
                tcfg["left_axis_remap"], tcfg["left_axis_sign"],
                chest_t=chest_t, chest_0=chest0,
            )

            # ── 工作空间安全限制 ──────────────────────────────
            if ws:
                tgt_r = _clamp_target(tgt_r, ws)
                tgt_l = _clamp_target(tgt_l, ws)

            # ── IK ────────────────────────────────────────────
            joints_r = ik_r.solve(tgt_r, yaw_r)
            joints_l = ik_l.solve(tgt_l, yaw_l)

            # ── 夹爪 ──────────────────────────────────────────
            g_r = g_l = 0.5
            if glove is not None:
                rj, lj = glove.read()
                g_r = gripper.compute(rj)
                g_l = gripper.compute(lj)

            # ── 执行 ──────────────────────────────────────────
            if args.dry_run:
                if step % hz == 0:   # 每秒打印一次
                    print(f"[{step:06d}] "
                          f"R: eef={np.round(tgt_r,3)} yaw={yaw_r:.2f} grip={g_r:.2f} | "
                          f"L: eef={np.round(tgt_l,3)} yaw={yaw_l:.2f} grip={g_l:.2f}")
            else:
                robot.send_action(_build_robot_action(joints_r, joints_l, g_r, g_l))

            step += 1
            elapsed = time.time() - t0
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        tracker.disconnect()
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
