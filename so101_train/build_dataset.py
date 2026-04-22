#!/usr/bin/env python3
"""
DexCap 录制数据 → robomimic 兼容 HDF5 数据集

动作格式 (10D per frame)：
  [right_dpos(3), right_dyaw(1), right_grip(1),
   left_dpos(3),  left_dyaw(1),  left_grip(1)]

观测格式：
  robot0_eef_pos   (3,)    右臂末端位置（机器人base frame，米）
  robot0_eef_quat  (4,)    右臂末端姿态（wxyz）
  robot1_eef_pos   (3,)    左臂末端位置
  robot1_eef_quat  (4,)    左臂末端姿态
  agentview_image  (H,W,3) uint8 RGB

用法：
  cd so101_train
  python build_dataset.py
  python build_dataset.py --output my_dataset.hdf5
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent.parent / "so101_replay"))
from data_loader import load_all_frames, lowpass
from gripper_utils import GripperMapper
from transform_utils import compute_target, extract_yaw


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def yaw_to_quat_wxyz(yaw_array):
    """(N,) yaw 角 → (N,4) 四元数 wxyz，绕 Z 轴旋转。"""
    half = np.asarray(yaw_array, dtype=np.float32) / 2.0
    q = np.zeros((len(half), 4), dtype=np.float32)
    q[:, 0] = np.cos(half)   # w
    q[:, 3] = np.sin(half)   # z
    return q


def build_demo(demo_dir, replay_cfg, action_gap, img_size):
    """
    处理单个演示目录，返回 obs dict 和 actions array。
    """
    tcfg = replay_cfg["transform"]
    fcfg = replay_cfg["filter"]
    hz   = replay_cfg["data"]["fps"]

    # ── 加载帧数据 ──────────────────────────────────────────────
    rp, rq, lp, lq, rj, lj, cp, _ = load_all_frames(str(demo_dir))
    n = len(rp)

    rp = lowpass(rp, fcfg["cutoff_hz"], hz, fcfg["order"])
    lp = lowpass(lp, fcfg["cutoff_hz"], hz, fcfg["order"])

    pos0_r, quat0_r = rp[0], rq[0]
    pos0_l, quat0_l = lp[0], lq[0]
    home_r = replay_cfg["ik"]["home_eef_right"]
    home_l = replay_cfg["ik"]["home_eef_left"]
    use_chest = np.any(cp != 0)
    chest0 = cp[0] if use_chest else None

    gripper = GripperMapper(
        replay_cfg["gripper"]["dist_open"],
        replay_cfg["gripper"]["dist_closed"],
    )

    # ── 计算每帧的末端目标和夹爪值 ──────────────────────────────
    eef_pos_r = np.zeros((n, 3), dtype=np.float32)
    eef_yaw_r = np.zeros(n, dtype=np.float32)
    eef_pos_l = np.zeros((n, 3), dtype=np.float32)
    eef_yaw_l = np.zeros(n, dtype=np.float32)
    grip_r    = np.zeros(n, dtype=np.float32)
    grip_l    = np.zeros(n, dtype=np.float32)

    for i in range(n):
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
        eef_pos_r[i] = tgt_r
        eef_yaw_r[i] = yaw_r
        eef_pos_l[i] = tgt_l
        eef_yaw_l[i] = yaw_l
        grip_r[i] = gripper.compute(rj[i])
        grip_l[i] = gripper.compute(lj[i])

    # ── 构建 obs ───────────────────────────────────────────────
    eef_quat_r = yaw_to_quat_wxyz(eef_yaw_r)
    eef_quat_l = yaw_to_quat_wxyz(eef_yaw_l)

    # ── 构建 delta actions ─────────────────────────────────────
    # action[i] = (state[i+gap] - state[i])，末尾用最后一帧填充
    def delta_pad(arr, gap):
        d = arr[gap:] - arr[:-gap]
        pad = np.tile(arr[-1:] - arr[-1:], (gap, *([1] * (arr.ndim - 1))))
        return np.concatenate([d, pad], axis=0)

    dpos_r = delta_pad(eef_pos_r, action_gap)   # (N, 3)
    dyaw_r = delta_pad(eef_yaw_r[:, None], action_gap).squeeze(1)  # (N,)
    dyaw_r = np.arctan2(np.sin(dyaw_r), np.cos(dyaw_r))  # 环绕到 [-π, π]
    dpos_l = delta_pad(eef_pos_l, action_gap)
    dyaw_l = delta_pad(eef_yaw_l[:, None], action_gap).squeeze(1)
    dyaw_l = np.arctan2(np.sin(dyaw_l), np.cos(dyaw_l))

    # 夹爪用绝对目标值（超前 gap 帧）
    grip_r_act = np.concatenate([grip_r[action_gap:], np.full(action_gap, grip_r[-1])])
    grip_l_act = np.concatenate([grip_l[action_gap:], np.full(action_gap, grip_l[-1])])

    actions = np.concatenate([
        dpos_r, dyaw_r[:, None], grip_r_act[:, None],
        dpos_l, dyaw_l[:, None], grip_l_act[:, None],
    ], axis=1).astype(np.float32)  # (N, 10)

    # ── 加载图像 ───────────────────────────────────────────────
    frame_dirs = sorted(
        [d for d in Path(demo_dir).iterdir()
         if d.is_dir() and d.name.startswith("frame_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    H, W = img_size if img_size else (480, 640)
    images = np.zeros((n, H, W, 3), dtype=np.uint8)
    for fi, fd in enumerate(frame_dirs[:n]):
        img_path = fd / "color.png"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img_size:
                    img = cv2.resize(img, (W, H))
                images[fi] = img

    obs = {
        "robot0_eef_pos":  eef_pos_r,
        "robot0_eef_quat": eef_quat_r.astype(np.float32),
        "robot1_eef_pos":  eef_pos_l,
        "robot1_eef_quat": eef_quat_l.astype(np.float32),
        "agentview_image": images,
    }

    return obs, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config_train.yaml")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg        = load_config(args.config)
    replay_cfg = load_config(Path(args.config).parent / cfg["replay_config"])
    out_path   = Path(args.output or cfg["output"])
    action_gap = cfg.get("action_gap", 1)
    img_size   = cfg.get("image_size")   # [H, W] or None
    demo_dirs  = cfg["demo_dirs"]

    print(f"Building dataset: {len(demo_dirs)} demo(s)  action_gap={action_gap}  → {out_path}")

    with h5py.File(out_path, "w") as f:
        data_grp = f.create_group("data")
        total = 0

        for idx, rel_dir in enumerate(demo_dirs):
            demo_dir = Path(args.config).parent.parent / rel_dir
            print(f"  [{idx}] {demo_dir} ...")

            obs, actions = build_demo(demo_dir, replay_cfg, action_gap, img_size)
            n = len(actions)

            grp     = data_grp.create_group(f"demo_{idx}")
            obs_grp = grp.create_group("obs")

            grp.attrs["num_samples"] = n
            for k, v in obs.items():
                obs_grp.create_dataset(k, data=v, compression="gzip", compression_opts=4)

            grp.create_dataset("actions", data=actions)
            grp.create_dataset("dones",   data=np.zeros(n, dtype=np.uint8))
            grp.create_dataset("rewards", data=np.zeros(n, dtype=np.float32))
            grp.create_dataset("states",  data=np.zeros((n, 1), dtype=np.float32))

            total += n
            print(f"       frames={n}  actions={actions.shape}  "
                  f"dpos_r_range=[{actions[:,:3].min():.3f}, {actions[:,:3].max():.3f}]")

        data_grp.attrs["total"] = total
        data_grp.attrs["env_args"] = json.dumps({
            "env_name": "SO101DualArm",
            "type": 3,
            "env_kwargs": {},
        })

    print(f"\nDone. Total={total} frames  →  {out_path}")
    print(f"Action 格式 (10D): [right_dpos(3), right_dyaw(1), right_grip(1),"
          f" left_dpos(3), left_dyaw(1), left_grip(1)]")


if __name__ == "__main__":
    main()
