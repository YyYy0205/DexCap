#!/usr/bin/env python3
"""
DexCap 采集数据 → LeRobot 数据集格式（HuggingFace）

输出结构：
  dataset/<repo_id>/
    ├── meta/
    │   ├── info.json       # 数据集元信息
    │   ├── episodes.jsonl  # 每个episode的帧数等
    │   ├── tasks.jsonl     # 任务描述列表
    │   └── stats.json      # 特征统计（自动计算）
    ├── data/chunk-000/episode_NNNNNN.parquet
    └── videos/chunk-000/observation.images.top/episode_NNNNNN.mp4

特征：
  action              (10,)    [right_dpos(3), right_dyaw(1), right_grip(1),
                                left_dpos(3),  left_dyaw(1),  left_grip(1)]
  observation.state   (14,)    [r_eef_pos(3), r_eef_quat(4), l_eef_pos(3), l_eef_quat(4)]
  observation.images.top  (H,W,3)  RGB 图像

用法：
  cd so101_lerobot
  python build_lerobot_dataset.py
  python build_lerobot_dataset.py --config config.yaml
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# 复用 so101_replay 的数据加载和变换逻辑
sys.path.insert(0, str(Path(__file__).parent.parent / "so101_replay"))
from data_loader import load_all_frames, lowpass
from gripper_utils import GripperMapper
from transform_utils import compute_target

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def yaw_to_quat_wxyz(yaw_array):
    """(N,) yaw → (N,4) 四元数 wxyz，绕 Z 轴。"""
    half = np.asarray(yaw_array, dtype=np.float32) / 2.0
    q = np.zeros((len(half), 4), dtype=np.float32)
    q[:, 0] = np.cos(half)
    q[:, 3] = np.sin(half)
    return q


def process_demo(demo_dir: Path, replay_cfg: dict, action_gap: int, img_size):
    """
    处理单个 demo，返回:
        states     (N, 14)  observation.state
        actions    (N, 10)  action
        images     (N, H, W, 3)  uint8 RGB
    """
    tcfg = replay_cfg["transform"]
    fcfg = replay_cfg["filter"]
    hz   = replay_cfg["data"]["fps"]

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

    eef_pos_r = np.zeros((n, 3), dtype=np.float32)
    eef_yaw_r = np.zeros(n, dtype=np.float32)
    eef_pos_l = np.zeros((n, 3), dtype=np.float32)
    eef_yaw_l = np.zeros(n, dtype=np.float32)
    grip_r = np.zeros(n, dtype=np.float32)
    grip_l = np.zeros(n, dtype=np.float32)

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

    eef_quat_r = yaw_to_quat_wxyz(eef_yaw_r)
    eef_quat_l = yaw_to_quat_wxyz(eef_yaw_l)

    # observation.state (N, 14)
    states = np.concatenate([eef_pos_r, eef_quat_r, eef_pos_l, eef_quat_l],
                            axis=1).astype(np.float32)

    # delta actions (N, 10)
    def delta_pad(arr, gap):
        d = arr[gap:] - arr[:-gap]
        pad = np.zeros((gap, *arr.shape[1:]), dtype=arr.dtype)
        return np.concatenate([d, pad], axis=0)

    dpos_r = delta_pad(eef_pos_r, action_gap)
    dyaw_r = delta_pad(eef_yaw_r[:, None], action_gap).squeeze(1)
    dyaw_r = np.arctan2(np.sin(dyaw_r), np.cos(dyaw_r))
    dpos_l = delta_pad(eef_pos_l, action_gap)
    dyaw_l = delta_pad(eef_yaw_l[:, None], action_gap).squeeze(1)
    dyaw_l = np.arctan2(np.sin(dyaw_l), np.cos(dyaw_l))

    grip_r_act = np.concatenate([grip_r[action_gap:], np.full(action_gap, grip_r[-1])])
    grip_l_act = np.concatenate([grip_l[action_gap:], np.full(action_gap, grip_l[-1])])

    actions = np.concatenate([
        dpos_r, dyaw_r[:, None], grip_r_act[:, None],
        dpos_l, dyaw_l[:, None], grip_l_act[:, None],
    ], axis=1).astype(np.float32)

    # 图像
    frame_dirs = sorted(
        [d for d in demo_dir.iterdir() if d.is_dir() and d.name.startswith("frame_")],
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

    return states, actions, images


def build_features(img_size, use_videos):
    """LeRobot features 字典。"""
    H, W = img_size if img_size else (480, 640)
    return {
        "action": {
            "dtype": "float32",
            "shape": (10,),
            "names": [
                "right_dpos_x", "right_dpos_y", "right_dpos_z",
                "right_dyaw", "right_grip",
                "left_dpos_x", "left_dpos_y", "left_dpos_z",
                "left_dyaw", "left_grip",
            ],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
                "right_eef_x", "right_eef_y", "right_eef_z",
                "right_eef_qw", "right_eef_qx", "right_eef_qy", "right_eef_qz",
                "left_eef_x", "left_eef_y", "left_eef_z",
                "left_eef_qw", "left_eef_qx", "left_eef_qy", "left_eef_qz",
            ],
        },
        "observation.images.top": {
            "dtype": "video" if use_videos else "image",
            "shape": (H, W, 3),
            "names": ["height", "width", "channels"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force",  action="store_true",
                    help="输出目录已存在时强制覆盖")
    args = ap.parse_args()

    here = Path(__file__).parent
    cfg = load_config(args.config)
    replay_cfg = load_config(here / cfg["replay_config"])

    repo_id    = cfg["repo_id"]
    output_root = (here / cfg["output_root"]).resolve()
    fps        = cfg["fps"]
    action_gap = cfg.get("action_gap", 1)
    img_size   = cfg.get("image_size")
    use_videos = cfg.get("use_videos", True)
    task_desc  = cfg.get("task", "dual-arm manipulation")
    demo_dirs  = cfg["demo_dirs"]

    dataset_root = output_root / repo_id
    if dataset_root.exists():
        if args.force:
            print(f"Removing existing: {dataset_root}")
            shutil.rmtree(dataset_root)
        else:
            print(f"[ERROR] 目录已存在：{dataset_root}\n使用 --force 覆盖。")
            sys.exit(1)

    print(f"Building LeRobot dataset: {repo_id}")
    print(f"  Output       : {dataset_root}")
    print(f"  FPS          : {fps}")
    print(f"  Image size   : {img_size}")
    print(f"  Use videos   : {use_videos}")
    print(f"  Demos        : {len(demo_dirs)}")
    print(f"  Action gap   : {action_gap}")
    print()

    features = build_features(img_size, use_videos)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=dataset_root,
        robot_type="so101_dual",
        use_videos=use_videos,
        image_writer_processes=0,
        image_writer_threads=4,
    )

    total = 0
    for idx, rel_dir in enumerate(demo_dirs):
        demo_dir = (here / rel_dir).resolve()
        print(f"  [{idx}] {demo_dir} ...")
        states, actions, images = process_demo(demo_dir, replay_cfg, action_gap, img_size)
        n = len(actions)

        for i in range(n):
            frame = {
                "action":                 actions[i],
                "observation.state":      states[i],
                "observation.images.top": images[i],
                "task":                   task_desc,
            }
            dataset.add_frame(frame)

        dataset.save_episode()
        total += n
        print(f"       frames={n}  actions_range=[{actions.min():.3f}, {actions.max():.3f}]")

    print(f"\nDone. episodes={len(demo_dirs)}  total_frames={total}")
    print(f"  → {dataset_root}")
    print(f"\n加载测试：")
    print(f"  from lerobot.datasets.lerobot_dataset import LeRobotDataset")
    print(f"  ds = LeRobotDataset('{repo_id}', root='{dataset_root}')")
    print(f"  print(ds[0].keys())")


if __name__ == "__main__":
    main()
