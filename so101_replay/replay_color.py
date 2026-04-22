#!/usr/bin/env python3
"""
回放采集数据中的 color.png 图像序列。

用法：
  cd so101_replay
  python replay_color.py                              # 使用 config.yaml 里的 demo_dir
  python replay_color.py --dir ../demo_test/data      # 指定目录
  python replay_color.py --fps 5                      # 慢放
  python replay_color.py --fps 30                     # 快放

键盘控制：
  空格     暂停 / 继续
  →  / d  下一帧（暂停时）
  ←  / a  上一帧（暂停时）
  q / ESC  退出
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def detect_frames(demo_dir):
    p = Path(demo_dir)
    dirs = sorted(
        [d for d in p.iterdir() if d.is_dir() and d.name.startswith("frame_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    return dirs


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir",    default=None, help="demo data 目录（含 frame_XXXX 子目录）")
    ap.add_argument("--config", default="config.yaml", help="replay 配置文件")
    ap.add_argument("--fps",    type=float, default=None, help="回放帧率（默认读 config.yaml）")
    ap.add_argument("--scale",  type=float, default=1.0,  help="显示缩放比例（0.5=半尺寸）")
    args = ap.parse_args()

    # ── 确定 demo 目录和帧率 ─────────────────────────────────────
    cfg_path = Path(args.config)
    if args.dir:
        demo_dir = Path(args.dir)
    elif cfg_path.exists():
        cfg = load_config(cfg_path)
        demo_dir = cfg_path.parent / cfg["data"]["demo_dir"]
    else:
        print(f"[ERROR] 请用 --dir 指定目录，或确保 {cfg_path} 存在。")
        sys.exit(1)

    if not demo_dir.exists():
        print(f"[ERROR] 目录不存在：{demo_dir}")
        sys.exit(1)

    fps = args.fps
    if fps is None and cfg_path.exists():
        try:
            fps = load_config(cfg_path)["data"]["fps"]
        except Exception:
            fps = 15.0
    if fps is None:
        fps = 15.0

    frame_dirs = detect_frames(demo_dir)
    if not frame_dirs:
        print(f"[ERROR] 在 {demo_dir} 中找不到 frame_XXXX 目录")
        sys.exit(1)

    n = len(frame_dirs)
    dt_ms = max(1, int(1000 / fps))
    print(f"目录：{demo_dir}")
    print(f"帧数：{n}  帧率：{fps} fps  每帧等待：{dt_ms} ms")
    print("键盘：空格=暂停  ←→=逐帧  q/ESC=退出")

    # ── 主循环 ────────────────────────────────────────────────────
    i = 0
    paused = False
    win = "color replay"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        frame_dir = frame_dirs[i]
        img_path = frame_dir / "color.png"

        if img_path.exists():
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            img = np.zeros((480, 640, 3), dtype=np.uint8)

        if args.scale != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * args.scale), int(h * args.scale)))

        # 叠加帧信息
        label = f"{frame_dir.name}  [{i+1}/{n}]  {fps:.0f}fps"
        if paused:
            label += "  [PAUSED]"
        cv2.putText(img, label, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(win, img)

        wait = 0 if paused else dt_ms
        key = cv2.waitKey(wait) & 0xFF

        if key in (ord('q'), 27):   # q / ESC
            break
        elif key == ord(' '):
            paused = not paused
        elif key in (83, ord('d')):  # → 或 d
            i = min(i + 1, n - 1)
        elif key in (81, ord('a')):  # ← 或 a
            i = max(i - 1, 0)
        else:
            if not paused:
                i += 1
                if i >= n:
                    print("回放结束。")
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
