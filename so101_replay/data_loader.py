import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt


def _load_pose(path):
    """读取 pose 文件，返回 (pos[3], quat_wxyz[4])，失败返回 (None, None)。"""
    try:
        v = np.loadtxt(path).flatten()
        if v.size < 3:
            return None, None
        return v[:3].astype(float), v[3:7].astype(float)
    except Exception:
        return None, None


def _load_joints(path):
    """读取 21×3 手部关节文件，失败返回 None。"""
    try:
        j = np.loadtxt(path)
        if j.shape == (21, 3):
            return j.astype(float)
        return None
    except Exception:
        return None


def _detect_frames(demo_dir):
    """自动扫描 demo_dir 下所有 frame_XXXX 目录，按编号排序。"""
    p = Path(demo_dir)
    dirs = sorted(
        [d for d in p.iterdir() if d.is_dir() and d.name.startswith("frame_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    return dirs


def load_all_frames(demo_dir):
    """
    加载 demo_dir 下所有帧，自动处理缺帧（用上一帧填充）。

    返回:
        right_pos    (N, 3)
        right_quat   (N, 4)  quat 格式：wxyz
        left_pos     (N, 3)
        left_quat    (N, 4)
        right_joints (N, 21, 3)
        left_joints  (N, 21, 3)
        chest_pos    (N, 3)   胸前tracker位置（无则全零）
        chest_quat   (N, 4)
    """
    frame_dirs = _detect_frames(demo_dir)
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_XXXX directories found in {demo_dir}")

    right_pos, right_quat = [], []
    left_pos,  left_quat  = [], []
    right_joints, left_joints = [], []
    chest_pos, chest_quat = [], []

    fallback_rp = np.zeros(3);  fallback_rq = np.array([1., 0., 0., 0.])
    fallback_lp = np.zeros(3);  fallback_lq = np.array([1., 0., 0., 0.])
    fallback_cp = np.zeros(3);  fallback_cq = np.array([1., 0., 0., 0.])
    fallback_rj = np.zeros((21, 3))
    fallback_lj = np.zeros((21, 3))

    for d in frame_dirs:
        rp, rq = _load_pose(d / "right_pose.txt")
        lp, lq = _load_pose(d / "left_pose.txt")
        cp, cq = _load_pose(d / "chest_pose.txt")
        rj = _load_joints(d / "raw_right_hand_joint_xyz.txt")
        lj = _load_joints(d / "raw_left_hand_joint_xyz.txt")

        fallback_rp = rp if rp is not None else fallback_rp
        fallback_rq = rq if rq is not None else fallback_rq
        fallback_lp = lp if lp is not None else fallback_lp
        fallback_lq = lq if lq is not None else fallback_lq
        fallback_cp = cp if cp is not None else fallback_cp
        fallback_cq = cq if cq is not None else fallback_cq
        fallback_rj = rj if rj is not None else fallback_rj
        fallback_lj = lj if lj is not None else fallback_lj

        right_pos.append(fallback_rp.copy())
        right_quat.append(fallback_rq.copy())
        left_pos.append(fallback_lp.copy())
        left_quat.append(fallback_lq.copy())
        chest_pos.append(fallback_cp.copy())
        chest_quat.append(fallback_cq.copy())
        right_joints.append(fallback_rj.copy())
        left_joints.append(fallback_lj.copy())

    return (np.array(right_pos), np.array(right_quat),
            np.array(left_pos),  np.array(left_quat),
            np.array(right_joints), np.array(left_joints),
            np.array(chest_pos), np.array(chest_quat))


def lowpass(data, cutoff_hz, sample_hz, order=4):
    """对 (N, D) 轨迹做低通滤波，沿时间轴（axis=0）。帧数不足时跳过。"""
    padlen = 3 * order
    if data.shape[0] <= padlen:
        return data
    nyq = sample_hz / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, data, axis=0)
