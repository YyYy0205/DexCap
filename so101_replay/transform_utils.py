import numpy as np


def extract_yaw(quat_wxyz):
    """从四元数提取绕Y轴（竖直轴）的yaw角，返回弧度。"""
    w, x, y, z = quat_wxyz
    return float(np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z)))


def remap_axes(delta_xyz, axis_remap, axis_sign):
    """
    把 Tracker OpenXR LOCAL 坐标轴重映射到机器人任务空间。
    axis_remap: [a, b, c] 表示 robot_x=tracker[a], robot_y=tracker[b], robot_z=tracker[c]
    axis_sign:  [s0, s1, s2] 每轴的符号
    """
    out = np.array([
        delta_xyz[axis_remap[0]] * axis_sign[0],
        delta_xyz[axis_remap[1]] * axis_sign[1],
        delta_xyz[axis_remap[2]] * axis_sign[2],
    ])
    return out


def compute_target(pos_t, quat_t, pos_0, quat_0,
                   home_eef, scale, axis_remap, axis_sign,
                   chest_t=None, chest_0=None):
    """
    输入当前帧和参考帧（frame_0）的 Tracker 位姿，输出机器人末端目标。

    chest_t / chest_0: 胸前tracker当前帧和参考帧位置。
        提供时使用相对坐标（arm - chest），消除人体整体漂移。
        不提供时退化为原来的绝对位移模式。

    返回:
        target_pos: (3,) 末端目标位置（米，机器人base frame）
        delta_yaw:  float yaw偏移量（弧度）
    """
    if chest_t is not None and chest_0 is not None:
        # 相对模式：以胸前tracker为参考，消除体移
        rel_t = pos_t - chest_t
        rel_0 = pos_0 - chest_0
        delta_raw = rel_t - rel_0
    else:
        # 绝对模式（向后兼容）
        delta_raw = pos_t - pos_0

    delta_robot = remap_axes(delta_raw, axis_remap, axis_sign) * scale
    target_pos  = np.array(home_eef, dtype=float) + delta_robot
    delta_yaw   = extract_yaw(quat_t) - extract_yaw(quat_0)
    return target_pos, delta_yaw
