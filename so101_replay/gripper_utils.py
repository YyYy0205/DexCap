import numpy as np


class GripperMapper:
    def __init__(self, dist_open=0.08, dist_closed=0.01):
        self.dist_open   = dist_open
        self.dist_closed = dist_closed

    def compute(self, joints_21x3):
        """
        joints_21x3: (21, 3) 手部关节坐标
        返回 [0.0, 1.0]，0=夹爪闭合，1=张开
        """
        thumb_tip = joints_21x3[4]
        index_tip = joints_21x3[8]
        dist = float(np.linalg.norm(thumb_tip - index_tip))
        span = self.dist_open - self.dist_closed
        if abs(span) < 1e-6:
            return 0.5   # 未标定时返回中间值
        v = (dist - self.dist_closed) / span
        return float(np.clip(v, 0.0, 1.0))


def calibrate_gripper(joints_all):
    """
    joints_all: (N, 21, 3)
    打印捏合距离统计，用于设置 dist_open / dist_closed。
    """
    dists = [np.linalg.norm(j[4] - j[8]) for j in joints_all]
    dists = np.array(dists)
    print(f"捏合距离统计：max={dists.max():.4f}  min={dists.min():.4f}  "
          f"mean={dists.mean():.4f}  std={dists.std():.4f}")
    print(f"建议：dist_open={dists.max():.4f}  dist_closed={dists.min():.4f}")
    return float(dists.max()), float(dists.min())
