import numpy as np
import warnings
import ikpy.chain
from scipy.spatial.transform import Rotation

# active_links_mask 经过 URDF 链结构验证：
# [Base(fixed), shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper_frame(fixed)]
_ACTIVE_MASK = [False, True, True, True, True, True, False]
_N_LINKS = 7  # 链总节点数（含首尾fixed）


class SO101IK:
    def __init__(self, urdf_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.chain = ikpy.chain.Chain.from_urdf_file(
                urdf_path,
                active_links_mask=_ACTIVE_MASK,
            )
        self._prev = np.zeros(_N_LINKS)

    def solve(self, target_pos, target_yaw):
        """
        target_pos: (3,) 末端位置（米，机器人base frame）
        target_yaw: float 末端绕Z轴旋转角（弧度）
        返回: (5,) 关节角（弧度），对应 shoulder_pan ~ wrist_roll
        """
        rot = Rotation.from_euler("z", target_yaw).as_matrix()
        target_frame = np.eye(4)
        target_frame[:3, :3] = rot
        target_frame[:3,  3] = target_pos

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = self.chain.inverse_kinematics_frame(
                target=target_frame,
                initial_position=self._prev,
                max_iter=200,
            )

        self._prev = sol
        return sol[1:6].copy()  # 5个活动关节角

    def forward(self, joint_angles_5):
        """正运动学：(5,) → 末端4×4位姿矩阵（用于验证）"""
        full = np.zeros(_N_LINKS)
        full[1:6] = joint_angles_5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.chain.forward_kinematics(full)

    def reset(self):
        self._prev = np.zeros(_N_LINKS)
