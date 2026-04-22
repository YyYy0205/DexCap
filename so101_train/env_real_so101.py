#!/usr/bin/env python3
"""
SO-101 双臂 robomimic 环境包装器

动作格式 (10D)：
  [right_dpos(3), right_dyaw(1), right_grip(1),
   left_dpos(3),  left_dyaw(1),  left_grip(1)]
  所有值需先经 action_scale 反归一化

用法（策略推理）：
  env = EnvRealSO101.from_config("config_train.yaml")
  obs = env.reset()
  obs, reward, done, info = env.step(action)
  env.close()
"""

import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "so101_replay"))
from so101_ik import SO101IK
from gripper_utils import GripperMapper

try:
    import robomimic.envs.env_base as EB
    _BASE = EB.EnvBase
except ImportError:
    _BASE = object   # robomimic 未安装时退化为普通类


_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


class EnvRealSO101(_BASE):
    """
    SO-101 双臂实机环境，兼容 robomimic EnvBase 接口。
    动作输入为归一化后的增量控制指令。
    """

    def __init__(
        self,
        replay_cfg: dict,
        train_cfg: dict,
        camera_name: str = "agentview",
        render: bool = False,
        replay_cfg_dir: Path | None = None,
    ):
        self._rcfg  = replay_cfg
        self._tcfg  = train_cfg
        self._camera = camera_name

        # URDF 路径相对 replay 配置文件所在目录解析（默认 so101_replay/）
        urdf_base = replay_cfg_dir or (Path(__file__).parent.parent / "so101_replay")
        urdf = (urdf_base / replay_cfg["ik"]["urdf_path"]).resolve()
        self._ik_r = SO101IK(str(urdf))
        self._ik_l = SO101IK(str(urdf))

        self._gripper_map = GripperMapper(
            replay_cfg["gripper"]["dist_open"],
            replay_cfg["gripper"]["dist_closed"],
        )

        # 动作缩放（归一化 [-1,1] → 真实单位）
        ascale = train_cfg.get("action_scale", {})
        dp = ascale.get("dpos", 0.05)
        dy = ascale.get("dyaw", 0.3)
        dg = ascale.get("grip", 1.0)
        self._action_scale = np.array([dp, dp, dp, dy, dg,
                                        dp, dp, dp, dy, dg], dtype=np.float32)

        self._rate_dt = 1.0 / train_cfg["deploy"]["control_freq"]

        # 当前末端状态（robot frame）
        self._eef_pos_r = np.array(replay_cfg["ik"]["home_eef_right"], dtype=np.float32)
        self._eef_yaw_r = 0.0
        self._eef_pos_l = np.array(replay_cfg["ik"]["home_eef_left"],  dtype=np.float32)
        self._eef_yaw_l = 0.0
        self._grip_r    = 0.5
        self._grip_l    = 0.5

        self._robot  = None   # BiSOFollower，connect() 后赋值
        self._camera_dev = None

    # ──────────────────────────────────────────────────────────────
    # 工厂方法
    # ──────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, train_config_path: str) -> "EnvRealSO101":
        path = Path(train_config_path).resolve()
        with open(path) as f:
            tcfg = yaml.safe_load(f)
        replay_cfg_path = (path.parent / tcfg["replay_config"]).resolve()
        with open(replay_cfg_path) as f:
            rcfg = yaml.safe_load(f)
        return cls(rcfg, tcfg, replay_cfg_dir=replay_cfg_path.parent)

    # ──────────────────────────────────────────────────────────────
    # 连接 / 断开
    # ──────────────────────────────────────────────────────────────

    def connect(self):
        from lerobot.robots.bi_so_follower import BiSOFollower
        from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
        from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig

        rcfg = self._rcfg["robot"]
        self._robot = BiSOFollower(BiSOFollowerConfig(
            id="so101_dual",
            right_arm_config=SOFollowerConfig(port=rcfg["port_right"]),
            left_arm_config=SOFollowerConfig(port=rcfg["port_left"]),
        ))
        self._robot.connect()
        print("Robot connected.")

    def close(self):
        if self._robot is not None:
            try:
                self._robot.disconnect()
            except Exception:
                pass
        if self._camera_dev is not None:
            try:
                self._camera_dev.stop()
            except Exception:
                pass

    # ──────────────────────────────────────────────────────────────
    # robomimic 接口
    # ──────────────────────────────────────────────────────────────

    def reset(self):
        """重置到 home 位置并返回初始观测。"""
        self._eef_pos_r = np.array(self._rcfg["ik"]["home_eef_right"], dtype=np.float32)
        self._eef_yaw_r = 0.0
        self._eef_pos_l = np.array(self._rcfg["ik"]["home_eef_left"],  dtype=np.float32)
        self._eef_yaw_l = 0.0
        self._ik_r.reset()
        self._ik_l.reset()

        if self._robot is not None:
            # 发送归零动作
            joints_r = self._ik_r.solve(self._eef_pos_r, self._eef_yaw_r)
            joints_l = self._ik_l.solve(self._eef_pos_l, self._eef_yaw_l)
            self._robot.send_action(self._build_action(joints_r, joints_l, 0.5, 0.5))
            time.sleep(1.0)

        return self.get_observation()

    def step(self, action: np.ndarray):
        """
        action: (10,) 归一化动作，范围约 [-1, 1]
        返回: (obs, reward, done, info)
        """
        t0 = time.time()
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (10,), f"Expected (10,) action, got {action.shape}"

        # 反归一化
        raw = action * self._action_scale

        # 更新末端状态
        self._eef_pos_r = self._eef_pos_r + raw[0:3]
        self._eef_yaw_r = self._eef_yaw_r + raw[3]
        self._grip_r    = float(np.clip(raw[4], 0.0, 1.0))
        self._eef_pos_l = self._eef_pos_l + raw[5:8]
        self._eef_yaw_l = self._eef_yaw_l + raw[8]
        self._grip_l    = float(np.clip(raw[9], 0.0, 1.0))

        # IK 求解
        joints_r = self._ik_r.solve(self._eef_pos_r, self._eef_yaw_r)
        joints_l = self._ik_l.solve(self._eef_pos_l, self._eef_yaw_l)

        if self._robot is not None:
            self._robot.send_action(
                self._build_action(joints_r, joints_l, self._grip_r, self._grip_l)
            )

        # 限速
        elapsed = time.time() - t0
        time.sleep(max(0.0, self._rate_dt - elapsed))

        obs    = self.get_observation()
        reward = 0.0
        done   = False
        info   = {}
        return obs, reward, done, info

    def get_observation(self) -> dict:
        obs = {
            "robot0_eef_pos":  self._eef_pos_r.copy(),
            "robot0_eef_quat": self._yaw_to_quat(self._eef_yaw_r),
            "robot1_eef_pos":  self._eef_pos_l.copy(),
            "robot1_eef_quat": self._yaw_to_quat(self._eef_yaw_l),
        }

        img = self._capture_image()
        if img is not None:
            obs["agentview_image"] = img

        return obs

    # ──────────────────────────────────────────────────────────────
    # 内部工具
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _yaw_to_quat(yaw: float) -> np.ndarray:
        """yaw (rad) → wxyz quaternion，绕 Z 轴旋转。"""
        half = float(yaw) / 2.0
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float32)

    @staticmethod
    def _build_action(joints_r, joints_l, g_r, g_l) -> dict:
        action = {}
        for name, val in zip(_JOINT_NAMES, joints_r):
            action[f"right_{name}.pos"] = float(np.degrees(val))
        for name, val in zip(_JOINT_NAMES, joints_l):
            action[f"left_{name}.pos"] = float(np.degrees(val))
        action["right_gripper.pos"] = float(g_r * 100.0)
        action["left_gripper.pos"]  = float(g_l * 100.0)
        return action

    def _capture_image(self):
        """抓取相机图像（未接相机时返回 None）。"""
        if self._camera_dev is None:
            return None
        try:
            import pyrealsense2 as rs
            frames = self._camera_dev.wait_for_frames()
            color  = frames.get_color_frame()
            if not color:
                return None
            import numpy as np
            img = np.asanyarray(color.get_data())
            img_size = self._tcfg.get("image_size")
            if img_size:
                import cv2
                img = cv2.resize(img, (img_size[1], img_size[0]))
            return img
        except Exception:
            return None

    def connect_camera(self):
        """可选：连接 RealSense 相机。"""
        try:
            import pyrealsense2 as rs
            pipeline = rs.pipeline()
            config   = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
            pipeline.start(config)
            self._camera_dev = pipeline
            print("RealSense camera connected.")
        except Exception as e:
            print(f"[WARN] Camera not available: {e}")

    # ──────────────────────────────────────────────────────────────
    # robomimic EnvBase 要求的额外方法（最小实现）
    # ──────────────────────────────────────────────────────────────

    @property
    def name(self):
        return "SO101DualArm"

    @property
    def type(self):
        try:
            return EB.EnvType.REAL_ROBOT_TYPE
        except Exception:
            return 3

    def serialize(self):
        return {"env_name": self.name, "type": self.type, "env_kwargs": {}}

    @classmethod
    def create_for_data_processing(cls, *args, **kwargs):
        raise NotImplementedError

    # ── robomimic EnvBase 抽象方法的最小 stub 实现 ────────────────
    def action_dimension(self): return 10
    def get_state(self):        return {}
    def get_reward(self):       return 0.0
    def get_goal(self):         return {}
    def set_goal(self, **kw):   pass
    def is_done(self):          return False
    def is_success(self):       return {"task": False}
    def reset_to(self, state):  return self.reset()
    def render(self, **kw):     return None

    @property
    def rollout_exceptions(self):
        return ()


if __name__ == "__main__":
    # 快速测试（不连接机器人）
    env = EnvRealSO101.from_config("config_train.yaml")
    obs = env.reset()
    print("reset obs keys:", list(obs.keys()))
    print("robot0_eef_pos:", obs["robot0_eef_pos"])

    action = np.zeros(10, dtype=np.float32)
    action[2] = 0.5   # 右臂 dpos_z = +0.5 → 上移 0.025m（×scale=0.05）
    obs, r, done, info = env.step(action)
    print("after step robot0_eef_pos:", obs["robot0_eef_pos"])
    env.close()
