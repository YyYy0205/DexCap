"""
实时 Vive Tracker 读取器（OpenXR 无头模式）。

从 vive_realsense_glove_datacollection_headless.py 中提取 tracker 读取逻辑，
封装为可复用的类。不依赖 RealSense 相机。
"""

import ctypes
import time
from ctypes import POINTER, byref, cast

import numpy as np
import xr


_ROLES = [
    "handheld_object", "left_foot", "right_foot",
    "left_shoulder",   "right_shoulder",
    "left_elbow",      "right_elbow",
    "left_knee",       "right_knee",
    "waist", "chest", "camera", "keyboard",
]


class TrackerReader:
    """
    OpenXR 无头模式 Vive Tracker 读取器。

    用法：
        reader = TrackerReader()
        reader.connect()          # 初始化 OpenXR，等待 FOCUSED
        poses = reader.read()     # 返回 {role: (pos_xyz, quat_wxyz)} dict
        reader.disconnect()
    """

    def __init__(self):
        self.instance = None
        self.session  = None
        self.space    = None
        self._action_set     = None
        self._pose_action    = None
        self._tracker_spaces = None
        self._last: dict = {}   # role → (pos np(3,), quat np(4,))

    def connect(self):
        # ── 创建 Instance ─────────────────────────────────────────
        self.instance = xr.create_instance(
            xr.InstanceCreateInfo(
                enabled_extension_names=[
                    "XR_MND_headless",
                    "XR_HTCX_vive_tracker_interaction",
                ]
            )
        )

        system_id = xr.get_system(
            self.instance,
            xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY),
        )

        # ── 创建无图形绑定的 headless session ─────────────────────
        self.session = xr.create_session(
            self.instance,
            xr.SessionCreateInfo(system_id=system_id),
        )

        # ── 参考空间 (LOCAL) ──────────────────────────────────────
        self.space = xr.create_reference_space(
            self.session,
            xr.ReferenceSpaceCreateInfo(
                reference_space_type=xr.ReferenceSpaceType.LOCAL,
                pose_in_reference_space=xr.Posef(
                    orientation=xr.Quaternionf(x=0, y=0, z=0, w=1),
                    position=xr.Vector3f(x=0, y=0, z=0),
                ),
            ),
        )

        # ── Action set & pose action ──────────────────────────────
        role_path_strs = [f"/user/vive_tracker_htcx/role/{r}" for r in _ROLES]
        role_paths = (xr.Path * len(_ROLES))(
            *[xr.string_to_path(self.instance, s) for s in role_path_strs]
        )

        self._action_set = xr.create_action_set(
            self.instance,
            xr.ActionSetCreateInfo(
                action_set_name="teleop_tracker_set",
                localized_action_set_name="Teleop Tracker Set",
                priority=0,
            ),
        )

        self._pose_action = xr.create_action(
            self._action_set,
            xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="tracker_pose",
                localized_action_name="Tracker Pose",
                count_subaction_paths=len(role_paths),
                subaction_paths=role_paths,
            ),
        )

        xr.suggest_interaction_profile_bindings(
            self.instance,
            xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(
                    self.instance, "/interaction_profiles/htc/vive_tracker_htcx"
                ),
                count_suggested_bindings=len(role_path_strs),
                suggested_bindings=(xr.ActionSuggestedBinding * len(role_path_strs))(
                    *[xr.ActionSuggestedBinding(
                        self._pose_action,
                        xr.string_to_path(self.instance, f"{s}/input/grip/pose"),
                    )
                      for s in role_path_strs]
                ),
            ),
        )

        self._tracker_spaces = (xr.Space * len(_ROLES))(
            *[xr.create_action_space(
                self.session,
                xr.ActionSpaceCreateInfo(action=self._pose_action, subaction_path=rp),
            )
              for rp in role_paths]
        )

        xr.attach_session_action_sets(
            self.session,
            xr.SessionActionSetsAttachInfo(
                count_action_sets=1,
                action_sets=(xr.ActionSet * 1)(self._action_set),
            ),
        )

        # ── 等待 session 进入 FOCUSED ─────────────────────────────
        print("  Waiting for OpenXR session FOCUSED state...")
        for _ in range(100):
            try:
                while True:
                    ev = xr.poll_event(self.instance)
                    try:
                        et = xr.StructureType(ev.type)
                    except ValueError:
                        continue

                    if et == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                        e = cast(byref(ev),
                                 POINTER(xr.EventDataSessionStateChanged)).contents
                        if e.state == xr.SessionState.READY:
                            xr.begin_session(
                                self.session,
                                xr.SessionBeginInfo(
                                    primary_view_configuration_type=
                                        xr.ViewConfigurationType.PRIMARY_MONO,
                                ),
                            )
                        elif e.state == xr.SessionState.FOCUSED:
                            print("  OpenXR session FOCUSED.")
                            return
            except xr.EventUnavailable:
                pass
            time.sleep(0.1)

        raise RuntimeError("OpenXR session did not reach FOCUSED state within 10s")

    def read(self) -> dict:
        """
        同步 actions，定位所有 tracker。
        返回 dict: role → (pos np(3,) float32, quat_wxyz np(4,) float32)
        失效 tracker 沿用上一帧数据。
        """
        active = xr.ActiveActionSet(
            action_set=self._action_set,
            subaction_path=xr.NULL_PATH,
        )
        xr.sync_actions(
            self.session,
            xr.ActionsSyncInfo(
                count_active_action_sets=1,
                active_action_sets=ctypes.pointer(active),
            ),
        )

        # 清空事件队列
        try:
            while True:
                xr.poll_event(self.instance)
        except xr.EventUnavailable:
            pass

        now = int(time.time() * 1e9)
        for i, role in enumerate(_ROLES):
            try:
                loc = xr.locate_space(
                    space=self._tracker_spaces[i],
                    base_space=self.space,
                    time=now,
                )
                if loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                    p = loc.pose.position
                    o = loc.pose.orientation
                    self._last[role] = (
                        np.array([p.x, p.y, p.z], dtype=np.float32),
                        np.array([o.w, o.x, o.y, o.z], dtype=np.float32),
                    )
            except Exception:
                pass

        return dict(self._last)

    def disconnect(self):
        if self.session is not None:
            try:
                xr.destroy_session(self.session)
            except Exception:
                pass
            self.session = None
        if self.instance is not None:
            try:
                xr.destroy_instance(self.instance)
            except Exception:
                pass
            self.instance = None
