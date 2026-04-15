#!/usr/bin/env python3
"""
真正的无头模式Vive Tracker测试
不依赖显示，直接轮询tracker数据
"""

import ctypes
from ctypes import cast, byref, POINTER
import time
import xr
import numpy as np

def main():
    print("Testing true headless mode for Vive Trackers...")
    
    # 创建实例，启用headless和tracker扩展
    instance_create_info = xr.InstanceCreateInfo(
        enabled_extension_names=[
            xr.MND_HEADLESS_EXTENSION_NAME,
            xr.extension.HTCX_vive_tracker_interaction.NAME,
        ],
    )
    
    try:
        instance = xr.create_instance(create_info=instance_create_info)
        print(f"✅ Instance created: {instance}")
        
        # 获取系统 - 关键：不使用HEAD_MOUNTED_DISPLAY
        # 而是尝试获取任意可用的XR系统
        system_get_info = xr.SystemGetInfo(
            form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY,  # 暂时还是用这个，因为SteamVR需要它
        )
        system_id = xr.get_system(instance, system_get_info)
        print(f"✅ System ID: {system_id}")
        
        # 创建无图形绑定的会话（真正的无头模式）
        session_create_info = xr.SessionCreateInfo(
            system_id=system_id,
            # 注意：没有graphicsBinding，这是headless模式的关键
        )
        
        session = xr.create_session(instance, session_create_info)
        print(f"✅ Headless session created: {session}")
        
        # 创建参考空间
        reference_space_create_info = xr.ReferenceSpaceCreateInfo(
            reference_space_type=xr.ReferenceSpaceType.LOCAL_FLOOR,
            pose_in_reference_space=xr.Posef.identity(),
        )
        space = xr.create_reference_space(session, reference_space_create_info)
        print(f"✅ Reference space created: {space}")
        
        # 设置Vive Tracker交互
        enumerateViveTrackerPathsHTCX = cast(
            xr.get_instance_proc_addr(instance, "xrEnumerateViveTrackerPathsHTCX"),
            xr.PFN_xrEnumerateViveTrackerPathsHTCX
        )
        
        # 枚举tracker路径
        n_paths = ctypes.c_uint32(0)
        result = enumerateViveTrackerPathsHTCX(instance, 0, byref(n_paths), None)
        print(f"📊 Number of tracker paths: {n_paths.value}")
        
        if n_paths.value > 0:
            vive_tracker_paths = (xr.ViveTrackerPathsHTCX * n_paths.value)(
                *([xr.ViveTrackerPathsHTCX()] * n_paths.value)
            )
            result = enumerateViveTrackerPathsHTCX(instance, n_paths, byref(n_paths), vive_tracker_paths)
            
            for i in range(n_paths.value):
                paths = vive_tracker_paths[i]
                print(f"  Tracker {i}: persistent_path={paths.persistent_path}, role_path={paths.role_path}")
        
        # 创建action set
        action_set = xr.create_action_set(
            instance=instance,
            create_info=xr.ActionSetCreateInfo(
                action_set_name="tracker_action_set",
                localized_action_set_name="Tracker Action Set",
                priority=0,
            ),
        )
        
        # 定义角色
        role_strings = [
            "handheld_object", "left_foot", "right_foot", 
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow", "left_knee", 
            "right_knee", "waist", "chest",
            "camera", "keyboard",
        ]
        
        # 创建pose action
        role_path_strings = [f"/user/vive_tracker_htcx/role/{role}" for role in role_strings]
        role_paths = (xr.Path * len(role_path_strings))(
            *[xr.string_to_path(instance, role_string) for role_string in role_path_strings]
        )
        
        pose_action = xr.create_action(
            action_set=action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name="tracker_pose",
                localized_action_name="Tracker Pose",
                count_subaction_paths=len(role_paths),
                subaction_paths=role_paths,
            ),
        )
        
        # 建议绑定
        suggested_binding_paths = (xr.ActionSuggestedBinding * len(role_path_strings))(
            *[xr.ActionSuggestedBinding(
                pose_action, 
                xr.string_to_path(instance, f"{role_path_string}/input/grip/pose")
            ) for role_path_string in role_path_strings]
        )
        
        xr.suggest_interaction_profile_bindings(
            instance=instance,
            suggested_bindings=xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(instance, 
                    "/interaction_profiles/htc/vive_tracker_htcx"),
                count_suggested_bindings=len(suggested_binding_paths), 
                suggested_bindings=suggested_binding_paths,
            )
        )
        
        # 创建action spaces
        tracker_action_spaces = (xr.Space * len(role_paths))(
            *[xr.create_action_space(
                session=session,
                create_info=xr.ActionSpaceCreateInfo(action=pose_action, subaction_path=role_path)
            ) for role_path in role_paths]
        )
        
        # 附加action set
        xr.attach_session_action_sets(
            session=session,
            attach_info=xr.SessionActionSetsAttachInfo(
                count_action_sets=1,
                action_sets=(xr.ActionSet * 1)(action_set),
            ),
        )
        
        print("\n🎮 Starting tracker polling loop...")
        print("Press Ctrl+C to exit\n")
        
        # 关键区别：不使用xr.wait_frame()，而是直接轮询
        while True:
            # 同步actions
            active_action_set = xr.ActiveActionSet(
                action_set=action_set, 
                subaction_path=xr.NULL_PATH
            )
            xr.sync_actions(
                session=session,
                sync_info=xr.ActionsSyncInfo(
                    count_active_action_sets=1,
                    active_action_sets=ctypes.pointer(active_action_set),
                )
            )
            
            # 处理事件
            try:
                while True:
                    event_buffer = xr.poll_event(instance)
                    event_type = xr.StructureType(event_buffer.type)
                    
                    if event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                        event = cast(
                            byref(event_buffer),
                            POINTER(xr.EventDataSessionStateChanged)
                        ).contents
                        print(f"  Session state changed to: {event.state}")
                        
                        if event.state == xr.SessionState.READY:
                            xr.begin_session(
                                session=session,
                                begin_info=xr.SessionBeginInfo(
                                    view_configuration_type=xr.ViewConfigurationType.PRIMARY_MONO,
                                ),
                            )
                            print("  ✅ Session began")
                    
                    elif event_type == xr.StructureType.EVENT_DATA_VIVE_TRACKER_CONNECTED_HTCX:
                        print("  📡 Vive Tracker connected event!")
                        
            except xr.EventUnavailable:
                pass
            
            # 使用当前时间获取tracker位置（不依赖predicted_display_time）
            current_time = int(time.time() * 1e9)  # 纳秒
            
            found_trackers = 0
            for i, space in enumerate(tracker_action_spaces):
                try:
                    space_location = xr.locate_space(
                        space=space,
                        base_space=space,  # 使用自身作为参考
                        time=current_time,
                    )
                    
                    if space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                        pos = space_location.pose.position
                        print(f"  Tracker {role_strings[i]}: ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
                        found_trackers += 1
                        
                except Exception as e:
                    pass  # 忽略单个tracker的错误
            
            if found_trackers > 0:
                print(f"\n✅ Found {found_trackers} trackers\n")
            else:
                print("❌ No trackers found")
            
            time.sleep(0.1)  # 10Hz轮询
            
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"\n💥 Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理
        try:
            if 'session' in locals():
                xr.destroy_session(session)
            if 'instance' in locals():
                xr.destroy_instance(instance)
        except:
            pass

if __name__ == "__main__":
    main()
