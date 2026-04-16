#!/usr/bin/env python3
"""
Vive Tracker + RealSense + Glove Data Collection
使用真正的无头模式（True Headless Mode）
不依赖显示，直接轮询tracker数据
"""

import os
import ctypes
from ctypes import cast, byref, POINTER
import time
import xr
import numpy as np
import pyrealsense2 as rs
from open3d_vis_obj import VIVEOpen3DVisualizer
import recording_utils as utils
import cv2
import redis


def main(dataset_folder):
    # 设置数据文件夹结构
    dataset_folder, data_folder = utils.setup_folder_structure(dataset_folder)

    # Initialize Redis connection
    redis_host = "localhost"
    redis_port = 6669
    redis_password = ""
    r = redis.StrictRedis(
        host=redis_host, port=redis_port, password=redis_password, decode_responses=False
    )

    visualizer = VIVEOpen3DVisualizer()
    first = True
    first_2 = True
    first_3 = True

    # 配置RealSense相机
    pipeline, pipeline_profile = utils.configure_realsense()
    intrinsics = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    utils.save_camera_intrinsics(dataset_folder, intrinsics)
    align = rs.align(rs.stream.color)

    # ===============================
    # 真正的无头模式OpenXR初始化
    # ===============================
    print("🚀 Initializing OpenXR in TRUE HEADLESS mode...")
    
    instance_create_info = xr.InstanceCreateInfo(
        enabled_extension_names=[
            xr.MND_HEADLESS_EXTENSION_NAME,
            xr.extension.HTCX_vive_tracker_interaction.NAME,
        ],
    )
    
    instance = xr.create_instance(create_info=instance_create_info)
    print(f"✅ Instance created")
    
    # 获取系统
    system_get_info = xr.SystemGetInfo(
        form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY,
    )
    system_id = xr.get_system(instance, system_get_info)
    print(f"✅ System ID: {system_id}")
    
    # 创建无图形绑定的会话（真正的无头模式）
    session_create_info = xr.SessionCreateInfo(
        system_id=system_id,
        # 注意：没有graphicsBinding，这是headless模式的关键
    )
    session = xr.create_session(instance, session_create_info)
    print(f"✅ Headless session created")
    
    # 创建参考空间（使用LOCAL，LOCAL_FLOOR在headless模式下不支持）
    identity_pose = xr.Posef(
        orientation=xr.Quaternionf(x=0, y=0, z=0, w=1),
        position=xr.Vector3f(x=0, y=0, z=0),
    )
    reference_space_create_info = xr.ReferenceSpaceCreateInfo(
        reference_space_type=xr.ReferenceSpaceType.LOCAL,
        pose_in_reference_space=identity_pose,
    )
    space = xr.create_reference_space(session, reference_space_create_info)
    print(f"✅ Reference space created")
    
    # 设置Vive Tracker交互
    enumerateViveTrackerPathsHTCX = cast(
        xr.get_instance_proc_addr(instance, "xrEnumerateViveTrackerPathsHTCX"),
        xr.PFN_xrEnumerateViveTrackerPathsHTCX
    )
    
    # 定义角色
    role_strings = [
        "handheld_object", "left_foot", "right_foot", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_knee", "right_knee", "waist", "chest",
        "camera", "keyboard",
    ]
    
    # 创建action set
    action_set = xr.create_action_set(
        instance=instance,
        create_info=xr.ActionSetCreateInfo(
            action_set_name="tracker_action_set",
            localized_action_set_name="Tracker Action Set",
            priority=0,
        ),
    )
    
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
            interaction_profile=xr.string_to_path(instance, "/interaction_profiles/htc/vive_tracker_htcx"),
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
    
    # 等待session进入FOCUSED状态
    print("⏳ Waiting for session to become FOCUSED...")
    session_is_running = False
    for _ in range(100):  # 等待最多10秒
        try:
            while True:
                event_buffer = xr.poll_event(instance)
                try:
                    event_type = xr.StructureType(event_buffer.type)
                except ValueError:
                    continue
                
                if event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                    event = cast(
                        byref(event_buffer),
                        POINTER(xr.EventDataSessionStateChanged)
                    ).contents
                    
                    if event.state == xr.SessionState.READY:
                        try:
                            xr.begin_session(
                                session=session,
                                begin_info=xr.SessionBeginInfo(
                                    primary_view_configuration_type=xr.ViewConfigurationType.PRIMARY_MONO,
                                ),
                            )
                            print("  ✅ Session began")
                        except Exception as e:
                            print(f"  ⚠️ begin_session: {e}")
                    
                    elif event.state == xr.SessionState.FOCUSED:
                        print("  ✅ Session FOCUSED!")
                        session_is_running = True
                        break
        except xr.EventUnavailable:
            pass
        
        if session_is_running:
            break
        time.sleep(0.1)
    
    if not session_is_running:
        print("❌ Failed to enter FOCUSED state. Exiting.")
        return
    
    # 枚举tracker路径
    n_paths = ctypes.c_uint32(0)
    result = enumerateViveTrackerPathsHTCX(instance, 0, byref(n_paths), None)
    print(f"📊 Found {n_paths.value} Vive tracker paths")
    
    # ===============================
    # 主采集循环
    # ===============================
    print("\n🎮 Starting data collection loop...")
    print("Press Ctrl+C to stop\n")
    
    frame_index = 0
    
    try:
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
                    try:
                        event_type = xr.StructureType(event_buffer.type)
                    except ValueError:
                        continue
                    
                    if event_type == xr.StructureType.EVENT_DATA_VIVE_TRACKER_CONNECTED_HTCX:
                        print("📡 Vive Tracker connected event!")
            except xr.EventUnavailable:
                pass
            
            # 使用当前时间（真正的无头模式，不依赖predicted_display_time）
            current_time = int(time.time() * 1e9)
            
            # 创建帧文件夹
            frame_folder = os.path.join(data_folder, f"frame_{frame_index:04d}")
            os.makedirs(frame_folder, exist_ok=True)
            
            found_tracker_count = 0
            
            # 获取每个tracker的位置
            for index, tracker_space in enumerate(tracker_action_spaces):
                try:
                    space_location = xr.locate_space(
                        space=tracker_space,
                        base_space=space,
                        time=current_time,
                    )
                    
                    if space_location.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                        pos = space_location.pose.position
                        ori = space_location.pose.orientation
                        
                        # 可视化
                        if role_strings[index] == 'right_elbow':
                            if first:
                                visualizer.set_pose_first([pos.x, pos.y, pos.z], [ori.w, ori.x, ori.y, ori.z], 0)
                                first = False
                            else:
                                visualizer.set_pose([pos.x, pos.y, pos.z], [ori.w, ori.x, ori.y, ori.z], 0)
                        elif role_strings[index] == 'left_elbow':
                            if first_2:
                                visualizer.set_pose_first([pos.x, pos.y, pos.z], [ori.w, ori.x, ori.y, ori.z], 1)
                                first_2 = False
                            else:
                                visualizer.set_pose([pos.x, pos.y, pos.z], [ori.w, ori.x, ori.y, ori.z], 1)
                        elif role_strings[index] == 'chest':
                            if first_3:
                                visualizer.set_pose_first([pos.x, pos.y, pos.z], [ori.w, ori.x, ori.y, ori.z], 2)
                                first_3 = False
                            else:
                                visualizer.set_pose([pos.x, pos.y, pos.z], [ori.w, ori.x, ori.y, ori.z], 2)
                        
                        # 保存数据
                        if role_strings[index] == 'right_elbow':
                            utils.save_pose(os.path.join(frame_folder, "right_pose.txt"), 
                                          [pos.x, pos.y, pos.z], [ori.w, ori.x, ori.y, ori.z])
                        elif role_strings[index] == 'left_elbow':
                            utils.save_pose(os.path.join(frame_folder, "left_pose.txt"),
                                          [pos.x, pos.y, pos.z], [ori.w, ori.x, ori.y, ori.z])
                        elif role_strings[index] == 'chest':
                            utils.save_pose(os.path.join(frame_folder, "chest_pose.txt"),
                                          [pos.x, pos.y, pos.z], [ori.w, ori.x, ori.y, ori.z])
                        
                        found_tracker_count += 1
                        
                except Exception as e:
                    pass
            
            if found_tracker_count == 0:
                print(f"Frame {frame_index}: no trackers found")
            else:
                print(f"Frame {frame_index}: {found_tracker_count} trackers found")
            
            # 获取RealSense帧
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if depth_frame and color_frame:
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                utils.save_image(os.path.join(frame_folder, "color.png"), color_image)
                cv2.imwrite(os.path.join(frame_folder, "depth.png"), depth_image)
            
            # 获取手套数据
            try:
                raw_left_hand_joint_xyz = np.frombuffer(r.get("rawLeftHandJointXyz"), dtype=np.float64).reshape((21, 3))
                raw_right_hand_joint_xyz = np.frombuffer(r.get("rawRightHandJointXyz"), dtype=np.float64).reshape((21, 3))
                raw_left_hand_joint_orientation = np.frombuffer(r.get("rawLeftHandJointOrientation"), dtype=np.float64).reshape((21, 4))
                raw_right_hand_joint_orientation = np.frombuffer(r.get("rawRightHandJointOrientation"), dtype=np.float64).reshape((21, 4))
                
                np.savetxt(os.path.join(frame_folder, "raw_left_hand_joint_xyz.txt"), raw_left_hand_joint_xyz)
                np.savetxt(os.path.join(frame_folder, "raw_right_hand_joint_xyz.txt"), raw_right_hand_joint_xyz)
                np.savetxt(os.path.join(frame_folder, "raw_left_hand_joint_orientation.txt"), raw_left_hand_joint_orientation)
                np.savetxt(os.path.join(frame_folder, "raw_right_hand_joint_orientation.txt"), raw_right_hand_joint_orientation)
            except Exception as e:
                print(f"  ⚠️ Glove data error: {e}")
            
            frame_index += 1
            
    except KeyboardInterrupt:
        print("\n👋 Stopping data collection...")
    
    finally:
        # 清理
        print("🧹 Cleaning up...")
        pipeline.stop()
        
        if 'session' in locals():
            xr.destroy_session(session)
        if 'instance' in locals():
            xr.destroy_instance(instance)
        
        print(f"✅ Done! Collected {frame_index} frames to {data_folder}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record Vive Tracker and RealSense Data (True Headless Mode)")
    parser.add_argument("dataset_folder", type=str, help="Folder to save the dataset")
    args = parser.parse_args()
    main(args.dataset_folder)
