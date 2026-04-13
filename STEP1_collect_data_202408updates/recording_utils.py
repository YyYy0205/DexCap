import os
import shutil
import cv2
import pyrealsense2 as rs

def save_pose(file_path, translation, rotation):
    with open(file_path, 'w') as f:
        f.write(' '.join(map(str, translation)) + ' ' + ' '.join(map(str, rotation)))

def save_image(file_path, image):
    cv2.imwrite(file_path, image)

def setup_folder_structure(dataset_folder):
    if os.path.exists(dataset_folder):
        overwrite = input(f"Folder {dataset_folder} already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() == 'y':
            shutil.rmtree(dataset_folder)
        else:
            print("Operation cancelled.")
            exit()
    os.makedirs(dataset_folder)
    data_folder = os.path.join(dataset_folder, "data")
    os.makedirs(data_folder)
    return dataset_folder, data_folder

def save_camera_intrinsics(dataset_folder, intrinsics):
    camera_matrix_path = os.path.join(dataset_folder, "camera_matrix.txt")
    with open(camera_matrix_path, 'w') as f:
        f.write(f"fx {intrinsics.fx}\n")
        f.write(f"fy {intrinsics.fy}\n")
        f.write(f"ppx {intrinsics.ppx}\n")
        f.write(f"ppy {intrinsics.ppy}\n")
        f.write(f"width {intrinsics.width}\n")
        f.write(f"height {intrinsics.height}\n")

def configure_realsense():
    # Check for connected devices
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No RealSense devices found!")
        print("Please check:")
        print("1. RealSense camera is connected via USB 3.0")
        print("2. RealSense drivers are installed")
        print("3. Camera is not being used by another application")
        raise RuntimeError("No RealSense devices found")
    
    print(f"Found {len(devices)} RealSense device(s):")
    for i, dev in enumerate(devices):
        print(f"  Device {i}: {dev.get_info(rs.camera_info.name)}")
        print(f"    Serial: {dev.get_info(rs.camera_info.serial_number)}")
        print(f"    Firmware: {dev.get_info(rs.camera_info.firmware_version)}")
    
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Use supported 30fps configuration (D435 doesn't support 60fps at 640x480)
    try:
        # Primary configuration - 30fps which is supported by D435
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline_profile = pipeline.start(config)
        print("Successfully started RealSense with 640x480@30fps")
        return pipeline, pipeline_profile
    except RuntimeError as e:
        print(f"Failed to start with 640x480@30fps: {e}")
        try:
            # Try 1280x720@15fps (supported by RGB camera)
            config = rs.config()
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)  # Depth only supports 6fps at 1280x720
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)  # Color supports 15fps at 1280x720
            pipeline_profile = pipeline.start(config)
            print("Successfully started RealSense with 1280x720 (depth@6fps, color@15fps)")
            return pipeline, pipeline_profile
        except RuntimeError as e2:
            print(f"Failed to start with 1280x720: {e2}")
            try:
                # Try 424x240@60fps (supported configuration)
                config = rs.config()
                config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 60)
                config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 60)
                pipeline_profile = pipeline.start(config)
                print("Successfully started RealSense with 424x240@60fps")
                return pipeline, pipeline_profile
            except RuntimeError as e3:
                print(f"Failed to start with 424x240@60fps: {e3}")
                raise RuntimeError(f"Could not configure RealSense camera. Tried multiple configurations. Last error: {e3}")
