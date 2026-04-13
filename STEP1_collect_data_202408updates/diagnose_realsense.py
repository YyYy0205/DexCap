#!/usr/bin/env python3
"""
RealSense Diagnostic Script
Run this script to diagnose RealSense camera issues on Windows
"""

import pyrealsense2 as rs

def main():
    print("=== RealSense Diagnostic Tool ===")
    print()
    
    # Check SDK version
    print("1. Checking RealSense SDK version:")
    try:
        print(f"   pyrealsense2 version: {rs.__version__}")
    except:
        print("   Could not determine pyrealsense2 version")
    print()
    
    # Check for connected devices
    print("2. Checking for connected devices:")
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("   No RealSense devices found!")
            print()
            print("   Troubleshooting steps:")
            print("   - Ensure camera is connected via USB 3.0 port")
            print("   - Check if RealSense D400 series camera is properly powered")
            print("   - Verify Windows recognizes the device in Device Manager")
            print("   - Try unplugging and reconnecting the camera")
            print("   - Restart the computer")
            return
        
        print(f"   Found {len(devices)} RealSense device(s):")
        for i, dev in enumerate(devices):
            print(f"   Device {i}:")
            print(f"     Name: {dev.get_info(rs.camera_info.name)}")
            print(f"     Serial: {dev.get_info(rs.camera_info.serial_number)}")
            print(f"     Firmware: {dev.get_info(rs.camera_info.firmware_version)}")
            print(f"     Product ID: {dev.get_info(rs.camera_info.product_id)}")
            print(f"     Product Line: {dev.get_info(rs.camera_info.product_line)}")
            
    except Exception as e:
        print(f"   Error checking devices: {e}")
    print()
    
    # Test device capabilities
    print("3. Testing device capabilities:")
    try:
        for i, dev in enumerate(devices):
            print(f"   Device {i} supported streams:")
            sensors = dev.query_sensors()
            for j, sensor in enumerate(sensors):
                print(f"     Sensor {j}: {sensor.get_info(rs.camera_info.name)}")
                stream_profiles = sensor.get_stream_profiles()
                for profile in stream_profiles:
                    if profile.is_video_stream_profile():
                        vp = profile.as_video_stream_profile()
                        print(f"       {vp.stream_type()} {vp.width()}x{vp.height()} @ {vp.fps()}fps ({vp.format()})")
    except Exception as e:
        print(f"   Error checking capabilities: {e}")
    print()
    
    # Test pipeline creation
    print("4. Testing pipeline creation:")
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Test different configurations
        test_configs = [
            (640, 480, 30, "640x480@30fps"),
            (640, 480, 60, "640x480@60fps"),
            (1280, 720, 30, "1280x720@30fps"),
            (1920, 1080, 30, "1920x1080@30fps")
        ]
        
        for width, height, fps, desc in test_configs:
            try:
                config = rs.config()
                config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                
                # Try to start pipeline briefly
                pipeline_profile = pipeline.start(config)
                print(f"   SUCCESS: {desc}")
                pipeline.stop()
                break
            except Exception as e:
                print(f"   FAILED: {desc} - {e}")
                
    except Exception as e:
        print(f"   Error testing pipeline: {e}")
    print()
    
    print("=== Diagnostic Complete ===")
    print()
    print("If you're still experiencing issues, please:")
    print("1. Ensure you have the latest Intel RealSense SDK for Windows")
    print("2. Check Windows Device Manager for RealSense device status")
    print("3. Try running Intel RealSense Viewer to verify camera works")
    print("4. Make sure no other application is using the camera")

if __name__ == "__main__":
    main()
