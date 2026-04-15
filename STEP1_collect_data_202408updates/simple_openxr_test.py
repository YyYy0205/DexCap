#!/usr/bin/env python3
"""
Simple OpenXR test to diagnose runtime issues
"""

import xr

def test_openxr_runtime():
    print("Testing OpenXR runtime...")
    
    try:
        # Test 1: Get runtime name
        print(f"Runtime name: {xr.get_runtime_name()}")
        print(f"Runtime version: {xr.get_runtime_version()}")
        
        # Test 2: Create instance without extensions
        print("Creating instance without extensions...")
        instance_create_info = xr.InstanceCreateInfo()
        instance = xr.create_instance(instance_create_info)
        print("✅ Instance created successfully")
        
        # Test 3: Get system
        print("Getting system...")
        system_id = xr.get_system(instance, xr.FormFactor.HEAD_MOUNTED_DISPLAY)
        print(f"✅ System ID: {system_id}")
        
        # Test 4: Create session
        print("Creating session...")
        session_create_info = xr.SessionCreateInfo(system=system_id)
        session = xr.create_session(instance, session_create_info)
        print("✅ Session created successfully")
        
        # Clean up
        xr.destroy_session(session)
        xr.destroy_instance(instance)
        print("✅ Cleanup completed")
        
    except xr.exception.RuntimeFailureError as e:
        print(f"❌ RuntimeFailureError: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_openxr_runtime()
    if success:
        print("\n🎉 OpenXR runtime test passed!")
    else:
        print("\n💥 OpenXR runtime test failed!")
        print("\nTroubleshooting suggestions:")
        print("1. Restart SteamVR")
        print("2. Check SteamVR is in headless mode")
        print("3. Ensure Vive trackers are connected and ready")
        print("4. Try running as administrator")
