#!/usr/bin/env python3
"""
Playback script for DexCap recorded data.
Visualizes color images, depth, and hand joints.

Usage:
    python playback_dataset.py -i ./data_test [--fps 30]
"""

import argparse
import os
import sys
import numpy as np
import cv2
import open3d as o3d


def load_frame_data(frame_dir, has_hand_data=True):
    """Load all data for a single frame."""
    data = {}
    
    # Load color image
    color_path = os.path.join(frame_dir, "color_image.jpg")
    if os.path.exists(color_path):
        data['color'] = cv2.imread(color_path)
        data['color'] = cv2.cvtColor(data['color'], cv2.COLOR_BGR2RGB)
    
    # Load depth image
    depth_path = os.path.join(frame_dir, "depth_image.png")
    if os.path.exists(depth_path):
        data['depth'] = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    # Load hand joints
    if has_hand_data:
        left_joint_path = os.path.join(frame_dir, "left_hand_joint.txt")
        right_joint_path = os.path.join(frame_dir, "right_hand_joint.txt")
        left_ori_path = os.path.join(frame_dir, "left_hand_joint_ori.txt")
        right_ori_path = os.path.join(frame_dir, "right_hand_joint_ori.txt")
        
        if os.path.exists(left_joint_path):
            data['left_joints'] = np.loadtxt(left_joint_path).reshape(-1, 3)
        if os.path.exists(right_joint_path):
            data['right_joints'] = np.loadtxt(right_joint_path).reshape(-1, 3)
        if os.path.exists(left_ori_path):
            data['left_ori'] = np.loadtxt(left_ori_path).reshape(-1, 4)
        if os.path.exists(right_ori_path):
            data['right_ori'] = np.loadtxt(right_ori_path).reshape(-1, 4)
    
    return data


def create_hand_geometry(joints, color=[1, 0, 0], radius=0.005):
    """Create point cloud and lines for hand visualization."""
    if joints is None or len(joints) == 0:
        return None, None
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(joints)
    pcd.paint_uniform_color(color)
    
    # Create lines for finger connections (21 joints: wrist + 5 fingers x 4 joints)
    # Finger indices: thumb(1-4), index(5-8), middle(9-12), ring(13-16), pinky(17-20)
    lines = [
        # Wrist to finger bases
        [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
        # Thumb
        [1, 2], [2, 3], [3, 4],
        # Index
        [5, 6], [6, 7], [7, 8],
        # Middle
        [9, 10], [10, 11], [11, 12],
        # Ring
        [13, 14], [14, 15], [15, 16],
        # Pinky
        [17, 18], [18, 19], [19, 20],
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(joints)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    
    return pcd, line_set


def depth_to_pointcloud(depth, color=None, fx=600, fy=600, cx=320, cy=240, depth_scale=1000.0):
    """Convert depth image to point cloud."""
    if depth is None:
        return None
    
    height, width = depth.shape
    
    # Create coordinate grids
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Convert to 3D
    z = depth.astype(float) / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack and filter valid points
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = (z > 0).ravel() & (z < 10).ravel()  # Filter depth range
    points = points[valid]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if color is not None:
        colors = color.reshape(-1, 3) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors[valid])
    
    return pcd


class DatasetVisualizer:
    def __init__(self, data_dir, fps=30, has_hand=True, auto_play=True):
        self.data_dir = data_dir
        self.fps = fps
        self.has_hand = has_hand
        self.auto_play = auto_play
        self.is_playing = auto_play
        self.frame_delay = int(1000 / fps)
        
        # Get sorted frame list
        self.frames = sorted([
            d for d in os.listdir(data_dir) 
            if d.startswith("frame_") and os.path.isdir(os.path.join(data_dir, d))
        ])
        self.num_frames = len(self.frames)
        self.current_idx = 0
        
        print(f"[INFO] Found {self.num_frames} frames in {data_dir}")
        print(f"[INFO] Auto-play: {auto_play}, FPS: {fps}")
        
        # Open3D visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="DexCap Playback", width=1280, height=720)
        
        # Setup render options
        opt = self.vis.get_render_option()
        opt.point_size = 3.0
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        
        # Initialize geometries
        self.left_hand_pcd = None
        self.left_hand_lines = None
        self.right_hand_pcd = None
        self.right_hand_lines = None
        self.scene_pcd = None
        
        # Add coordinate frame
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(self.coord_frame)
        
        # Register key callbacks
        self._register_callbacks()
        
        # Load first frame
        self._load_and_display_frame(0)
    
    def _register_callbacks(self):
        """Register keyboard callbacks for navigation."""
        def toggle_play(vis):
            self.is_playing = not self.is_playing
            status = "PLAYING" if self.is_playing else "PAUSED"
            print(f"[STATUS] {status}")
            return False
        
        def next_frame(vis):
            self.is_playing = False
            self.current_idx = (self.current_idx + 1) % self.num_frames
            self._load_and_display_frame(self.current_idx)
            return False
        
        def prev_frame(vis):
            self.is_playing = False
            self.current_idx = (self.current_idx - 1) % self.num_frames
            self._load_and_display_frame(self.current_idx)
            return False
        
        def quit_vis(vis):
            self.vis.close()
            return False
        
        self.vis.register_key_callback(ord(" "), toggle_play)  # Space to play/pause
        self.vis.register_key_callback(ord("N"), next_frame)   # N for next
        self.vis.register_key_callback(ord("P"), prev_frame)   # P for previous
        self.vis.register_key_callback(ord("Q"), quit_vis)     # Q to quit
        self.vis.register_key_callback(ord("\x1b"), quit_vis)  # ESC to quit
    
    def _load_and_display_frame(self, idx):
        """Load frame data and update visualization."""
        frame_name = self.frames[idx]
        frame_dir = os.path.join(self.data_dir, frame_name)
        
        print(f"[FRAME {idx+1}/{self.num_frames}] {frame_name}")
        
        # Load data
        data = load_frame_data(frame_dir, self.has_hand)
        
        # Update scene point cloud
        if 'depth' in data and 'color' in data:
            new_pcd = depth_to_pointcloud(data['depth'], data['color'])
            if new_pcd is not None:
                # Downsample for performance
                new_pcd = new_pcd.voxel_down_sample(voxel_size=0.01)
                
                if self.scene_pcd is not None:
                    self.vis.remove_geometry(self.scene_pcd)
                self.scene_pcd = new_pcd
                self.vis.add_geometry(self.scene_pcd)
        
        # Update left hand
        if 'left_joints' in data:
            pcd, lines = create_hand_geometry(data['left_joints'], color=[0, 1, 0])
            if self.left_hand_pcd is not None:
                self.vis.remove_geometry(self.left_hand_pcd)
                if self.left_hand_lines is not None:
                    self.vis.remove_geometry(self.left_hand_lines)
            if pcd is not None:
                self.left_hand_pcd = pcd
                self.left_hand_lines = lines
                self.vis.add_geometry(self.left_hand_pcd)
                self.vis.add_geometry(self.left_hand_lines)
        
        # Update right hand
        if 'right_joints' in data:
            pcd, lines = create_hand_geometry(data['right_joints'], color=[1, 0, 0])
            if self.right_hand_pcd is not None:
                self.vis.remove_geometry(self.right_hand_pcd)
                if self.right_hand_lines is not None:
                    self.vis.remove_geometry(self.right_hand_lines)
            if pcd is not None:
                self.right_hand_pcd = pcd
                self.right_hand_lines = lines
                self.vis.add_geometry(self.right_hand_pcd)
                self.vis.add_geometry(self.right_hand_lines)
        
        # Also show 2D image with overlays
        if 'color' in data:
            self._show_2d_overlay(data, frame_name)
        
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def _show_2d_overlay(self, data, frame_name):
        """Show 2D image overlay in separate window."""
        img = data['color'].copy()
        
        # Add text info
        status = "PLAYING" if self.is_playing else "PAUSED"
        text = f"{frame_name} [{self.current_idx+1}/{self.num_frames}] | {status} | SPACE: Play/Pause, N: Next, P: Prev, Q: Quit"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw hand joints if available
        # Note: This requires camera intrinsics to project 3D joints to 2D
        # Since we don't have wrist pose, we can't accurately project
        # Just show raw image for now
        
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("Color View", img_bgr)
        cv2.waitKey(1)
    
    def run(self):
        """Main loop with auto-playback."""
        print("\n[CONTROLS]")
        print("  SPACE: Play/Pause toggle")
        print("  N: Next frame (pauses auto-play)")
        print("  P: Previous frame (pauses auto-play)")
        print("  Q or ESC: Quit")
        print()
        
        last_frame_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        while self.vis.poll_events():
            self.vis.update_renderer()
            
            # Handle OpenCV window and keyboard
            wait_time = 1 if self.is_playing else 50  # Faster poll when playing
            key = cv2.waitKey(wait_time) & 0xFF
            
            if key == ord('q') or key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - toggle play/pause
                self.is_playing = not self.is_playing
                status = "PLAYING" if self.is_playing else "PAUSED"
                print(f"[STATUS] {status}")
            elif key == ord('n'):  # Next frame
                self.is_playing = False
                self.current_idx = (self.current_idx + 1) % self.num_frames
                self._load_and_display_frame(self.current_idx)
            elif key == ord('p'):  # Previous frame
                self.is_playing = False
                self.current_idx = (self.current_idx - 1) % self.num_frames
                self._load_and_display_frame(self.current_idx)
            
            # Auto-advance frame if playing
            if self.is_playing:
                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                if (current_time - last_frame_time) * 1000 >= self.frame_delay:
                    self.current_idx = (self.current_idx + 1) % self.num_frames
                    self._load_and_display_frame(self.current_idx)
                    last_frame_time = current_time
        
        self.vis.destroy_window()
        cv2.destroyAllWindows()
        print("[INFO] Playback ended")


def simple_cv_playback(data_dir, fps=30, has_hand=True):
    """Simple OpenCV-only playback (no 3D) with auto-play and controls."""
    frames = sorted([
        d for d in os.listdir(data_dir) 
        if d.startswith("frame_") and os.path.isdir(os.path.join(data_dir, d))
    ])
    
    delay = int(1000 / fps)
    is_playing = True
    i = 0
    num_frames = len(frames)
    
    print("\n[CONTROLS]")
    print("  SPACE: Play/Pause")
    print("  N: Next frame")
    print("  P: Previous frame")
    print("  Q or ESC: Quit")
    print()
    
    while i < num_frames:
        frame_name = frames[i]
        frame_dir = os.path.join(data_dir, frame_name)
        data = load_frame_data(frame_dir, has_hand)
        
        if 'color' in data:
            img = cv2.cvtColor(data['color'], cv2.COLOR_RGB2BGR)
            status = "PLAYING" if is_playing else "PAUSED"
            text = f"Frame {i+1}/{num_frames} - {frame_name} - {status}"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, "SPACE: Play/Pause, N: Next, P: Prev, Q: Quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Playback", img)
        
        wait_time = delay if is_playing else 0
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q') or key == 27:  # ESC
            break
        elif key == ord(' '):  # SPACE - toggle play/pause
            is_playing = not is_playing
        elif key == ord('n'):  # Next frame
            is_playing = False
            i = min(i + 1, num_frames - 1)
        elif key == ord('p'):  # Previous frame
            is_playing = False
            i = max(i - 1, 0)
        elif is_playing:
            i += 1  # Auto-advance when playing
    
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Playback DexCap recorded dataset")
    parser.add_argument("-i", "--input", required=True, help="Path to dataset directory")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS")
    parser.add_argument("--no-hand", action="store_true", help="Dataset has no hand data")
    parser.add_argument("--simple", action="store_true", help="Use simple OpenCV playback only")
    parser.add_argument("--no-auto", action="store_true", help="Start paused instead of auto-playing")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[ERROR] Directory not found: {args.input}")
        sys.exit(1)
    
    has_hand = not args.no_hand
    auto_play = not args.no_auto
    
    if args.simple:
        simple_cv_playback(args.input, args.fps, has_hand)
    else:
        visualizer = DatasetVisualizer(args.input, args.fps, has_hand, auto_play)
        visualizer.run()


if __name__ == "__main__":
    main()
