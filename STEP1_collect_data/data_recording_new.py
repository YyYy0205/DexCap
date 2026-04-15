import argparse
import copy
import numpy as np
import open3d as o3d
import os
import shutil
import sys
import pyrealsense2 as rs
import cv2

from realsense_helper import get_profiles
import redis
import concurrent.futures


def save_frame(
    frame_id,
    out_directory,
    color_buffer,
    depth_buffer,
    rightHandJoint_buffer,
    leftHandJoint_buffer,
    rightHandJointOri_buffer,
    leftHandJointOri_buffer,
    save_hand,
):
    frame_directory = os.path.join(out_directory, f"frame_{frame_id}")
    os.makedirs(frame_directory, exist_ok=True)

# 30-37 Save color and depth images
    cv2.imwrite(
        os.path.join(frame_directory, "color_image.jpg"),
        color_buffer[frame_id][:, :, ::-1],
    )
    cv2.imwrite(
        os.path.join(frame_directory, "depth_image.png"),
        depth_buffer[frame_id],
    )

# 38-44 Save hand joints and orientations
    if save_hand:
        np.savetxt(
            os.path.join(frame_directory, "right_hand_joint.txt"),
            rightHandJoint_buffer[frame_id],
        )
        np.savetxt(
            os.path.join(frame_directory, "left_hand_joint.txt"),
            leftHandJoint_buffer[frame_id],
        )
        np.savetxt(
            os.path.join(frame_directory, "right_hand_joint_ori.txt"),
            rightHandJointOri_buffer[frame_id],
        )
        np.savetxt(
            os.path.join(frame_directory, "left_hand_joint_ori.txt"),
            leftHandJointOri_buffer[frame_id],
        )

    return f"frame {frame_id} saved"


class RealsenseProcessor:
    def __init__(
        self,
        total_frame,
        store_frame=False,
        out_directory=None,
        save_hand=False,
        enable_visualization=False,
    ):
        self.total_frame = total_frame
        self.store_frame = store_frame
        self.out_directory = out_directory
        self.save_hand = save_hand
        self.enable_visualization = enable_visualization

        self.pipeline = None
        self.align = None
        self.rds = None

        self.color_buffer = []
        self.depth_buffer = []

        self.rightHandJoint_buffer = []
        self.leftHandJoint_buffer = []
        self.rightHandJointOri_buffer = []
        self.leftHandJointOri_buffer = []

    def configure_stream(self):
        print("[INFO] Configuring RealSense...")

        if self.save_hand:
            self.rds = redis.Redis(host="localhost", port=6669, db=0)
            print("[INFO] Connected to Redis")

        self.pipeline = rs.pipeline()
        config = rs.config()

        color_profiles, depth_profiles = get_profiles()

        # 选第一个稳定配置
        #w, h, fps, fmt = depth_profiles[0]
        #config.enable_stream(rs.stream.depth, w, h, fmt, fps)
        #w, h, fps, fmt = color_profiles[0]
        #config.enable_stream(rs.stream.color, w, h, fmt, fps)

        # ✅ 稳定配置（关键）
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        print("[INFO] RealSense started")

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        return depth_image, color_image

    def process_frame(self):
        print("[INFO] Start streaming...")
        frame_count = 0

        try:
            while frame_count < self.total_frame:
                depth_frame, color_frame = self.get_frame()

                if self.save_hand:
                    left = np.frombuffer(
                        self.rds.get("rawLeftHandJointXyz"), dtype=np.float64
                    ).reshape(21, 3)

                    right = np.frombuffer(
                        self.rds.get("rawRightHandJointXyz"), dtype=np.float64
                    ).reshape(21, 3)

                    left_ori = np.frombuffer(
                        self.rds.get("rawLeftHandJointOrientation"),
                        dtype=np.float64,
                    ).reshape(21, 4)

                    right_ori = np.frombuffer(
                        self.rds.get("rawRightHandJointOrientation"),
                        dtype=np.float64,
                    ).reshape(21, 4)

                if self.store_frame:
                    self.depth_buffer.append(copy.deepcopy(depth_frame))
                    self.color_buffer.append(copy.deepcopy(color_frame))

                    if self.save_hand:
                        self.leftHandJoint_buffer.append(left)
                        self.rightHandJoint_buffer.append(right)
                        self.leftHandJointOri_buffer.append(left_ori)
                        self.rightHandJointOri_buffer.append(right_ori)

                print(f"[FRAME] {frame_count}")
                frame_count += 1

        except Exception as e:
            print("[ERROR]", e)

        finally:
            print("[INFO] Stopping pipeline...")
            self.pipeline.stop()

            if self.store_frame:
                print("[INFO] Saving frames...")
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            save_frame,
                            i,
                            self.out_directory,
                            self.color_buffer,
                            self.depth_buffer,
                            self.rightHandJoint_buffer,
                            self.leftHandJoint_buffer,
                            self.rightHandJointOri_buffer,
                            self.leftHandJointOri_buffer,
                            self.save_hand,
                        )
                        for i in range(frame_count)
                    ]

                    for f in concurrent.futures.as_completed(futures):
                        print(f.result())


def main(args):
    processor = RealsenseProcessor(
        total_frame=1000,
        store_frame=args.store_frame,
        out_directory=args.out_directory,
        save_hand=args.store_hand,
        enable_visualization=args.enable_vis,
    )

    processor.configure_stream()
    processor.process_frame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--store_frame", action="store_true")
    parser.add_argument("--store_hand", action="store_true")
    parser.add_argument("-v", "--enable_vis", action="store_true")
    parser.add_argument("-o", "--out_directory", default="./saved_data")

    args = parser.parse_args()

    if os.path.exists(args.out_directory):
        if input("Override? (y/n): ").lower() != "y":
            sys.exit()
        shutil.rmtree(args.out_directory)

    if args.store_frame:
        os.makedirs(args.out_directory)

    main(args)
