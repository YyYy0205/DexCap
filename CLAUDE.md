# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DexCap is a scalable and portable motion capture (mocap) data collection system for dexterous manipulation. It captures human hand motion using Rokoko mocap gloves and Intel RealSense cameras, then processes the data to train robot policies using imitation learning.

## Architecture

The codebase is organized into three sequential pipeline steps:

### STEP1_collect_data
Data collection scripts using Rokoko mocap gloves and RealSense cameras. Runs on a Windows mini-PC (NUC) connected to a portable Wi-Fi router.
- `redis_glove_server.py` - Streams mocap glove data via Redis
- `data_recording.py` - Records synchronized camera + mocap data
- `replay_human_traj_vis.py` - Visualizes collected data with Open3D
- `transform_to_robot_table.py` - Aligns mocap data to robot coordinate frame
- `demo_clipping_3d.py` - Splits episodes into task demonstrations

### STEP1_collect_data_202408updates
Hardware updates (August 2024) for data collection using HTC Vive trackers and updated RealSense integration.

### STEP2_build_dataset
Converts raw mocap data into training datasets using inverse kinematics.
- `demo_create_hdf5.py` - Creates HDF5 datasets in robomimic format
- `pybullet_ik_bimanual.py` - PyBullet-based IK to match robot LEAP hand fingertips to human fingertips
- `dataset_utils.py` - Dataset manipulation utilities

### STEP3_train_policy
Policy training built on [robomimic](https://github.com/ARISE-Initiative/robomimic). The default training config uses Diffusion Policy with point cloud observations.
- Training entry: `python scripts/train.py --config training_config/[NAME].json`
- Located in `robomimic/` subdirectory

## Common Commands

### Environment Setup
```bash
# Windows NUC (data collection)
cd DexCap/install
conda env create -n mocap -f env_nuc_windows.yml
conda activate mocap

# Ubuntu workstation (data processing & training)
conda create -n dexcap python=3.8
conda activate dexcap
cd DexCap/install
pip install -r env_ws_requirements.txt
cd STEP3_train_policy
pip install -e .
```

### Data Collection
```bash
# Start mocap glove server
cd STEP1_collect_data
python redis_glove_server.py

# In another terminal, start recording
python data_recording.py -s --store_hand -o ./save_data_scenario_1
```

### Data Processing
```bash
# Visualize collected data
cd STEP1_collect_data
python replay_human_traj_vis.py --directory save_data_scenario_1

# Transform to robot table frame
python transform_to_robot_table.py --directory save_data_scenario_1

# Clip into task demonstrations
python demo_clipping_3d.py --directory save_data_scenario_1
```

### Dataset Building
```bash
cd STEP2_build_dataset
python demo_create_hdf5.py
```

### Policy Training
```bash
cd STEP3_train_policy/robomimic
python scripts/train.py --config training_config/diffusion_policy_pcd_packaging_1-20.json
```

## Data Format

Collected raw data is stored as:
```
save_data_scenario_1/
├── frame_0/
│   ├── color_image.jpg      # Chest camera RGB
│   ├── depth_image.png      # Chest camera depth
│   ├── pose.txt             # Chest camera 6-DoF pose
│   ├── pose_2.txt           # Left hand 6-DoF pose
│   ├── pose_3.txt           # Right hand 6-DoF pose
│   ├── left_hand_joint.txt  # Left hand joint positions
│   └── right_hand_joint.txt # Right hand joint positions
├── frame_1/
└── ...
```

## Key Dependencies

- **Rokoko Studio** - Mocap glove streaming software
- **Intel RealSense** - RGB-D camera
- **Redis** - Data streaming between processes
- **Open3D** - Point cloud visualization
- **PyBullet** - Inverse kinematics for robot hand retargeting
- **robomimic** - Robot learning framework (installed as editable)

## External Resources

- Paper: https://arxiv.org/abs/2403.07788
- Website: https://dex-cap.github.io/
- Raw dataset: https://huggingface.co/datasets/chenwangj/DexCap-Data
- Latest hardware tutorial: https://docs.google.com/document/d/1ANxSA_PctkqFf3xqAkyktgBgDWEbrFK7b1OnJe54ltw/edit