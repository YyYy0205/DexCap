# CLAUDE.md

本文档为 Claude Code (claude.ai/code) 在本项目中工作时提供指导。

## 项目概述

DexCap 是一套可扩展的便携式动作捕捉（mocap）数据采集系统，用于灵巧操作。该系统使用 Rokoko 动作捕捉手套和 Intel RealSense 相机捕获人类手部动作，然后处理数据并通过模仿学习训练机器人策略。

## 架构

代码库按三个顺序流程步骤组织：

### STEP1_collect_data
使用 Rokoko 动作捕捉手套和 RealSense 相机进行数据采集。在 Windows 迷你PC（NUC）上运行，连接到便携式 Wi-Fi 路由器。
- `redis_glove_server.py` - 通过 Redis 流式传输动作捕捉手套数据
- `data_recording.py` - 记录同步的相机 + 动作捕捉数据
- `replay_human_traj_vis.py` - 使用 Open3D 可视化采集的数据
- `transform_to_robot_table.py` - 将动作捕捉数据对齐到机器人坐标系
- `demo_clipping_3d.py` - 将片段拆分为任务演示

### STEP1_collect_data_202408updates
2024年8月的硬件更新，使用 HTC Vive 追踪器和更新的 RealSense 集成。

### STEP2_build_dataset
使用逆运动学将原始动作捕捉数据转换为训练数据集。
- `demo_create_hdf5.py` - 创建 robomimic 格式的 HDF5 数据集
- `pybullet_ik_bimanual.py` - 基于 PyBullet 的逆运动学，将机器人 LEAP 手指尖端与人类手指尖端匹配
- `dataset_utils.py` - 数据集操作工具

### STEP3_train_policy
基于 [robomimic](https://github.com/ARISE-Initiative/robomimic) 的策略训练。默认训练配置使用 Diffusion Policy 和点云观测。
- 训练入口：`python scripts/train.py --config training_config/[名称].json`
- 位于 `robomimic/` 子目录

## 详细操作流程

### 第一步：环境配置

#### Windows NUC（数据采集端）
```bash
cd DexCap/install
conda env create -n mocap -f env_nuc_windows.yml
conda activate mocap
```

#### Ubuntu 工作站（数据处理与训练端）
```bash
conda create -n dexcap python=3.8
conda activate dexcap
cd DexCap/install
pip install -r env_ws_requirements.txt
cd STEP3_train_policy
pip install -e .
```

---

### 第二步：数据采集

#### 1. 启动 Rokoko Studio
- 打开 Rokoko Studio 软件
- 确保动作捕捉手套已被检测到
- 选择 `Livestreaming` 功能
- 使用 `Custom connection` 配置：
  - Include connection: True
  - Forward IP: 192.168.0.200
  - Port: 14551
  - Data format: Json

#### 2. 配置网络
- 将 NUC 连接到便携式 Wi-Fi 路由器
- 设置 IP 地址为 `192.168.0.200`

#### 3. 启动手套数据服务器
```bash
conda activate mocap
cd DexCap/STEP1_collect_data
python redis_glove_server.py
```

#### 4. 开始录制
在另一个终端中执行：
```bash
conda activate mocap
cd DexCap/STEP1_collect_data
python data_recording.py -s --store_hand -o ./save_data_scenario_1
```

- 数据会先存储在内存中
- 按 `Ctrl+C` 停止录制
- 程序会自动将数据保存到本地 SSD（多线程方式）

---

### 第三步：数据处理与可视化

#### 1. 可视化采集的数据
```bash
cd DexCap/STEP1_collect_data
python replay_human_traj_vis.py --directory save_data_scenario_1
```
- 使用 Open3D 点云可视化器显示
- 可看到捕获的手部动作

#### 2. 校准（可选）
如果需要修正 SLAM 初始漂移：
```bash
python replay_human_traj_vis.py --directory save_data_scenario_1 --calib
python calculate_offset_vis_calib.py --directory save_data_scenario_1
```
- 使用数字键盘进行校正
- 校正将应用于整个视频

#### 3. 转换到机器人工作台坐标系
```bash
python transform_to_robot_table.py --directory save_data_scenario_1
```
- 使用数字键盘调整数据的世界坐标系
- 使其与机器人工作台坐标系对齐
- 此过程通常只需不到 10 秒，每个数据片段只需执行一次

#### 4. 剪辑任务演示
```bash
python demo_clipping_3d.py --directory save_data_scenario_1
```
- 将整个片段拆分为多个任务演示

---

### 第四步：构建训练数据集

#### 1. 传输数据
将处理后的数据从 NUC 传输到 Ubuntu 工作站

#### 2. 生成 HDF5 数据集
```bash
cd DexCap/STEP2_build_dataset
python demo_create_hdf5.py
```

此过程：
- 使用基于 PyBullet 的逆运动学
- 将机器人 LEAP 手的指尖与人类的指尖匹配
- 当人类手在相机视野中时，将机器人手的点云网格添加到点云观测中
- 移除多余点云（背景、桌面等）

---

### 第五步：策略训练

#### 1. 开始训练
```bash
cd DexCap/STEP3_train_policy/robomimic
python scripts/train.py --config training_config/diffusion_policy_pcd_packaging_1-20.json
```

#### 2. 训练配置
- 默认配置训练基于点云的 Diffusion Policy
- 输入：胸挂相机的点云观测（转换到固定世界坐标系）
- 输出：20 步动作序列，包含双手和双臂（总共 46 维）

#### 3. 可用训练配置
- `training_config/diffusion_policy_pcd_packaging_1-20.json` - 包装任务
- `training_config/diffusion_policy_pcd_wiping_1-14.json` - 擦拭任务

---

## 数据格式

采集的原始数据存储结构如下：
```
save_data_scenario_1/
├── frame_0/
│   ├── color_image.jpg      # 胸挂相机 RGB 图像
│   ├── depth_image.png      # 胸挂相机深度图像
│   ├── pose.txt             # 胸挂相机 6-DoF 位姿（世界坐标系）
│   ├── pose_2.txt           # 左手 6-DoF 位姿（世界坐标系）
│   ├── pose_3.txt           # 右手 6-DoF 位姿（世界坐标系）
│   ├── left_hand_joint.txt  # 左手关节位置（3D，掌心坐标系）
│   └── right_hand_joint.txt # 右手关节位置（3D，掌心坐标系）
├── frame_1/
└── ...
```

---

## 关键依赖

- **Rokoko Studio** - 动作捕捉手套流式传输软件
- **Intel RealSense** - RGB-D 相机
- **Redis** - 进程间数据流传输
- **Open3D** - 点云可视化
- **PyBullet** - 机器人手部重定向逆运动学
- **robomimic** - 机器人学习框架（可编辑模式安装）

---

## 外部资源

- 论文：https://arxiv.org/abs/2403.07788
- 项目网站：https://dex-cap.github.io/
- 原始数据集：https://huggingface.co/datasets/chenwangj/DexCap-Data
- 最新硬件教程：https://docs.google.com/document/d/1ANxSA_PctkqFf3xqAkyktgBgDWEbrFK7b1OnJe54ltw/edit

---

## 致谢

- 策略训练基于 [robomimic](https://github.com/ARISE-Initiative/robomimic) 和 [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- 机械臂控制器基于 [Deoxys](https://github.com/UT-Austin-RPL/deoxys_control)
- 机器人 LEAP 手控制器基于 [LEAP_Hand_API](https://github.com/leap-hand/LEAP_Hand_API)