# DEXCAP

> [Setup Tutorial](https://docs.google.com/document/d/1ANxSA_PctkqFf3xqAkyktgBgDWEb)

## 方案

1. **DexCap**
   > 提取：手掌位置, 手掌朝向（yaw）, 手指开合

2. **映射**
   > (x, y, z, yaw)

3. **IK**（SO-101）

4. **控制机器人**

## 难点

1. **工作空间不匹配**
   - 原因：人手范围 > 机器人范围

2. **抖动（非常严重）**
   > 解决：low-pass filter

3. **IK 不稳定**
   > 解决：限制姿态变化/用上一步解作为初值

## 数据格式

### 原始数据格式

**路径**：`demo1/data/frame_0001/`

```
├── frame_0
│   ├── color_image.jpg      # Chest camera RGB image
│   ├── depth_image.png      # Chest camera depth image
│   ├── pose.txt             # Chest camera 6-DoF pose in world frame
│   ├── pose_2.txt           # Left hand 6-DoF pose in world frame
│   ├── pose_3.txt           # Right hand 6-DoF pose in world frame
│   ├── left_hand_joint.txt  # Left hand joint positions (3D) in the palm frame
│   └── right_hand_joint.txt # Right hand joint positions (3D) in the palm frame
├── frame_1
└── ...
```

### Robomimic 格式（HDF5）

- `obs` - 观测数据
- `actions` - 动作数据
- `dones` - 结束标记

## 环境配置

### 设置 SteamVR 为无头模式

> 参考：https://github.com/username223/SteamVRNoHeadset
> 按照 README 配置好即可

> 视频：https://drive.google.com/file/d/19tjjfK6J3VbHLQXgypuBamkh9hWEK3mJ/view

### Redis Server（老版本，可忽略）

1. Windows 安装 Redis
2. 设置 Redis port 为 6669
3. 测试 Redis
   - 打开 PowerShell（管理员）：
     ```powershell
     dism /online /Enable-Feature /FeatureName:TelnetClient
     ```
   - 执行 telnet：
     ```bash
     telnet 127.0.0.1 6669
     ```
   - ping 返回 `+PONG` → 成功

### Step 1 on NUC

#### 1. 测试接收器连接

```bash
cd Desktop/Dexcap/STEP1_collect_data_202408updates
conda activate dexcap
python vive_test.py
```

![alt text](3d09afced766cb0537851d33e8edd694.jpg)

#### 2. ROKOKO 连接手套

1. 打开 ROKOKO
2. 连接手套，确保两个手套都连接成功
3. Activate Streaming（流传输）：
   - **IP**: `192.168.0.200`
   - **Port**: `14551`
   - **Data format**: `Json v3`
   > 勾选 **Include connection**

![alt text](微信图片_20260413162428_79_623.jpg)

> **Redis（老版本，可忽略）**
> - 端口：6669
> - 启动：`redis-server`

#### 3. 启动数据采集（NUC）

> 确保连接到专用网络

```bash
conda activate dexcap
cd DexCap/STEP1_collect_data
python redis_glove_server.py
```

> 成功后会显示：
> ```
> Server started, listening on port 14551
> ```
> 并显示手套数据

![alt text](37ab76ddf81cc54abd87f3fac031e5e7.jpg)

### Step 2 采集数据

#### 1. 采集数据（无 Tracker）

```bash
cd DexCap/STEP1_collect_data
python data_recording.py -s --store_hand -o ./data_test
```

![采数据](image.png)

* 数据格式：
```
data_test/
├── frame_0/
├── frame_1/
├── frame_2/
└── ...
      frame_x/
      ├── color_image.jpg            # RGB 彩色图像
      ├── depth_image.png            # 深度图像： 用于生成3d点云
      ├── left_hand_joint.txt        # 左手 21*3 个关节的 XYZ 坐标
      ├── right_hand_joint.txt       # 右手 21*3 个关节的 XYZ 坐标
      ├── left_hand_joint_ori.txt    # 左手 21*4 个关节的四元数旋转
      └── right_hand_joint_ori.txt   # 右手 21*4 个关节的四元数旋转
```
* 但是缺少：手腕位姿
> 但也可以转换为 HDF5（训练格式）用于手部抓取训练

* 可视化数据
```bash
python playback_dataset.py -i ./data_test

python playback_dataset.py -i ./data_test --fps 15
```

#### 2. 数据采集（带 Tracker）

```bash
cd DexCap/STEP1_collect_data_202408updates
python vive_realsense_glove_datacollection.py NAME_OF_DEMO
```

#### 3. 可视化数据

```bash
python vis_vive_realsense_glove_dataset.py NAME_OF_DEMO
```
![可视化数据](image-1.png)