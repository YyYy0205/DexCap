# DEXCAP
> [Setup Tutorial](https://docs.google.com/document/d/1ANxSA_PctkqFf3xqAkyktgBgDWEb)

## 方案
1. DexCap
   > 提取：手掌位置, 手掌朝向（yaw）, 手指开合
2. 映射
   > (x, y, z, yaw)
3. IK（SO-101）
4. 控制机器人

## 难点
1. 工作空间不匹配
原因：人手范围 > 机器人范围
2. 抖动（非常严重）
> 解决：low-pass filter
3. IK 不稳定
> 解决：限制姿态变化/用上一步解作为初值

## 数据格式
**demo1/data/frame_0001/**
* color.png              # 彩色图像
* depth.png              # 深度图像  
* right_pose.txt         # 右肘姿态
* left_pose.txt          # 左肘姿态
* chest_pose.txt         # 胸部姿态
* raw_left_hand_joint_xyz.txt          # 左手关节位置
* raw_right_hand_joint_xyz.txt         # 右手关节位置
* raw_left_hand_joint_orientation.txt  # 左手关节朝向
* raw_right_hand_joint_orientation.txt # 右手关节朝向

**Robomimic格式 -- HDF5**
* obs
* actions
* dones

## 环境配置
### 设置streamVR 为无头模式
> 参考：https://github.com/username223/SteamVRNoHeadset 按照readme配置好即可

> 视频：https://drive.google.com/file/d/19tjjfK6J3VbHLQXgypuBamkh9hWEK3mJ/view

### Redis serve （老版本，可忽律）
1. windows 安装 redis 
2. 设置 redis port 为 6669
3. 测试 Redis
   - 打开 PowerShell（管理员）`dism /online /Enable-Feature /FeatureName:TelnetClient`
   - telnet 127.0.0.1 6669
   - ping 返回 +PONG ———> 成功

### Step 1 on NUC
1. **测试接受器连接**
   1. cd Desktop/Dexcap/STEP1_collect_data_202408updates
   2. conda activate dexcap
   3. python vive_test.py

![alt text](3d09afced766cb0537851d33e8edd694.jpg)

1. **ROKOKO 连接手套**
   1. 打开 ROKOKO
   2. 连接手套，确保两个手套连接
   3. activate streaming 流传输
      * IP: 192.168.0.200
      * Port: 14551
      * Data format: Json v3
      > 勾选 Include connection

![alt text](微信图片_20260413162428_79_623.jpg)
   > * 打开redis --- Windows （可忽略，老版本）
       1. 端口 6669   
       2. 启动 redis-server

3. **启动数据采集 --- NUC**
   > 确保连接到专门网络
   1. conda activate dexcap
   2. cd DexCap/STEP1_collect_data
   3. python redis_glove_server.py
   > 成功后会显示 "Server started，listening on port 14551"  并显示手套数据

![alt text](37ab76ddf81cc54abd87f3fac031e5e7.jpg)

### Step 2 采集数据

1. **数据采集**
   1. cd DexCap/STEP1_collect_data_202408updates
python 
   2. python vive_realsense_glove_datacollection.py NAME_OF_DEMO

2. **可视化数据**
   1. python vis_vive_realsense_glove_dataset.py NAME_OF_DEMO