# DEXCAP

## 硬件安装

### Redis serve
1. windows 安装 redis 
2. 设置 redis port 为 6669
3. 测试 Redis
   - 打开 PowerShell（管理员）`dism /online /Enable-Feature /FeatureName:TelnetClient`
   - telnet 127.0.0.1 6669
   - ping 返回 +PONG ———> 成功

### Step 1 on NUC
* 测试接受器连接
   1. cd Desktop/Dexcap/STEP1_collect_data_202408updates
   2. conda activate dexcap
   3. python vive_test.py

* ROKOKO 连接手套
   1. 打开 ROKOKO
   2. 连接手套
   3. activate streaming
      * IP: 192.168.0.200
      * Port: 14551
      * Data format: Json v3
      > 勾选 Include connection

* 打开redis
   1. 端口 6669   
   2. 启动 redis-server

* 启动数据采集
   > 确保连接到专门网络
   1. cd DexCap/STEP1_collect_data
   2. python redis_glove_server.py
   > 成功后会显示 "Server started，listening on port 14551"

* 


