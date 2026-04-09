# DEXCAP

## 硬件安装

### Redis serve
1. windows 安装 redis 
2. 设置 redis port 为 6669
3. 测试 Redis
   - 打开 PowerShell（管理员）`dism /online /Enable-Feature /FeatureName:TelnetClient`
   - telnet 127.0.0.1 6669
   - ping 返回 +PONG ———> 成功
4. 
