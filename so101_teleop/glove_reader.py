import numpy as np
import redis


class GloveReader:
    """
    从 Redis 读取 Rokoko 手套实时数据。
    redis_glove_server.py 负责接收 UDP 并写入 Redis。
    """

    _KEY_R_XYZ  = "rawRightHandJointXyz"
    _KEY_L_XYZ  = "rawLeftHandJointXyz"
    _KEY_R_ORI  = "rawRightHandJointOrientation"
    _KEY_L_ORI  = "rawLeftHandJointOrientation"

    def __init__(self, host: str = "localhost", port: int = 6669, password: str = ""):
        self._r = redis.StrictRedis(host=host, port=port, password=password,
                                    decode_responses=False)
        self._last_rj = np.zeros((21, 3), dtype=np.float32)
        self._last_lj = np.zeros((21, 3), dtype=np.float32)

    def read(self):
        """
        返回 (right_joints, left_joints)，各为 (21, 3) float32。
        Redis 读取失败时返回上一帧数据（避免控制中断）。
        """
        try:
            rj = np.frombuffer(self._r.get(self._KEY_R_XYZ),
                               dtype=np.float64).reshape(21, 3).astype(np.float32)
            lj = np.frombuffer(self._r.get(self._KEY_L_XYZ),
                               dtype=np.float64).reshape(21, 3).astype(np.float32)
            self._last_rj = rj
            self._last_lj = lj
        except Exception:
            pass
        return self._last_rj, self._last_lj

    def is_available(self) -> bool:
        """测试 Redis 连通性。"""
        try:
            return self._r.ping()
        except Exception:
            return False
