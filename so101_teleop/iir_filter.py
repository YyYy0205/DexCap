import numpy as np
from scipy.signal import butter, lfilter_zi, lfilter


class IIRFilter:
    """
    单步 Butterworth 低通滤波器，保持跨帧状态。
    与 data_loader.lowpass (批处理 filtfilt) 等效，但适用于实时逐帧输入。
    """

    def __init__(self, cutoff_hz: float, sample_hz: float, order: int = 2, n_dim: int = 3):
        nyq = sample_hz / 2.0
        self.b, self.a = butter(order, cutoff_hz / nyq, btype="low")
        zi_1d = lfilter_zi(self.b, self.a)          # (order,)
        self.zi = np.zeros((len(zi_1d), n_dim))     # (order, n_dim)
        self.n_dim = n_dim
        self._init = False

    def step(self, x: np.ndarray) -> np.ndarray:
        """
        x: (n_dim,) 当前帧原始值
        返回: (n_dim,) 滤波后的值
        """
        x = np.asarray(x, dtype=float)
        if not self._init:
            zi_1d = lfilter_zi(self.b, self.a)
            # 用第一帧值初始化 zi，避免启动阶跃响应
            self.zi = np.outer(zi_1d, x)
            self._init = True

        y = np.empty(self.n_dim)
        for i in range(self.n_dim):
            out, self.zi[:, i] = lfilter(self.b, self.a, [x[i]], zi=self.zi[:, i])
            y[i] = out[0]
        return y

    def reset(self):
        self.zi[:] = 0.0
        self._init = False
