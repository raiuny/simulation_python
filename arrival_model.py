import numpy
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np

class ArrivalType(Enum):
    POSSION = 1
    BERNOULLI = 2
    DETERMINISTIC = 3
    UNIFORM = 4


class IntervalGenerator(ABC):
    @abstractmethod
    def get_itv(self):
        pass
        
class IntervalGeneratorFactory(object):
    @staticmethod
    def create(arr_type: ArrivalType, **kwargs) -> IntervalGenerator:
        if arr_type == ArrivalType.POSSION:
            return PossionGenerator(**kwargs)
        if arr_type == ArrivalType.BERNOULLI:
            return BernoulliGenerator(**kwargs)
        if arr_type == ArrivalType.DETERMINISTIC:
            return DeterministicGenerator(**kwargs)
        if arr_type == ArrivalType.UNIFORM:
            return UniformGenerator(**kwargs)

class PossionGenerator(IntervalGenerator):
    def __init__(self, lam):
        self.rng = numpy.random.default_rng()
        self.lam = lam
        
    def get_itv(self) -> int:
        return self.rng.geometric(self.lam)

class BernoulliGenerator(IntervalGenerator):
    def __init__(self, lam):
        self.rng = numpy.random.default_rng()
        self.lam = lam
        
    def get_itv(self) -> int:
        return self.rng.geometric(self.lam)
    
class DeterministicGenerator(IntervalGenerator):
    def __init__(self):
        fps = 72
        bps = 150
        u_interval = 1 / fps
        b_interval = 0.00065
        slot_duration = 9e-6  # 1 slot = 9微秒 = 9×10^-6秒
        self.u_slot = u_interval / slot_duration  # 单位转换,1543 slots
        self.b_slot = b_interval / slot_duration

        self.u_frame_length = (1.02 * bps + 0.28) * 1e6 / (8 * fps)  # 数据包大小的随机分布, 平均177个数据包
        self.b_frame_length = 124.9 * bps + 3080.6
        
    def get_itv(self) -> int:
        interval = max(1, np.random.laplace(self.u_slot, self.b_slot))
        interval = int(np.ceil(interval))  # 向上取整
        # print(interval)
        return interval
    
    def get_frame_length(self):
        frame_length = max(1, int(np.random.laplace(self.u_frame_length, self.b_frame_length)))
        return frame_length
    
class UniformGenerator(IntervalGenerator):
    def __init__(self, low: float, high: float):
        self.rng = numpy.random.RandomState()
        self.low = low
        self.high = high
        
    def get_itv(self) -> int:
        return self.rng.randint(self.low, self.high)

    
if __name__ == "__main__":
    a = IntervalGeneratorFactory.create(ArrivalType.POSSION, lam=10)
    print(a.get_itv())