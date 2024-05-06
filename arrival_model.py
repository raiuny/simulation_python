import numpy
from enum import Enum
from abc import ABC, abstractmethod

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
    def __init__(self, a: int):
        self.a = a
        
    def get_itv(self) -> int:
        return a
    
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