import numpy as np


class Constant:
    
    def __init__(self, value: float):
        self.value = value
        
    def __call__(self, current_epoch: int, max_epochs: int) -> float:
        return self.value
    
    
class Polynomial:
    
    def __init__(self, initial: float, final: float, power: float):
        self.initial = initial
        self.final = final
        self.power = power
        
    def __call__(self, current_epoch: int, max_epochs: int) -> float:
        current_rate = (self.initial - self.final) * (1 - (current_epoch / max_epochs))**self.power + self.final
        return current_rate
    
    
class Exponential:
    
    def __init__(self, initial: float, final: None | float = None, factor: None | float = None):
        
        assert not (final is None and factor is None), "final or factor should be set"
        assert not (final is not None and factor is not None), "only one of final or factor should be set"
        
        self.initial = initial
        self.final = final
        self.factor = factor
        
    def __call__(self, current_epoch: int, max_epochs: int) -> float:
        
        # Initialise once when we know max_epochs
        if self.factor is None:
            self.factor = - (np.log(self.final / self.initial)) / max_epochs
        
        current_rate = self.initial * np.exp(-self.factor * current_epoch)
        return current_rate