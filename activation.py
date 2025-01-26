import numpy as np

class ReLU:
    def __init__(self):
        pass

    def function(self, x):
        return np.maximum(0, x)

    def prime_function(self, x):
        return np.where(x > 0, 1, 0)
    

class Sigmoid:
    def __init__(self):
        pass

    def function(self, x):
        return 1 / (1 + np.exp(-x))
    
    def prime_function(self, x):
        sigmoid = self.function(x)
        return sigmoid * (1 - sigmoid)
    
