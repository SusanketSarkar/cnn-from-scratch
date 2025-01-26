import numpy as np

class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, clip_value=5.0):
        super().__init__(learning_rate)
        self.clip_value = clip_value

    def update(self, layer, dW, db):
        # Gradient clipping
        dW = np.clip(dW, -self.clip_value, self.clip_value)
        db = np.clip(db, -self.clip_value, self.clip_value)
        
        layer.weights -= self.learning_rate * dW
        layer.bias -= self.learning_rate * db