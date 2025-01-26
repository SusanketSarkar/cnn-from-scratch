import numpy as np
import tensorflow as tf

class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, clip_value=5.0):
        super().__init__(learning_rate)
        self.clip_value = clip_value

    def update(self, layer, dW, db):
        # Convert tensors to numpy if needed
        if tf.is_tensor(dW):
            dW = dW.numpy()
        if tf.is_tensor(db):
            db = db.numpy()
            
        # Clip gradients
        dW = np.clip(dW, -self.clip_value, self.clip_value)
        db = np.clip(db, -self.clip_value, self.clip_value)
        
        # Ensure db has correct shape for broadcasting
        if len(db.shape) > 1:
            db = np.mean(db, axis=1, keepdims=True)
        
        # Update weights and biases
        layer.weights -= self.learning_rate * dW
        layer.bias = layer.bias.reshape(-1, 1) - self.learning_rate * db.reshape(-1, 1)