import numpy as np
from activation import ReLU, Sigmoid
import tensorflow as tf

class conv:
    def __init__(self, input_channels : int = 1, filter : int = 32, kernel_size : tuple = (3, 3), stride : int = 2, padding : int = 0, activation : object = ReLU()):
        self.filter = filter
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation

        self.weights = np.random.randn(kernel_size[0], kernel_size[1], input_channels, filter) * 0.01
        self.biases = np.zeros(filter)

    def forward(self, x):
        # Add padding if specified
        if self.padding > 0:
            pad_width = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
            x = np.pad(x, pad_width, mode='constant')

        # Get dimensions
        batch_size, in_height, in_width, in_channels = x.shape
        kernel_h, kernel_w = self.kernel_size
        
        # Calculate output dimensions
        out_height = (in_height - kernel_h) // self.stride + 1
        out_width = (in_width - kernel_w) // self.stride + 1

        # Initialize output
        output = np.zeros((batch_size, out_height, out_width, self.filter))

        # Perform convolution
        for b in range(batch_size):
            for h in range(0, in_height - kernel_h + 1, self.stride):
                for w in range(0, in_width - kernel_w + 1, self.stride):
                    # Extract patch
                    patch = x[b, h:h+kernel_h, w:w+kernel_w, :]
                    
                    # Perform convolution for each filter
                    for f in range(self.filter):
                        output[b, h//self.stride, w//self.stride, f] = \
                            np.sum(patch * self.weights[:, :, :, f]) + self.biases[f]

        # Apply activation function
        if self.activation == 'relu':
            output = np.maximum(0, output)
        elif self.activation == 'sigmoid':
            output = 1 / (1 + np.exp(-output))
            
        self.output = output
        return output
    
    def backward(self, grad_output):
        batch_size, out_height, out_width, num_filters = grad_output.shape
        _, in_height, in_width, in_channels = self.input.shape
        kernel_h, kernel_w = self.kernel_size

        # Initialize gradients
        grad_input = np.zeros_like(self.input)
        grad_weights = np.zeros_like(self.weights)
        grad_biases = np.zeros(self.filter)

        # Apply activation gradient
        if self.activation == 'relu':
            grad_output = grad_output * (self.output > 0)
        elif self.activation == 'sigmoid':
            grad_output = grad_output * self.output * (1 - self.output)

        # Calculate gradients
        for b in range(batch_size):
            for h in range(0, in_height - kernel_h + 1, self.stride):
                for w in range(0, in_width - kernel_w + 1, self.stride):
                    for f in range(self.filter):
                        # Get patch
                        patch = self.input[b, h:h+kernel_h, w:w+kernel_w, :]
                        grad = grad_output[b, h//self.stride, w//self.stride, f]
                        
                        # Update gradients
                        grad_weights[:, :, :, f] += patch * grad
                        grad_input[b, h:h+kernel_h, w:w+kernel_w, :] += \
                            self.weights[:, :, :, f] * grad
                        grad_biases[f] += grad

        # Store gradients
        self.grads = [grad_weights, grad_biases]
        
        # Remove padding from gradient if necessary
        if self.padding > 0:
            grad_input = grad_input[:, self.padding:-self.padding, self.padding:-self.padding, :]
            
        return grad_input
    

class maxpool:
    def __init__(self, pool_size : tuple = (2, 2), stride : int = 2):
        self.kernel_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.input = x
        batch_size, height, width, channels = x.shape
        kernel_h, kernel_w = self.kernel_size
        
        # Calculate output dimensions
        out_height = (height - kernel_h) // self.stride + 1
        out_width = (width - kernel_w) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, out_height, out_width, channels))
        
        # Perform max pooling
        for b in range(batch_size):
            for h in range(0, height - kernel_h + 1, self.stride):
                for w in range(0, width - kernel_w + 1, self.stride):
                    for c in range(channels):
                        # Get patch and find maximum value
                        patch = x[b, h:h+kernel_h, w:w+kernel_w, c]
                        output[b, h//self.stride, w//self.stride, c] = np.max(patch)
        
        self.output = output
        return output
    
    def backward(self, grad_output):
        batch_size, height, width, channels = self.input.shape
        kernel_h, kernel_w = self.kernel_size
        
        # Initialize gradient with respect to input
        grad_input = np.zeros_like(self.input)
        
        # Calculate gradients
        for b in range(batch_size):
            for h in range(0, height - kernel_h + 1, self.stride):
                for w in range(0, width - kernel_w + 1, self.stride):
                    for c in range(channels):
                        # Get patch
                        patch = self.input[b, h:h+kernel_h, w:w+kernel_w, c]
                        
                        # Get max position
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                        
                        # Pass gradient to input that produced max
                        grad = grad_output[b, h//self.stride, w//self.stride, c]
                        grad_input[b, h+max_idx[0], w+max_idx[1], c] += grad
                        
        return grad_input
    

class flatten:
    def __init__(self):
        pass

    def forward(self, x):
        self.input = x
        return x.reshape(x.shape[0], -1)
    
    def backward(self, grad_output):
        return tf.reshape(grad_output, self.input.shape)
    

class dense:
    def __init__(self, input_size : int, output_size : int, activation : object = ReLU()):
        # He initialization
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0/input_size)
        self.bias = np.zeros((output_size, 1))
        self.activation = activation

    def forward(self, x):
        self.Z = np.dot(self.weights, x) + self.bias
        self.A = self.activation.function(self.Z)
        return self.A
    
    def backward(self, dA, A_p):
        m = A_p.shape[0]  # batch size
        
        # Ensure shapes are compatible
        if len(dA.shape) == 1:
            dA = tf.expand_dims(dA, axis=1)  # Add dimension to make it [batch_size, 1]
        
        # Ensure A_p is properly shaped
        if len(A_p.shape) > 2:
            A_p = tf.reshape(A_p, [m, -1])  # Flatten to [batch_size, features]
            
        dZ = dA * self.activation.prime_function(self.Z)
        dW = (1/m) * tf.matmul(dZ, tf.transpose(A_p))
        db = (1/m) * tf.reduce_sum(dZ, axis=0, keepdims=True)
        
        return dZ, dW, db