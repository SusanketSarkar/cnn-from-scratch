from layers import conv, maxpool, flatten, dense
import numpy as np
class CNN:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss

    def summary(self):
        for layer in self.layers:
            print(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, x, y):
        # Get output gradient from loss function
        dA = self.loss.binary_crossentropy_derivative(y, np.argmax(self.layers[-1].A, axis = 1))
        
        # Store previous activation for dense layer backward pass
        # Skip flatten layers when looking for previous activation
        prev_layer_idx = -1
        while isinstance(self.layers[prev_layer_idx], flatten) or isinstance(self.layers[prev_layer_idx], maxpool):
            prev_layer_idx -= 1
        A_prev = self.layers[prev_layer_idx].A if len(self.layers) > 1 else x
        
        # Iterate through layers backwards
        for i in range(len(self.layers)-1, -1, -1):
            layer = self.layers[i]
            
            # Handle different layer types
            if isinstance(layer, dense):
                # Dense layer backward pass
                dA, dW, db = layer.backward(dA, A_prev)
                # Update weights using optimizer
                self.optimizer.update(layer, dW, db)
                # Update A_prev for next iteration
                A_prev = self.layers[i-1].A if i > 0 else x
                
            elif isinstance(layer, conv):
                # Convolutional layer backward pass
                dA = layer.backward(dA)
                # Update weights using optimizer
                self.optimizer.update(layer, layer.grads[0], layer.grads[1])
                
            elif isinstance(layer, (maxpool, flatten)):
                # Maxpool and flatten layers just pass gradient back
                dA = layer.backward(dA)

    def fit(self, train_ds, epochs):
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for images, labels in train_ds:
                # Forward pass
                y_pred = self.forward(images)
                
                # Calculate loss
                y_pred = np.argmax(y_pred, axis = 1)
                batch_loss = self.loss.binary_crossentropy(labels, y_pred)
                epoch_loss += batch_loss
                batch_count += 1
                
                # Backward pass
                self.backward(images, labels)
            
            # Calculate average loss for epoch
            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")



    def evaluate(self, test_ds):
        predictions = []
        for images, _ in test_ds:
            # Forward pass through the network
            x = images
            for layer in self.layers:
                x = layer.forward(x)
            predictions.append(x)
            
        return predictions