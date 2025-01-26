# CNN Implementation from Scratch using TensorFlow

This project implements a Convolutional Neural Network (CNN) framework from scratch using TensorFlow's low-level operations. While frameworks like Keras provide high-level APIs for neural networks, this implementation helps understand the internal workings of CNNs by building each component manually.

## Key Features

- **Modular Layer Architecture**: Includes implementations of:
  - Convolutional layers (Conv2D)
  - Max Pooling layers
  - Flatten layers
  - Dense (Fully Connected) layers

- **Forward and Backward Propagation**: Custom implementations of both forward pass and backpropagation for each layer type

- **Optimization**: Gradient descent optimization with customizable learning rates

- **Training Pipeline**: Complete training loop with batch processing capability

## Technical Details

The implementation uses TensorFlow's tensor operations while maintaining the mathematical structure of a CNN. Key components include:

- Convolution operations with learnable filters
- Max pooling for spatial dimensionality reduction
- Flattening of 3D feature maps to 1D vectors
- Dense layers for final classification
- Backpropagation through each layer type

## Educational Value

This project is particularly valuable for:
- Understanding the mathematics behind CNNs
- Learning how backpropagation works in convolutional networks
- Gaining insights into tensor operations and shapes
- Appreciating the complexity of deep learning frameworks

## Usage

The framework allows you to build CNNs by stacking layers:

```python
model = CNN([
    conv(filters=16, kernel_size=3),
    maxpool(pool_size=2),
    flatten(),
    dense(units=10)
])

model.fit(train_ds, epochs=10)
```
