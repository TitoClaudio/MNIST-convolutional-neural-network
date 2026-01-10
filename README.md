# Convolutional Neural Network for Digit Recognition

A from-scratch implementation of a Convolutional Neural Network (CNN) built with NumPy for recognizing handwritten digits from the MNIST dataset.

## Project Overview

This project implements a complete CNN architecture without relying on high-level deep learning frameworks like TensorFlow or PyTorch (except for dataset loading). The network is trained to classify handwritten digits (0-9) with backpropagation implemented manually for each layer.

## Architecture

The network consists of three main layers:

```
Input (28×28 grayscale image)
    ↓
Convolutional Layer (8 filters, 3×3 kernel)
    ↓ Output: 26×26×8
Max Pooling Layer (2×2 pool size)
    ↓ Output: 13×13×8
Softmax Layer (Fully Connected)
    ↓ Output: 10 classes (digits 0-9)
```

### Layer Details

**Convolutional Layer** ([conv.py](src/conv.py))
- 8 filters with 3×3 kernels
- Extracts spatial features like edges, curves, and patterns
- Uses Xavier initialization for weights
- Forward and backward propagation implemented

**Max Pooling Layer** ([maxpool.py](src/maxpool.py))
- 2×2 pooling window with stride 2
- Reduces spatial dimensions by half
- Provides translation invariance
- Implements gradient routing in backpropagation

**Softmax Layer** ([softmax.py](src/softmax.py))
- Fully connected layer with 1,352 input features (flattened 13×13×8)
- 10 output nodes (one per digit class)
- Converts raw scores to probability distribution
- Implements cross-entropy loss

## Features

- **Built from scratch**: All layers implemented using only NumPy
- **Complete backpropagation**: Gradients computed manually for all layers
- **Training pipeline**: Includes data normalization, shuffling, and epoch-based training
- **Test evaluation**: Separate test set for model validation
- **Modular design**: Each layer is a standalone class with forward and backward methods

## Dataset

Uses the MNIST dataset:
- Training set: 2,000 images (subset of 60,000)
- Test set: 200 images (subset of 10,000)
- Image format: 28×28 grayscale pixels
- Normalization: Pixel values scaled to [-0.5, 0.5]

## Requirements

```
torch>=2.9.0
torchvision>=0.24.0
numpy>=2.4.0
```

Install dependencies:
```bash
pip install torch torchvision numpy
```

## Usage

Run the training and evaluation:
```bash
python src/cnn.py
```

The script will:
1. Load and preprocess the MNIST dataset
2. Train for 3 epochs with learning rate 0.005
3. Display training progress every 100 steps
4. Evaluate on the test set and print final accuracy

### Example Output

```
MNIST CNN initialized! Running test set...
--- Epoch 1 ---
[Step 100] Past 100 steps: Average Loss 2.302 | Accuracy: 18%
[Step 200] Past 100 steps: Average Loss 1.821 | Accuracy: 32%
...
--- Testing ---
Test Loss: 0.584
Test Accuracy: 0.86
```

## Implementation Details

### Forward Pass

Each layer transforms the input sequentially:

1. **Convolution**: Applies 8 learned filters to detect features
2. **Max Pooling**: Downsamples by taking maximum values in 2×2 regions
3. **Softmax**: Flattens and computes class probabilities

### Backward Pass

Gradients flow backward through the network:

1. **Loss gradient**: Computed from cross-entropy loss
2. **Softmax backprop**: Updates weights and biases, passes gradient to pooling
3. **MaxPool backprop**: Routes gradients only to max values
4. **Conv backprop**: Updates filter weights based on feature gradients

### Key Algorithms

**Xavier Initialization**:
```python
filters = np.random.randn(num_filters, 3, 3) / 9
weights = np.random.randn(input_len, nodes) / input_len
```

**Softmax Function**:
```python
exp = np.exp(totals)
probabilities = exp / np.sum(exp)
```

**Cross-Entropy Loss**:
```python
loss = -np.log(output[true_label])
```

## Project Structure

```
CNN/
├── src/
│   ├── cnn.py          # Main training script
│   ├── conv.py         # Convolutional layer implementation
│   ├── maxpool.py      # Max pooling layer implementation
│   └── softmax.py      # Softmax layer implementation
├── data/               # MNIST dataset (downloaded automatically)
└── README.md
```

## Learning Objectives

This project demonstrates understanding of:
- Convolutional neural network architecture
- Forward and backward propagation
- Gradient descent optimization
- NumPy array operations and broadcasting
- Image classification fundamentals
- Deep learning mathematics (chain rule, softmax derivative, etc.)

## References

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- NumPy Documentation

## License

This project is open source and available for educational purposes.
