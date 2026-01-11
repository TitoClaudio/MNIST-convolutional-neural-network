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

**Convolutional Layer** ([conv.py](src/layers/conv.py))
- 8 filters with 3×3 kernels
- Extracts spatial features like edges, curves, and patterns
- Uses Xavier initialization for weights
- Forward and backward propagation implemented

**Max Pooling Layer** ([maxpool.py](src/layers/maxpool.py))
- 2×2 pooling window with stride 2
- Reduces spatial dimensions by half
- Provides translation invariance
- Implements gradient routing in backpropagation

**Softmax Layer** ([softmax.py](src/layers/softmax.py))
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
- Training set: 4,000 images (subset of 60,000)
- Test set: 200 images (subset of 10,000)
- Image format: 28×28 grayscale pixels
- Normalization: Pixel values scaled to [-0.5, 0.5]

## Installation

1. Install the project in editable mode (this allows clean imports of the `src` package):
```bash
pip install -e .
```

This will install the project along with all dependencies:
- torch>=2.9.0
- torchvision>=0.24.0
- numpy>=2.4.0
- fastapi[standard]>=0.128.0
- pillow>=12.0.0

## Usage

### Training the Model

Run the training script:
```bash
python scripts/train.py
```

The script will:
1. Load and preprocess the MNIST dataset
2. Train for 3 epochs with learning rate 0.005
3. Display training progress every 100 steps
4. Evaluate on the test set and print final accuracy
5. Save the trained model to `trained_cnn_model.npz`

#### Training Output Example

```
MNIST CNN initialized! Running test set...
--- Epoch 1 ---
[Step 100] Past 100 steps: Average Loss 2.302 | Accuracy: 18%
[Step 200] Past 100 steps: Average Loss 1.821 | Accuracy: 32%
...
--- Epoch 3 ---
[Step 1900] Past 100 steps: Average Loss 0.621 | Accuracy: 78%
[Step 2000] Past 100 steps: Average Loss 0.584 | Accuracy: 82%

--- Testing ---
Test Loss: 0.584
Test Accuracy: 0.86

--- Saving Model ---
✓ Model saved to: trained_cnn_model.npz
  File size: 54.23 KB
  Parameters saved:
    - Conv filters: (8, 3, 3)
    - Softmax weights: (1352, 10)
    - Softmax biases: (10,)
  Metadata: {'test_accuracy': 0.875, 'test_loss': 0.421, ...}
```

### Making Predictions

Use the trained model to predict digits from custom images:

```python
from src.predict import predict_from_file

# Predict a digit from an image file
result = predict_from_file('path/to/image.png')

print(f"Predicted digit: {result['digit']}")
print(f"Confidence: {result['confidence']:.2%}")
```

The prediction module automatically:
- Loads and preprocesses images (resize to 28×28, grayscale conversion)
- Inverts colors if needed (black-on-white → white-on-black)
- Normalizes pixel values to match training data
- Returns prediction with confidence scores

#### Image Requirements

- **Format**: PNG, JPG, or any common image format
- **Content**: Single handwritten digit (0-9)
- **Background**: Any color (auto-inverted if needed)
- The image will be automatically resized to 28×28 pixels

### API Endpoints

The project includes a FastAPI server ([main.py](src/api/main.py)) with REST endpoints:

**GET /** - Health check and model information

**GET /model/info** - Returns model architecture details and training metadata

**POST /predict** - Upload an image file to predict the digit
- Request: multipart/form-data with image file
- Response: predicted digit, confidence score, and probability distribution

**POST /predict/base64** - Predict from base64 encoded image
- Request: JSON with `{"image": "base64_string"}`
- Response: predicted digit, confidence score, and probability distribution

Run the server:
```bash
# Development (with auto-reload)
fastapi dev src/api/main.py

# Production
fastapi run src/api/main.py
```

Interactive API documentation available at `/docs` endpoint.

## Deployment

The API is deployed on Google Cloud Run and can be accessed at your Cloud Run URL.

### Docker

The project includes a Dockerfile for containerization:

```bash
# Build Docker image locally
docker build -t cnn-digit-recognition .

# Run container locally
docker run -p 8080:8080 cnn-digit-recognition
```

### Google Cloud Run Deployment

To deploy to Google Cloud Run:

```bash
# Build and push image to Google Container Registry
gcloud builds submit --tag gcr.io/cnn-digit-recognition/cnn-api

# Deploy to Cloud Run
gcloud run deploy cnn-api \
  --image gcr.io/cnn-digit-recognition/cnn-api \
  --platform managed \
  --region southamerica-east1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

### Updating the Deployed Model

To update the model in production:

1. Train a new model: `python scripts/train.py`
2. Rebuild the image: `gcloud builds submit --tag gcr.io/cnn-digit-recognition/cnn-api`
3. Redeploy: `gcloud run deploy cnn-api --image gcr.io/cnn-digit-recognition/cnn-api ...`

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

## Model Persistence

The project includes utilities for saving and loading trained models.

### Saving Models

Models are automatically saved after training with metadata:

```python
from src.model_utils import save_model

save_model(
    conv.filters,
    softmax.weights,
    softmax.biases,
    filepath='my_model.npz',
    metadata={
        'test_accuracy': 0.875,
        'epochs': 3,
        'learning_rate': 0.005
    }
)
```

### Loading Models

Load a trained model for inference:

```python
from src.model_utils import load_model

# Load model parameters
model_data = load_model('trained_cnn_model.npz')

# Access parameters
conv_filters = model_data['conv_filters']
softmax_weights = model_data['softmax_weights']
softmax_biases = model_data['softmax_biases']
metadata = model_data['metadata']
```

The model file format (`.npz`) is:
- **Compressed**: Uses NumPy's compressed format for small file size (~54 KB)
- **Portable**: Works across different systems and Python versions
- **Complete**: Includes all learned parameters and training metadata

## Project Structure

```
CNN/
├── src/
│   ├── layers/              # Neural network layers
│   │   ├── conv.py          # Convolutional layer
│   │   ├── maxpool.py       # Max pooling layer
│   │   └── softmax.py       # Softmax layer
│   ├── utils/               # Utility modules
│   │   ├── model_utils.py   # Model save/load
│   │   └── predict.py       # Inference utilities
│   └── api/                 # FastAPI application
│       └── main.py          # API endpoints
├── scripts/
│   └── train.py             # Training script
├── config.py                # Configuration settings
├── data/                    # MNIST dataset (auto-downloaded)
├── trained_cnn_model.npz    # Saved model (generated)
├── pyproject.toml           # Package configuration
├── requirements.txt         # Dependencies
├── Dockerfile               # Docker container configuration
├── .dockerignore           # Docker build exclusions
├── .gcloudignore           # Cloud Build exclusions
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
