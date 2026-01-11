import numpy as np
from PIL import Image
from conv import Conv3x3
from maxpool import MaxPool
from softmax import Softmax
from model_utils import load_model

def load_trained_model(filepath='trained_cnn_model.npz'):
    """
    Load a trained CNN model and return initialized layers.

    Parameters:
    -----------
    filepath : str
        Path to the saved model file

    Returns:
    --------
    tuple : (conv, pool, softmax) layers with trained weights loaded
    """

    model_data = load_model(filepath)

    conv = Conv3x3(8)
    pool = MaxPool(2)
    softmax = Softmax(13 * 13 * 8, 10)

    # Restore trained weights
    conv.filters = model_data['conv_filters']
    softmax.weights = model_data['softmax_weights']
    softmax.biases = model_data['softmax_biases']

    return conv, pool, softmax, model_data['metadata']


def predict_digit(image, conv, pool, softmax):
    """
    Predict the digit in an image.

    Parameters:
    -----------
    image : numpy.ndarray
        28x28 grayscale image normalized to [-0.5, 0.5]
    conv : Conv3x3
        Trained convolutional layer
    pool : MaxPool
        Max pooling layer
    softmax : Softmax
        Trained softmax layer

    Returns:
    --------
    tuple : (predicted_digit, confidence, all_probabilities)
    """
    # Forward pass
    out = conv.forward(image)
    out = pool.forward(out)
    out = softmax.forward(out)

    # Get prediction
    predicted_digit = np.argmax(out)
    confidence = out[predicted_digit]

    return predicted_digit, confidence, out


def load_image_from_file(filepath):
    """
    Load and preprocess an image file for prediction.

    Parameters:
    -----------
    filepath : str
        Path to image file (PNG, JPG, etc.)

    Returns:
    --------
    numpy.ndarray : Preprocessed 28x28 image
    """
    # Load image and convert to grayscale 28x28
    img = Image.open(filepath).convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)

    # MNIST uses white digits on black background
    # If the image has a light background (average brightness > 127), invert it
    if np.mean(img_array) > 127:
        img_array = 255 - img_array

    # Normalize to [-0.5, 0.5]
    img_array = (img_array / 255.0) - 0.5

    return img_array


def predict_from_file(image_path, model_path='trained_cnn_model.npz'):
    """
    Load an image file and predict its digit.

    Parameters:
    -----------
    image_path : str
        Path to the image file
    model_path : str
        Path to the trained model

    Returns:
    --------
    dict : Prediction results with digit, confidence, and probabilities
    """

    conv, pool, softmax, metadata = load_trained_model(model_path)

    image = load_image_from_file(image_path)

    digit, confidence, probabilities = predict_digit(image, conv, pool, softmax)

    return {
        'digit': digit,
        'confidence': confidence,
        'probabilities': probabilities,
        'model_metadata': metadata
    }
