import numpy as np
import os
from datetime import datetime

def save_model(conv_filters, softmax_weights, softmax_biases, filepath=None, metadata=None):
    """
    Save CNN model parameters to a .npz file.

    Parameters:
    -----------
    conv_filters : numpy.ndarray
        Convolutional layer filters with shape (num_filters, kernel_h, kernel_w)

    softmax_weights : numpy.ndarray
        Softmax layer weights with shape (input_len, num_classes)

    softmax_biases : numpy.ndarray
        Softmax layer biases with shape (num_classes,)

    filepath : str, optional
        Path to save the model. If None, generates a timestamped filename.

    metadata : dict, optional
        Additional metadata to save (e.g., training accuracy, epochs, hyperparameters)

    Returns:
    --------
    str : The filepath where the model was saved
    """

    if filepath is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = f'cnn_model_{timestamp}.npz'

    if not filepath.endswith('.npz'):
        filepath += '.npz'

    save_dict = {
        'conv_filters': conv_filters,
        'softmax_weights': softmax_weights,
        'softmax_biases': softmax_biases
    }

    if metadata is not None:
        for key, value in metadata.items():
            if not isinstance(value, np.ndarray):
                save_dict[f'meta_{key}'] = np.array(value)
            else:
                save_dict[f'meta_{key}'] = value

    np.savez_compressed(filepath, **save_dict)

    file_size = os.path.getsize(filepath)
    size_kb = file_size / 1024

    print(f'✓ Model saved to: {filepath}')
    print(f'  File size: {size_kb:.2f} KB')
    print(f'  Parameters saved:')
    print(f'    - Conv filters: {conv_filters.shape}')
    print(f'    - Softmax weights: {softmax_weights.shape}')
    print(f'    - Softmax biases: {softmax_biases.shape}')

    if metadata:
        print(f'  Metadata: {list(metadata.keys())}')

    return filepath


def load_model(filepath):
    """
    Load CNN model parameters from a .npz file.

    Parameters:
    -----------
    filepath : str
        Path to the saved model file

    Returns:
    --------
    dict : Dictionary containing:
        - 'conv_filters': Convolutional layer filters
        - 'softmax_weights': Softmax layer weights
        - 'softmax_biases': Softmax layer biases
        - 'metadata': Dictionary of any saved metadata (or None)

    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    data = np.load(filepath)

    result = {
        'conv_filters': data['conv_filters'],
        'softmax_weights': data['softmax_weights'],
        'softmax_biases': data['softmax_biases'],
        'metadata': {}
    }

    for key in data.files:
        if key.startswith('meta_'):
            meta_key = key[5:]  # Remove 'meta_' prefix
            result['metadata'][meta_key] = data[key].item() if data[key].ndim == 0 else data[key]

    if not result['metadata']:
        result['metadata'] = None

    print(f'✓ Model loaded from: {filepath}')
    print(f'  Parameters loaded:')
    print(f'    - Conv filters: {result["conv_filters"].shape}')
    print(f'    - Softmax weights: {result["softmax_weights"].shape}')
    print(f'    - Softmax biases: {result["softmax_biases"].shape}')

    if result['metadata']:
        print(f'  Metadata: {result["metadata"]}')

    return result