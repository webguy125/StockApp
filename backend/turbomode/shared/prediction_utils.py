"""
TurboMode Prediction Utilities
Canonical 3-class prediction schema with GPU acceleration and TensorRT optimization
"""

import numpy as np
from .model_behavior_config import MODEL_BEHAVIOR_CONFIG

# Optional GPU/TensorRT imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    torch_tensorrt = None


# =============================================================================
# GPU/TensorRT Utilities
# =============================================================================

def get_device():
    """Get the compute device (GPU if available and enabled, else CPU)"""
    if not TORCH_AVAILABLE:
        return None

    if MODEL_BEHAVIOR_CONFIG['use_gpu'] and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def compile_with_tensorrt(model, input_shape):
    """
    Compile PyTorch model with TensorRT for faster inference

    Args:
        model: PyTorch model to compile
        input_shape: Input tensor shape (batch_size, features)

    Returns:
        Compiled model if successful, original model otherwise
    """
    if not TENSORRT_AVAILABLE or not MODEL_BEHAVIOR_CONFIG['use_tensorrt']:
        return model

    if not torch.cuda.is_available():
        return model

    try:
        device = get_device()
        model = model.to(device)

        # Create TensorRT input specification
        inputs = [
            torch_tensorrt.Input(
                min_shape=input_shape,
                opt_shape=input_shape,
                max_shape=input_shape,
                dtype=torch.float32
            )
        ]

        # Compile with TensorRT
        trt_model = torch_tensorrt.compile(
            model,
            inputs=inputs,
            enabled_precisions={torch.float32}
        )

        return trt_model
    except Exception as e:
        # If TensorRT compilation fails, return original model
        print(f"[WARNING] TensorRT compilation failed: {e}")
        return model


# =============================================================================
# Probability Normalization Utilities
# =============================================================================

def canonicalize_model_name(name):
    """Canonicalize model name (lowercase, underscores)"""
    if not MODEL_BEHAVIOR_CONFIG['canonicalize_model_name']:
        return name
    return name.lower().replace(' ', '_').replace('-', '_')


def handle_nans(probabilities):
    """Handle NaN values in probabilities"""
    if MODEL_BEHAVIOR_CONFIG['nan_handling'] == 'zero':
        return np.nan_to_num(probabilities, nan=0.0)
    return probabilities


def normalize_probabilities(probabilities):
    """
    Normalize a single probability vector to canonical 3-class schema

    Handles:
    - Binary classification (2 classes) → maps to ['sell', 'buy']
    - 3-class classification → ['sell', 'hold', 'buy']
    - Padding missing classes with zeros
    - Normalizing probabilities to sum to 1.0
    - NaN handling and fallback to uniform distribution

    Args:
        probabilities: Array of probabilities from model

    Returns:
        Normalized 3-element probability array
    """
    probabilities = np.array(probabilities, dtype=np.float64)
    probabilities = handle_nans(probabilities)

    labels = MODEL_BEHAVIOR_CONFIG['class_labels']
    m = len(labels)  # Target number of classes (3)
    n = len(probabilities)  # Actual number of classes from model

    padded = np.zeros(m, dtype=np.float64)

    if n == m:
        # Perfect match: 3 classes
        padded = probabilities
    elif n == 2 and MODEL_BEHAVIOR_CONFIG['auto_detect_classes']:
        # Binary classification: map SELL (0) and BUY (1) to positions 0 and 2
        padded[0] = probabilities[0]  # sell/down
        padded[2] = probabilities[1]  # buy/up
        # padded[1] remains 0 (hold/neutral)
    elif MODEL_BEHAVIOR_CONFIG['missing_class_behavior'] == 'pad_with_zero':
        # Pad remaining classes with zeros
        padded[:n] = probabilities
    else:
        raise ValueError(f"Unsupported number of classes: {n}")

    # Handle zero-sum case
    total = padded.sum()
    if total == 0:
        if MODEL_BEHAVIOR_CONFIG['fallback_probabilities'] == 'uniform':
            padded = np.ones(m, dtype=np.float64) / m
        else:
            raise ValueError("Probabilities sum to zero and no fallback defined")

    # Normalize to sum to 1.0
    if MODEL_BEHAVIOR_CONFIG['normalize_probabilities']:
        if total > 0:
            padded = padded / total

    return padded


def normalize_prediction_class(prediction_class, original_n):
    """
    Normalize a single prediction class index to canonical 3-class schema

    Handles:
    - Binary classification: 0 (sell) → 0, 1 (buy) → 2
    - 3-class classification: no change

    Args:
        prediction_class: Class index from model
        original_n: Original number of classes in model output

    Returns:
        Normalized class index for 3-class schema
    """
    m = len(MODEL_BEHAVIOR_CONFIG['class_labels'])  # Target (3)

    if original_n == m:
        # No remapping needed
        return prediction_class

    if original_n == 2 and MODEL_BEHAVIOR_CONFIG['auto_detect_classes']:
        # Binary classification remapping
        if prediction_class == 0:
            return 0  # sell → sell/down
        if prediction_class == 1:
            return m - 1  # buy → buy/up (index 2)
        raise ValueError(f"Invalid prediction class for binary: {prediction_class}")

    # Default: no remapping
    return prediction_class


def normalize_probabilities_batch(prob_array):
    """
    Normalize batch of probability vectors to canonical 3-class schema

    Args:
        prob_array: Array of shape (n_samples, n_classes)

    Returns:
        Normalized array of shape (n_samples, 3)
    """
    return np.apply_along_axis(normalize_probabilities, 1, prob_array)


def normalize_prediction_class_batch(pred_classes, original_n):
    """
    Normalize batch of prediction class indices to canonical 3-class schema

    Args:
        pred_classes: Array of class indices
        original_n: Original number of classes in model output

    Returns:
        Normalized class indices for 3-class schema
    """
    return np.vectorize(lambda c: normalize_prediction_class(c, original_n))(pred_classes)


# =============================================================================
# Main Prediction Formatting
# =============================================================================

def format_prediction(probabilities, prediction_class, model_name):
    """
    Format prediction result with canonical 3-class schema

    This is the main function that all model predict() methods should call.
    It ensures consistent output format across all models.

    Args:
        probabilities: Raw probability array from model (2 or 3 elements)
        prediction_class: Predicted class index from model
        model_name: Name of the model making the prediction

    Returns:
        Dictionary with canonical prediction format:
        {
            'prediction': 'buy' | 'hold' | 'sell',
            'confidence': float,
            'model': str,
            'buy_prob': float,
            'hold_prob': float,
            'sell_prob': float
        }
    """
    original_n = len(probabilities)

    # Cast prediction_class to int if configured
    if MODEL_BEHAVIOR_CONFIG['force_int_prediction_class']:
        prediction_class = int(prediction_class)

    # Normalize prediction class index
    prediction_class = normalize_prediction_class(prediction_class, original_n)

    # Normalize probabilities to 3-class schema
    probabilities = normalize_probabilities(probabilities)

    # Get canonical class labels
    labels = MODEL_BEHAVIOR_CONFIG['class_labels']

    # Extract prediction details
    prediction_label = labels[prediction_class]
    confidence = float(np.max(probabilities))

    # Canonicalize model name
    model_name = canonicalize_model_name(model_name)

    # Build result dictionary
    result = {
        'prediction': prediction_label,
        'confidence': confidence,
        'model': model_name
    }

    # Add probability for each class
    for i, label in enumerate(labels):
        result[f'{label}_prob'] = float(probabilities[i])

    return result
