"""
TurboMode Model Behavior Configuration
Canonical 3-class prediction schema with GPU acceleration
"""

MODEL_BEHAVIOR_CONFIG = {
    # Class labels in canonical order: SELL/DOWN (0), HOLD/NEUTRAL (1), BUY/UP (2)
    'class_labels': ['sell', 'hold', 'buy'],

    # Auto-detect number of classes from probability array
    # If True: 2 classes → ['sell', 'buy'], 3 classes → ['sell', 'hold', 'buy']
    'auto_detect_classes': True,

    # Force prediction_class to int type (prevents float indices)
    'force_int_prediction_class': True,

    # How to handle models with fewer classes than canonical schema
    # 'pad_with_zero': Fill missing classes with 0.0 probability
    'missing_class_behavior': 'pad_with_zero',

    # Fallback if all probabilities sum to zero
    # 'uniform': Distribute probability equally across all classes
    'fallback_probabilities': 'uniform',

    # Normalize probabilities to sum to 1.0
    'normalize_probabilities': True,

    # How to handle NaN values in probabilities
    # 'zero': Replace NaN with 0.0
    'nan_handling': 'zero',

    # Canonicalize model names (lowercase, underscores)
    'canonicalize_model_name': True,

    # GPU acceleration for predictions (requires CUDA)
    'use_gpu': True,

    # TensorRT optimization for PyTorch models (requires TensorRT)
    'use_tensorrt': False  # Disabled by default, enable after testing
}
