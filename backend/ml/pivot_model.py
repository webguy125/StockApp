"""
Pivot Reliability Model
PyTorch MLP for binary classification of pivot point reliability
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Model file location
MODEL_DIR = Path(__file__).parent / 'models'
MODEL_PATH = MODEL_DIR / 'pivot_model.pth'


class PivotClassifier(nn.Module):
    """
    Multi-Layer Perceptron for pivot reliability prediction

    Architecture:
        Input: 9 features
        Hidden 1: 32 neurons (ReLU)
        Hidden 2: 16 neurons (ReLU)
        Output: 1 neuron (Sigmoid) - probability [0, 1]
    """

    def __init__(self, input_dim=9, hidden1=32, hidden2=16):
        super(PivotClassifier, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, 9)

        Returns:
            Output tensor of shape (batch_size, 1) with probabilities
        """
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.sigmoid(x)

        return x

    def predict_proba(self, features):
        """
        Predict probability for feature array

        Args:
            features: numpy array of shape (n_samples, 9) or (9,)

        Returns:
            numpy array of probabilities
        """
        self.eval()

        with torch.no_grad():
            # Handle single sample
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Convert to tensor
            x = torch.FloatTensor(features)

            # Predict
            probs = self.forward(x)

            return probs.numpy().flatten()

    def predict(self, features, threshold=0.7):
        """
        Predict binary class (reliable pivot or not)

        Args:
            features: numpy array of shape (n_samples, 9) or (9,)
            threshold: Probability threshold for positive class

        Returns:
            numpy array of binary predictions (0 or 1)
        """
        probs = self.predict_proba(features)
        return (probs >= threshold).astype(int)


def save_model(model, optimizer=None, epoch=None, loss=None, path=None):
    """
    Save model checkpoint

    Args:
        model: PivotClassifier instance
        optimizer: Optimizer instance (optional)
        epoch: Current epoch number (optional)
        loss: Current loss value (optional)
        path: Save path (default: MODEL_PATH)
    """
    try:
        path = path or MODEL_PATH

        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': 9,
                'hidden1': 32,
                'hidden2': 16
            }
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        if loss is not None:
            checkpoint['loss'] = loss

        # Save
        torch.save(checkpoint, path)
        logger.info(f"✅ Model saved to {path}")

    except Exception as e:
        logger.error(f"❌ Model save failed: {e}")
        raise


def load_model(path=None, device='cpu'):
    """
    Load model from checkpoint

    Args:
        path: Model file path (default: MODEL_PATH)
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        PivotClassifier instance
    """
    try:
        path = path or MODEL_PATH

        if not Path(path).exists():
            logger.warning(f"⚠️ Model file not found: {path}")
            logger.info("Creating new untrained model...")
            return PivotClassifier()

        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)

        # Extract config
        config = checkpoint.get('model_config', {
            'input_dim': 9,
            'hidden1': 32,
            'hidden2': 16
        })

        # Create model
        model = PivotClassifier(**config)

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])

        # Set to evaluation mode
        model.eval()

        logger.info(f"✅ Model loaded from {path}")

        # Log training info if available
        if 'epoch' in checkpoint:
            logger.info(f"   Trained for {checkpoint['epoch']} epochs")
        if 'loss' in checkpoint:
            logger.info(f"   Final loss: {checkpoint['loss']:.4f}")

        return model

    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        raise


def inference(features, model=None, threshold=0.7):
    """
    Run inference on feature array

    Args:
        features: numpy array of shape (n_samples, 9) or (9,)
        model: PivotClassifier instance (will load if None)
        threshold: Probability threshold

    Returns:
        dict: {probabilities, predictions, reliable_count}
    """
    try:
        # Load model if not provided
        if model is None:
            model = load_model()

        # Predict
        probs = model.predict_proba(features)
        preds = (probs >= threshold).astype(int)

        result = {
            'probabilities': probs.tolist(),
            'predictions': preds.tolist(),
            'reliable_count': int(np.sum(preds)),
            'total_count': len(preds)
        }

        logger.info(f"✅ Inference: {result['reliable_count']}/{result['total_count']} reliable pivots")
        return result

    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        raise


def batch_inference(feature_arrays, model=None, threshold=0.7):
    """
    Run inference on multiple feature arrays (e.g., different assets/timeframes)

    Args:
        feature_arrays: dict of {key: features_array}
        model: PivotClassifier instance
        threshold: Probability threshold

    Returns:
        dict: {key: inference_result}
    """
    try:
        # Load model once
        if model is None:
            model = load_model()

        results = {}

        for key, features in feature_arrays.items():
            results[key] = inference(features, model=model, threshold=threshold)

        logger.info(f"✅ Batch inference complete: {len(results)} datasets")
        return results

    except Exception as e:
        logger.error(f"❌ Batch inference failed: {e}")
        raise


# Singleton model instance for Flask API
_model_instance = None


def get_model():
    """
    Get singleton model instance (for Flask API)

    Returns:
        PivotClassifier instance
    """
    global _model_instance

    if _model_instance is None:
        logger.info("📊 Loading ML model for the first time...")
        _model_instance = load_model()

    return _model_instance


def predict_pivot_reliability(features, threshold=0.7):
    """
    Predict pivot reliability using singleton model

    This is the main function used by the Flask API endpoint.

    Args:
        features: numpy array or list of 9 features

    Returns:
        float: Probability score [0, 1]
    """
    try:
        # Convert to numpy array if needed
        if not isinstance(features, np.ndarray):
            features = np.array(features, dtype=np.float32)

        # Validate shape
        if features.shape != (9,):
            raise ValueError(f"Expected 9 features, got {features.shape}")

        # Get model
        model = get_model()

        # Predict
        prob = model.predict_proba(features)[0]

        return float(prob)

    except Exception as e:
        logger.error(f"❌ Pivot reliability prediction failed: {e}")
        raise
