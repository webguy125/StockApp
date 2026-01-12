"""
TurboMode Inference Pipeline - Purified Architecture

Clean, deterministic inference system aligned with purified training architecture.
NO StandardScaler, NO preprocessing, NO contamination.

Architecture:
- 10 Base Models: 7 XGBoost variants + LightGBM + CatBoost + 2 Neural Networks
- 1 Meta-Learner: Combines base model predictions
- Feature Flow: RAW 179 features → Base Models → Probabilities → Meta-Learner → Final Prediction
"""

import os
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

# ============================================================================
# PURIFIED MODEL IMPORTS - backend.turbomode.models ONLY
# ============================================================================

# Tree-based models (scale-invariant, accept RAW features)
from backend.turbomode.models.xgboost_model import XGBoostModel
from backend.turbomode.models.xgboost_et_model import XGBoostETModel
from backend.turbomode.models.xgboost_hist_model import XGBoostHistModel
from backend.turbomode.models.xgboost_dart_model import XGBoostDARTModel
from backend.turbomode.models.xgboost_gblinear_model import XGBoostGBLinearModel
from backend.turbomode.models.xgboost_approx_model import XGBoostApproxModel
from backend.turbomode.models.lightgbm_model import LightGBMModel
from backend.turbomode.models.catboost_model import CatBoostModel

# Meta-learner (accepts RAW probability vectors)
from backend.turbomode.models.meta_learner import MetaLearner

# Neural network models (use training wrapper for compatibility)
from backend.turbomode.models.tc_nn_model import TurboCoreNNWrapper

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_BASE_PATH = 'backend/data/turbomode_models'

# Model registry (order must match training order for determinism)
BASE_MODELS = [
    ('xgboost', XGBoostModel, 'xgboost'),
    ('xgboost_et', XGBoostETModel, 'xgboost_et'),
    ('lightgbm', LightGBMModel, 'lightgbm'),
    ('catboost', CatBoostModel, 'catboost'),
    ('xgboost_hist', XGBoostHistModel, 'xgboost_hist'),
    ('xgboost_dart', XGBoostDARTModel, 'xgboost_dart'),
    ('xgboost_gblinear', XGBoostGBLinearModel, 'xgboost_gblinear'),
    ('xgboost_approx', XGBoostApproxModel, 'xgboost_approx'),
    ('tc_nn_lstm', TurboCoreNNWrapper, 'tc_nn_lstm'),
    ('tc_nn_gru', TurboCoreNNWrapper, 'tc_nn_gru')
]


# ============================================================================
# INFERENCE PIPELINE
# ============================================================================

class TurboModeInference:
    """
    Purified inference pipeline for TurboMode predictions.

    Loads trained models and generates predictions on RAW features.
    NO StandardScaler, NO preprocessing transforms.
    """

    def __init__(self, model_base_path: str = None, use_trt: bool = False):
        """
        Initialize inference pipeline.

        Args:
            model_base_path: Path to model directory
            use_trt: Whether to use TensorRT for neural networks (inference optimization)
        """
        self.model_base_path = model_base_path or MODEL_BASE_PATH
        self.use_trt = use_trt
        self.base_models = {}
        self.meta_learner = None
        self.is_loaded = False

    def load_models(self):
        """
        Load all trained models from disk.

        Loads:
        - 8 tree-based models (XGBoost variants, LightGBM, CatBoost)
        - 2 neural network models (LSTM, GRU) with optional TRT optimization
        - 1 meta-learner (ensemble)
        """
        print("\n" + "="*80)
        print("LOADING TURBOMODE MODELS (Purified Architecture)")
        print("="*80)

        # Load base models
        for model_name, model_class, model_path in BASE_MODELS:
            print(f"\nLoading: {model_name.upper()}")
            full_path = os.path.join(self.model_base_path, model_path)

            try:
                # Handle neural network models differently
                if 'tc_nn' in model_name:
                    recurrent_type = 'lstm' if 'lstm' in model_name else 'gru'
                    model = model_class(
                        input_dim=179,  # RAW feature dimension
                        num_classes=3,  # Force 3-class mode
                        recurrent_type=recurrent_type,
                        model_name=model_name
                    )
                    # Load model weights from .pth file
                    model.load(os.path.join(full_path, f'{model_name}.pth'))
                else:
                    model = model_class(model_path=full_path, use_gpu=True)
                    # Load model weights
                    model.load()

                self.base_models[model_name] = model
                print(f"  ✅ {model_name} loaded")

            except Exception as e:
                print(f"  ❌ Failed to load {model_name}: {e}")
                raise

        # Load meta-learner
        print(f"\nLoading: META-LEARNER")
        meta_path = os.path.join(self.model_base_path, 'meta_learner')

        try:
            self.meta_learner = MetaLearner(model_path=meta_path, use_gpu=True)
            self.meta_learner.load()
            print(f"  ✅ Meta-learner loaded")
        except Exception as e:
            print(f"  ❌ Failed to load meta-learner: {e}")
            raise

        self.is_loaded = True
        print("\n" + "="*80)
        print("✅ ALL MODELS LOADED SUCCESSFULLY")
        print("="*80 + "\n")

    def predict(self, features: np.ndarray) -> Dict[str, float]:
        """
        Generate 3-class prediction for a single sample.

        Args:
            features: RAW feature vector (179 dimensions, NO scaling)

        Returns:
            Dictionary with:
            - prob_down: Probability of down movement
            - prob_neutral: Probability of neutral/sideways movement
            - prob_up: Probability of up movement
            - predicted_class: 0=down, 1=neutral, 2=up
            - base_predictions: Individual model 3-class predictions (Dict[str, np.ndarray])
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")

        if features.ndim == 1:
            features = features.reshape(1, -1)

        if features.shape[1] != 179:
            raise ValueError(f"Expected 179 features, got {features.shape[1]}")

        # Step 1: Get 3-class predictions from all base models
        base_predictions = {}

        for model_name, model in self.base_models.items():
            try:
                probs = model.predict(features[0])  # Returns np.ndarray([prob_down, prob_neutral, prob_up])
                if len(probs) != 3:
                    raise ValueError(f"{model_name} returned {len(probs)} classes, expected 3")
                base_predictions[model_name] = probs
            except Exception as e:
                print(f"Warning: {model_name} prediction failed: {e}")
                # Use uniform fallback for 3-class
                base_predictions[model_name] = np.array([0.333, 0.333, 0.333], dtype=np.float32)

        # Step 2: Get final prediction from meta-learner
        final_pred = self.meta_learner.predict(base_predictions)

        # Step 3: Return comprehensive 3-class prediction
        return {
            'prob_down': final_pred['prob_down'],
            'prob_neutral': final_pred['prob_neutral'],
            'prob_up': final_pred['prob_up'],
            'predicted_class': final_pred['predicted_class'],  # 0=down, 1=neutral, 2=up
            'base_predictions': base_predictions  # Dict of model_name -> np.ndarray([prob_down, prob_neutral, prob_up])
        }

    def predict_batch(self, features: np.ndarray) -> List[Dict[str, float]]:
        """
        Generate predictions for multiple samples.

        Args:
            features: RAW feature matrix (N x 179, NO scaling)

        Returns:
            List of prediction dictionaries
        """
        if not self.is_loaded:
            raise ValueError("Models not loaded. Call load_models() first.")

        if features.shape[1] != 179:
            raise ValueError(f"Expected 179 features, got {features.shape[1]}")

        predictions = []

        for i in range(len(features)):
            pred = self.predict(features[i])
            predictions.append(pred)

        return predictions


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_turbomode_inference(use_trt: bool = False) -> TurboModeInference:
    """
    Load TurboMode inference pipeline (convenience function).

    Args:
        use_trt: Whether to use TensorRT for neural networks

    Returns:
        Initialized TurboModeInference instance
    """
    inference = TurboModeInference(use_trt=use_trt)
    inference.load_models()
    return inference


def predict_single(features: np.ndarray, use_trt: bool = False) -> Dict[str, float]:
    """
    One-shot prediction on a single sample (convenience function).

    Args:
        features: RAW feature vector (179 dimensions)
        use_trt: Whether to use TensorRT

    Returns:
        Prediction dictionary
    """
    inference = load_turbomode_inference(use_trt=use_trt)
    return inference.predict(features)


# ============================================================================
# MAIN (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TURBOMODE INFERENCE PIPELINE TEST")
    print("="*80)

    # Test with dummy features (179 dimensions)
    print("\nGenerating test features (179 dimensions)...")
    test_features = np.random.randn(179)

    # Load inference pipeline
    print("\nLoading inference pipeline...")
    inference = load_turbomode_inference(use_trt=False)

    # Generate prediction
    print("\nGenerating prediction...")
    prediction = inference.predict(test_features)

    # Display results
    print("\n" + "="*80)
    print("PREDICTION RESULTS (3-Class)")
    print("="*80)
    print(f"Down Probability:    {prediction['prob_down']:.4f}")
    print(f"Neutral Probability: {prediction['prob_neutral']:.4f}")
    print(f"Up Probability:      {prediction['prob_up']:.4f}")
    print(f"Predicted Class:     {prediction['predicted_class']} (0=down, 1=neutral, 2=up)")

    print("\nBase Model Predictions:")
    for model_name, pred in prediction['base_predictions'].items():
        print(f"  {model_name:20s}: down={pred[0]:.4f}, neutral={pred[1]:.4f}, up={pred[2]:.4f}")

    print("\n" + "="*80)
    print("✅ INFERENCE TEST COMPLETE")
    print("="*80 + "\n")
