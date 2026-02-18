
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Fast Ensemble Inference Engine - 14-Day Swing Trade Predictions

ARCHITECTURE: 5 fast base models + MetaLearner per sector
- 3 GPU models: LightGBM-GPU, CatBoost-GPU, XGBoost-Hist-GPU
- 2 CPU models: XGBoost-Linear, RandomForest
- 1 MetaLearner: LogisticRegression (trained on stacked base model outputs)

**14-DAY SWING TRADE SEMANTICS:**
- Index 0: SELL (14-day bearish swing, go short)
- Index 1: HOLD (14-day neutral, no directional edge)
- Index 2: BUY (14-day bullish swing, go long)

**LABEL TRAINING:**
- BUY: Expected 14-day return >= +3%
- SELL: Expected 14-day return <= -3%
- HOLD: Expected 14-day return between -3% and +3%

**NOTE:** This matches the training labels in sector_batch_trainer.py (0=SELL, 1=HOLD, 2=BUY)

INFERENCE PIPELINE:
1. Load all 6 models for the sector (cached via model_loader)
2. Run predict_proba on all 5 base models
3. Stack base model outputs into feature vector (15 features: 5 models x 3 classes)
4. Pass stacked features to MetaLearner.predict_proba
5. Return final 14-day swing BUY/SELL/HOLD probabilities
"""

import numpy as np
from typing import Dict
import logging

# Import model loader
from backend.turbomode.core_engine.model_loader import load_sector_models
from backend.turbomode.core_engine.model_registry import BASE_MODELS, META_LEARNER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_ensemble(sector: str, X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Run ensemble inference for multiple samples.

    ARCHITECTURE: 5 base models + MetaLearner per sector

    Args:
        sector: Sector name (e.g., 'technology')
        X: Feature matrix (N, n_features)

    Returns:
        Dictionary with:
            - "probs": Final probabilities from MetaLearner, shape (N, 3)
            - "labels": String labels ("SELL", "HOLD", "BUY"), shape (N,)
            - "class_indices": Integer class indices (0=SELL, 1=HOLD, 2=BUY), shape (N,)
    """
    # Load all 6 models for sector (cached via lru_cache)
    models = load_sector_models(sector)

    # Step 1: Run predict_proba on all 5 base models
    base_predictions = []
    for model_name in BASE_MODELS:
        model = models[model_name]
        probs = model.predict_proba(X)  # shape: (N, 3)

        # Verify 3 classes
        if probs.shape[1] != 3:
            raise ValueError(
                f"{model_name} returned {probs.shape[1]} classes, expected 3 (BUY/SELL/HOLD)"
            )

        base_predictions.append(probs)

    # Step 2: Stack base model predictions into feature matrix
    # Shape: (N, 15) = 5 models x 3 classes per model
    stacked_features = np.concatenate(base_predictions, axis=1)

    # Step 3: Run MetaLearner on stacked features
    meta_learner = models[META_LEARNER]
    final_probs = meta_learner.predict_proba(stacked_features)  # shape: (N, 3)

    # Verify 3 classes
    if final_probs.shape[1] != 3:
        raise ValueError(
            f"MetaLearner returned {final_probs.shape[1]} classes, expected 3 (BUY/SELL/HOLD)"
        )

    # Step 4: Get class indices and labels
    class_indices = np.argmax(final_probs, axis=1)  # shape: (N,)

    # CLASS SEMANTICS: Index 0=SELL, 1=HOLD, 2=BUY (matches training labels)
    class_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    labels = np.array([class_map[idx] for idx in class_indices])

    return {
        'probs': final_probs,
        'labels': labels,
        'class_indices': class_indices,
    }


def predict_single(sector: str, features: np.ndarray) -> Dict[str, float]:
    """
    Run ensemble inference for a single sample.

    ARCHITECTURE: 5 base models + MetaLearner per sector

    Args:
        sector: Sector name (e.g., 'technology')
        features: Feature vector (n_features,) or (1, n_features)

    Returns:
        Dictionary with:
            - "signal": "SELL", "HOLD", or "BUY"
            - "prob_sell": Probability of SELL (index 0)
            - "prob_hold": Probability of HOLD (index 1)
            - "prob_buy": Probability of BUY (index 2)
            - "confidence": Max probability
    """
    # Ensure 2D
    if features.ndim == 1:
        features = features.reshape(1, -1)

    # Run ensemble inference
    result = predict_ensemble(sector, features)

    # Extract single sample results
    probs = result['probs'][0]  # shape: (3,)
    signal = result['labels'][0]  # "SELL", "HOLD", or "BUY"

    # CLASS SEMANTICS: Index 0=SELL, 1=HOLD, 2=BUY (matches training labels)
    return {
        'signal': signal,
        'prob_sell': float(probs[0]),
        'prob_hold': float(probs[1]),
        'prob_buy': float(probs[2]),
        'confidence': float(probs.max()),
    }


# ============================================================================
# BACKWARD COMPATIBILITY WRAPPERS
# ============================================================================

def load_model(sector: str):
    """
    DEPRECATED: Legacy wrapper for backward compatibility.

    Old code expected a single model object. Now we return the MetaLearner
    as the "primary" model, but this breaks the old inference pipeline.

    New code should use predict_ensemble(sector, X) or predict_single(sector, X) directly.
    """
    logger.warning(
        f"[DEPRECATED] load_model() called for {sector}. "
        f"Use predict_ensemble() or predict_single() instead."
    )
    models = load_sector_models(sector)
    return models[META_LEARNER]


def load_fastmode_models(sector: str, horizon: str = "1d"):
    """
    DEPRECATED: Legacy wrapper for backward compatibility.

    Ignores horizon parameter (only 1d/5% models exist).
    """
    if horizon != "1d":
        logger.warning(f"[DEPRECATED] Ignoring horizon '{horizon}' - only 1d/5% models exist")

    return load_model(sector)


def predict(model, X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    DEPRECATED: Legacy wrapper for backward compatibility.

    Old code passed a single model object. This function cannot work properly
    in the new ensemble architecture since we need the sector name to load
    all 6 models.

    Raises NotImplementedError to force migration to new API.
    """
    raise NotImplementedError(
        "predict(model, X) is deprecated in ensemble architecture.\n"
        "Use predict_ensemble(sector, X) instead.\n"
        "Example: result = predict_ensemble('technology', X)"
    )


if __name__ == '__main__':
    # Test ensemble inference with synthetic data
    logger.info("Testing ensemble inference with synthetic data...")

    import numpy as np

    np.random.seed(42)
    n_samples = 10
    n_features = 179

    # Generate synthetic features
    X_test = np.random.randn(n_samples, n_features)

    # Test for a sector (assumes models are already trained)
    test_sector = 'technology'

    try:
        logger.info(f"\nTesting ensemble inference for {test_sector}...")

        # Test batch prediction
        result = predict_ensemble(test_sector, X_test)
        logger.info(f"  ✓ Batch prediction successful")
        logger.info(f"    Output shape: {result['probs'].shape}")
        logger.info(f"    Signals: {result['labels']}")

        # Test single prediction
        single_result = predict_single(test_sector, X_test[0])
        logger.info(f"  ✓ Single prediction successful")
        logger.info(f"    Signal: {single_result['signal']}")
        logger.info(f"    BUY: {single_result['prob_buy']:.3f}")
        logger.info(f"    SELL: {single_result['prob_sell']:.3f}")
        logger.info(f"    HOLD: {single_result['prob_hold']:.3f}")
        logger.info(f"    Confidence: {single_result['confidence']:.3f}")

    except FileNotFoundError as e:
        logger.error(f"  ✗ Models not found: {e}")
        logger.info(f"    Run train_sector_models.py to train {test_sector} models first")
    except Exception as e:
        logger.error(f"  ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
