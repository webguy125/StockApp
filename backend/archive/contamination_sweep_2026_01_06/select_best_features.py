"""
Feature Selection for TurboMode Models
Analyzes feature importance from trained models and selects top N features
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from advanced_ml.backtesting.historical_backtest import HistoricalBacktest

# Configuration
DB_PATH = backend_path / "data" / "turbomode.db"
MODEL_PATH = backend_path / "data" / "turbomode_models"
OUTPUT_FILE = backend_path / "turbomode" / "selected_features.json"

# Number of features to select
TOP_N_FEATURES = 100  # Can adjust: 50, 75, 100, 150


def load_feature_importance_from_models():
    """Extract feature importance from all trained models"""

    print("\n" + "=" * 70)
    print("FEATURE SELECTION - ANALYZE MODEL IMPORTANCE")
    print("=" * 70)

    # Import models
    from advanced_ml.models.xgboost_rf_model import XGBoostRFModel
    from advanced_ml.models.xgboost_model import XGBoostModel
    from advanced_ml.models.xgboost_et_model import XGBoostETModel
    from advanced_ml.models.lightgbm_model import LightGBMModel

    # Load models (skip CatBoost - has compatibility issues with feature importance extraction)
    print("\n[1/5] Loading trained models...")
    models = {
        'xgboost_rf': XGBoostRFModel(model_path=str(MODEL_PATH / "xgboost_rf")),
        'xgboost': XGBoostModel(model_path=str(MODEL_PATH / "xgboost")),
        'xgboost_et': XGBoostETModel(model_path=str(MODEL_PATH / "xgboost_et")),
        'lightgbm': LightGBMModel(model_path=str(MODEL_PATH / "lightgbm"))
    }

    # Get feature importance from each model
    print("\n[2/5] Extracting feature importance from each model...")
    all_importances = {}

    for name, model in models.items():
        if not model.is_trained:
            print(f"  [SKIP] {name} - not trained")
            continue

        try:
            # Get feature importance
            if hasattr(model.model, 'feature_importances_'):
                importances = model.model.feature_importances_
            elif hasattr(model.model, 'get_feature_importance'):
                # CatBoost returns array or list
                importances = np.array(model.model.get_feature_importance())
            elif hasattr(model.model, 'get_booster'):
                booster = model.model.get_booster()
                importance_dict = booster.get_score(importance_type='gain')
                # Convert to array
                importances = np.zeros(176)  # Assuming 176 features
                for feat, score in importance_dict.items():
                    idx = int(feat.replace('f', ''))
                    importances[idx] = score
            else:
                print(f"  [SKIP] {name} - no feature importance available")
                continue

            # Ensure it's a numpy array
            if not isinstance(importances, np.ndarray):
                importances = np.array(importances)

            all_importances[name] = importances
            print(f"  [OK] {name} - {len(importances)} features")

        except Exception as e:
            print(f"  [ERROR] {name} - {e}")
            continue

    if not all_importances:
        print("\n[ERROR] No feature importances could be extracted!")
        return None

    # Combine importances (average across models)
    print(f"\n[3/5] Combining importance scores from {len(all_importances)} models...")

    # Stack all importance arrays
    importance_arrays = list(all_importances.values())

    # Ensure all same length
    min_len = min(len(arr) for arr in importance_arrays)
    importance_arrays = [arr[:min_len] for arr in importance_arrays]

    # Average importance across models
    combined_importance = np.mean(importance_arrays, axis=0)

    # Normalize to 0-1
    if combined_importance.max() > 0:
        combined_importance = combined_importance / combined_importance.max()

    return combined_importance


def select_top_features(importance_scores, top_n=100):
    """Select top N features based on importance"""

    print(f"\n[4/5] Selecting top {top_n} features...")

    # Load feature name mapping
    mapping_file = backend_path / "turbomode" / "feature_name_mapping.json"
    if not mapping_file.exists():
        print(f"[ERROR] Feature name mapping not found: {mapping_file}")
        print(f"[ERROR] Run create_feature_mapping.py first!")
        return None

    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
        feature_names = mapping_data['feature_names_in_order']

    print(f"[OK] Loaded {len(feature_names)} feature names from mapping")

    # Get indices of top N features
    top_indices = np.argsort(importance_scores)[::-1][:top_n]
    top_indices_sorted = sorted(top_indices.tolist())  # Keep in original order as list

    # Get importance scores for top features
    top_scores = importance_scores[top_indices]

    # Create feature info with REAL names
    feature_info = []
    for idx, score in zip(top_indices, top_scores):
        actual_name = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
        feature_info.append({
            'index': int(idx),
            'name': actual_name,  # Real name like 'sma_20', 'rsi_14'
            'importance': float(score)
        })

    # Sort by importance (descending)
    feature_info = sorted(feature_info, key=lambda x: x['importance'], reverse=True)

    # Print top 20
    print("\nTop 20 most important features:")
    for i, feat in enumerate(feature_info[:20], 1):
        print(f"  {i:2d}. {feat['name']:25s} - Importance: {feat['importance']:.4f}")

    return {
        'top_n': top_n,
        'total_features': len(importance_scores),
        'selected_indices': top_indices_sorted,
        'feature_info': feature_info,
        'selection_date': pd.Timestamp.now().isoformat()
    }


def save_selected_features(selection_data, output_file):
    """Save selected features to JSON"""

    print(f"\n[5/5] Saving selected features to {output_file}...")

    with open(output_file, 'w') as f:
        json.dump(selection_data, f, indent=2)

    print(f"[OK] Saved {len(selection_data['selected_indices'])} features")
    print(f"\nFeature reduction: {selection_data['total_features']} -> {selection_data['top_n']}")
    reduction_pct = (1 - selection_data['top_n'] / selection_data['total_features']) * 100
    print(f"Speed improvement: ~{reduction_pct:.1f}% fewer features to compute")


def main():
    """Run feature selection"""

    print("\n" + "=" * 70)
    print("TURBOMODE FEATURE SELECTION")
    print("=" * 70)
    print(f"Database: {DB_PATH}")
    print(f"Models: {MODEL_PATH}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Selecting top {TOP_N_FEATURES} features")

    # Step 1: Load feature importance
    importance_scores = load_feature_importance_from_models()

    if importance_scores is None:
        print("\n[FAILED] Could not extract feature importance")
        return 1

    # Step 2: Select top features
    selection_data = select_top_features(importance_scores, top_n=TOP_N_FEATURES)

    # Step 3: Save results
    save_selected_features(selection_data, OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("FEATURE SELECTION COMPLETE!")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"1. Review selected features in: {OUTPUT_FILE}")
    print(f"2. Modify GPUFeatureEngineer to use only selected features")
    print(f"3. Re-run backtest with reduced feature set")
    print(f"4. Train models on faster, cleaner data")

    return 0


if __name__ == "__main__":
    sys.exit(main())
