"""
Meta-Feature Matrix Diagnostic Tool

Analyzes the 30-column meta-feature matrix used by TurboMode meta-learner
to identify invalid columns and problematic base models.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent)
sys.path.insert(0, project_root)

from backend.turbomode.turbomode_training_loader import TurboModeTrainingDataLoader
from backend.turbomode.models.xgboost_model import XGBoostModel
from backend.turbomode.models.xgboost_et_model import XGBoostETModel
from backend.turbomode.models.lightgbm_model import LightGBMModel
from backend.turbomode.models.catboost_model import CatBoostModel
from backend.turbomode.models.xgboost_hist_model import XGBoostHistModel
from backend.turbomode.models.xgboost_dart_model import XGBoostDARTModel
from backend.turbomode.models.xgboost_gblinear_model import XGBoostGBLinearModel
from backend.turbomode.models.xgboost_approx_model import XGBoostApproxModel
from backend.turbomode.models.tc_nn_model import TurboCoreNNWrapper

MODEL_BASE_PATH = 'backend/data/turbomode_models'

# Model registry with column mapping
BASE_MODELS = [
    ('xgboost', XGBoostModel, 'xgboost', (0, 1, 2)),
    ('xgboost_et', XGBoostETModel, 'xgboost_et', (3, 4, 5)),
    ('lightgbm', LightGBMModel, 'lightgbm', (6, 7, 8)),
    ('catboost', CatBoostModel, 'catboost', (9, 10, 11)),
    ('xgboost_hist', XGBoostHistModel, 'xgboost_hist', (12, 13, 14)),
    ('xgboost_dart', XGBoostDARTModel, 'xgboost_dart', (15, 16, 17)),
    ('xgboost_gblinear', XGBoostGBLinearModel, 'xgboost_gblinear', (18, 19, 20)),
    ('xgboost_approx', XGBoostApproxModel, 'xgboost_approx', (21, 22, 23)),
    ('tc_nn_lstm', TurboCoreNNWrapper, 'tc_nn_lstm', (24, 25, 26)),
    ('tc_nn_gru', TurboCoreNNWrapper, 'tc_nn_gru', (27, 28, 29)),
]

CLASS_NAMES = ['DOWN', 'NEUTRAL', 'UP']


def load_data():
    """Load validation data"""
    print("\n" + "="*80)
    print("LOADING VALIDATION DATA")
    print("="*80)

    loader = TurboModeTrainingDataLoader()
    X_all, y_all = loader.load_training_data()

    # Use same train/val split as training
    split_idx = int(0.8 * len(X_all))
    X_val = X_all[split_idx:]
    y_val = y_all[split_idx:]

    print(f"[OK] Loaded {len(X_val):,} validation samples")
    return X_val, y_val


def load_models():
    """Load all 10 trained base models"""
    print("\n" + "="*80)
    print("LOADING TRAINED MODELS")
    print("="*80)

    models = []

    for model_name, model_class, model_dir, col_indices in BASE_MODELS:
        model_path = os.path.join(MODEL_BASE_PATH, model_dir)

        # Special handling for neural networks
        if model_name in ['tc_nn_lstm', 'tc_nn_gru']:
            recurrent_type = 'lstm' if 'lstm' in model_name else 'gru'
            model = model_class(
                input_dim=179,
                recurrent_type=recurrent_type,
                model_name=model_name
            )
            pth_file = f"{model_path}.pth"
            if os.path.exists(pth_file):
                model.load(pth_file)
                print(f"[OK] {model_name:20s} loaded from {pth_file}")
                models.append((model_name, model, col_indices))
            else:
                print(f"[FAIL] {model_name:20s} NOT FOUND at {pth_file}")
                models.append((model_name, None, col_indices))
        else:
            try:
                model = model_class(model_path=model_path, use_gpu=True)
                model.load()
                print(f"[OK] {model_name:20s} loaded from {model_path}")
                models.append((model_name, model, col_indices))
            except Exception as e:
                print(f"[FAIL] {model_name:20s} FAILED: {e}")
                models.append((model_name, None, col_indices))

    return models


def generate_predictions(models, X_val):
    """Generate predictions from all models"""
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)

    n_samples = len(X_val)
    meta_features = np.zeros((n_samples, 30), dtype=np.float32)

    for model_name, model, col_indices in models:
        if model is None:
            print(f"[WARN]  {model_name:20s} SKIPPED (not loaded)")
            # Fill with NaN to mark as missing
            meta_features[:, col_indices[0]:col_indices[2]+1] = np.nan
            continue

        try:
            probs = model.predict_proba(X_val)

            # Validate shape
            if probs.shape != (n_samples, 3):
                print(f"[FAIL] {model_name:20s} INVALID SHAPE: {probs.shape} (expected {(n_samples, 3)})")
                meta_features[:, col_indices[0]:col_indices[2]+1] = np.nan
                continue

            # Store predictions
            meta_features[:, col_indices[0]] = probs[:, 0]  # DOWN
            meta_features[:, col_indices[1]] = probs[:, 1]  # NEUTRAL
            meta_features[:, col_indices[2]] = probs[:, 2]  # UP

            print(f"[OK] {model_name:20s} predictions generated: {probs.shape}")

        except Exception as e:
            print(f"[FAIL] {model_name:20s} PREDICTION FAILED: {e}")
            meta_features[:, col_indices[0]:col_indices[2]+1] = np.nan

    return meta_features


def analyze_column(col_data, col_idx, model_name, class_name):
    """Analyze a single column for issues"""
    issues = []

    # Check for NaN
    if np.isnan(col_data).any():
        n_nan = np.isnan(col_data).sum()
        issues.append(f"NaN values: {n_nan}/{len(col_data)}")

    # Check for Inf
    if np.isinf(col_data).any():
        n_inf = np.isinf(col_data).sum()
        issues.append(f"Inf values: {n_inf}/{len(col_data)}")

    # Filter out NaN/Inf for further analysis
    valid_data = col_data[~np.isnan(col_data) & ~np.isinf(col_data)]

    if len(valid_data) == 0:
        issues.append("No valid values")
        return issues

    # Check variance
    std = np.std(valid_data)
    if std < 1e-10:
        issues.append(f"Zero variance (std={std:.2e})")

    # Check unique values
    n_unique = len(np.unique(valid_data))
    if n_unique == 1:
        issues.append(f"Constant value: {valid_data[0]:.6f}")
    elif n_unique < 10:
        issues.append(f"Low diversity: {n_unique} unique values")

    # Check range
    min_val, max_val = np.min(valid_data), np.max(valid_data)
    if min_val == max_val:
        issues.append(f"Constant: {min_val:.6f}")

    # Check if all zeros
    if np.all(valid_data == 0):
        issues.append("All zeros")

    # Check if all ones
    if np.all(valid_data == 1):
        issues.append("All ones")

    return issues


def analyze_meta_features(meta_features, models):
    """Analyze all 30 columns"""
    print("\n" + "="*80)
    print("ANALYZING META-FEATURE MATRIX")
    print("="*80)

    print(f"\nMatrix shape: {meta_features.shape}")
    print(f"Expected shape: ({len(meta_features)}, 30)")

    problematic_models = {}
    all_issues = {}

    print("\n" + "-"*80)
    print("COLUMN-BY-COLUMN ANALYSIS")
    print("-"*80)

    for model_name, model, col_indices in models:
        model_issues = []

        for i, class_idx in enumerate(col_indices):
            col_data = meta_features[:, class_idx]
            class_name = CLASS_NAMES[i]

            issues = analyze_column(col_data, class_idx, model_name, class_name)

            if issues:
                issue_str = "; ".join(issues)
                print(f"[FAIL] Col {class_idx:2d} [{model_name:20s} - {class_name:7s}]: {issue_str}")
                all_issues[class_idx] = (model_name, class_name, issues)
                model_issues.extend(issues)
            else:
                # Compute stats for valid columns
                valid_data = col_data[~np.isnan(col_data) & ~np.isinf(col_data)]
                mean_val = np.mean(valid_data)
                std_val = np.std(valid_data)
                min_val = np.min(valid_data)
                max_val = np.max(valid_data)
                print(f"[OK] Col {class_idx:2d} [{model_name:20s} - {class_name:7s}]: "
                      f"mean={mean_val:.4f}, std={std_val:.4f}, "
                      f"range=[{min_val:.4f}, {max_val:.4f}]")

        if model_issues:
            problematic_models[model_name] = model_issues

    return all_issues, problematic_models


def print_recommendations(all_issues, problematic_models):
    """Print recommendations"""
    print("\n" + "="*80)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*80)

    if not all_issues:
        print("\n[OK] ALL 30 COLUMNS ARE VALID")
        print("   No issues detected in meta-feature matrix.")
        print("   The LightGBM error may be due to other factors.")
        return

    print(f"\n[FAIL] FOUND {len(all_issues)} INVALID COLUMNS")
    print("\nInvalid columns:")
    for col_idx, (model_name, class_name, issues) in all_issues.items():
        print(f"  • Column {col_idx}: {model_name} - {class_name}")
        for issue in issues:
            print(f"    - {issue}")

    print(f"\n[FAIL] FOUND {len(problematic_models)} PROBLEMATIC MODELS")
    for model_name, issues in problematic_models.items():
        print(f"  • {model_name}")

    print("\n" + "-"*80)
    print("RECOMMENDATIONS")
    print("-"*80)

    if problematic_models:
        print("\n1. EXCLUDE problematic models from meta-learner:")
        print("   Edit train_turbomode_models.py BASE_MODELS list to remove:")
        for model_name in problematic_models.keys():
            print(f"     - {model_name}")

        print("\n2. OR RETRAIN problematic models with different settings:")
        for model_name in problematic_models.keys():
            if 'tc_nn' in model_name:
                print(f"   • {model_name}: Increase learning rate, reduce dropout, check input reshaping")
            else:
                print(f"   • {model_name}: Check hyperparameters, increase regularization")

        print("\n3. TEMPORARY WORKAROUND:")
        print("   Train meta-learner with only the valid models:")
        print(f"   Current valid models: {10 - len(problematic_models)}/10")


def main():
    print("\n" + "="*80)
    print("META-FEATURE MATRIX DIAGNOSTIC TOOL")
    print("="*80)

    # Load data
    X_val, y_val = load_data()

    # Load models
    models = load_models()

    # Generate predictions
    meta_features = generate_predictions(models, X_val)

    # Analyze
    all_issues, problematic_models = analyze_meta_features(meta_features, models)

    # Print recommendations
    print_recommendations(all_issues, problematic_models)

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
