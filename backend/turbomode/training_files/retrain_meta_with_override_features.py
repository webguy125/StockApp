"""
Retrain Final Meta-Learner with Override-Aware Features

This script retrains the final meta-learner using:
1. Original 24 base model features (8 models × 3 probabilities)
2. Override-aware features (asymmetry, directional confidence, neutral dominance)

The enhanced meta-learner learns to produce stronger directional signals
when appropriate, reducing reliance on the post-hoc override layer.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.turbomode.turbomode_training_loader import TurboModeTrainingDataLoader


def add_override_aware_features(df: pd.DataFrame, model_names: list) -> pd.DataFrame:
    """
    Add override-aware features to the dataset.

    These features help the meta-learner learn when to produce
    stronger directional signals without needing post-hoc override.

    Args:
        df: DataFrame with base model probabilities
        model_names: List of 8 model names

    Returns:
        DataFrame with additional override-aware features
    """
    # For each model, calculate override-aware features
    for model_name in model_names:
        prob_up = df[f'{model_name}_prob_up']
        prob_down = df[f'{model_name}_prob_down']
        prob_neutral = df[f'{model_name}_prob_neutral']

        # Asymmetry between buy and sell
        df[f'{model_name}_asymmetry'] = np.abs(prob_up - prob_down)

        # Maximum directional probability
        df[f'{model_name}_max_directional'] = np.maximum(prob_up, prob_down)

        # Neutral dominance (how much neutral exceeds directional)
        df[f'{model_name}_neutral_dominance'] = prob_neutral - np.maximum(prob_up, prob_down)

    # Aggregate features across all models
    asymmetry_cols = [f'{m}_asymmetry' for m in model_names]
    max_dir_cols = [f'{m}_max_directional' for m in model_names]
    neutral_dom_cols = [f'{m}_neutral_dominance' for m in model_names]

    df['avg_asymmetry'] = df[asymmetry_cols].mean(axis=1)
    df['max_asymmetry'] = df[asymmetry_cols].max(axis=1)
    df['avg_max_directional'] = df[max_dir_cols].mean(axis=1)
    df['avg_neutral_dominance'] = df[neutral_dom_cols].mean(axis=1)

    # Consensus features (how many models agree on direction)
    up_cols = [f'{m}_prob_up' for m in model_names]
    down_cols = [f'{m}_prob_down' for m in model_names]

    df['models_favor_up'] = (df[up_cols].values > df[down_cols].values).sum(axis=1)
    df['models_favor_down'] = (df[down_cols].values > df[up_cols].values).sum(axis=1)
    df['directional_consensus'] = np.abs(df['models_favor_up'] - df['models_favor_down']) / 8.0

    return df


def retrain_meta_learner(
    training_db_path: str = None,
    output_path: str = None,
    use_class_weights: bool = True,
    test_size: float = 0.2,
    save_model: bool = True
):
    """
    Retrain meta-learner with override-aware features.

    Args:
        training_db_path: Path to TurboMode training database
        output_path: Where to save new model (default: turbomode_models/meta_learner_v2)
        use_class_weights: Whether to use class weights to handle imbalance
        test_size: Validation split ratio
        save_model: Whether to save the trained model

    Returns:
        Dictionary with training results and model
    """
    print("=" * 80)
    print("RETRAINING META-LEARNER WITH OVERRIDE-AWARE FEATURES")
    print("=" * 80)

    # Default paths
    if training_db_path is None:
        training_db_path = Path(__file__).parent.parent / 'data' / 'turbomode.db'

    if output_path is None:
        output_path = Path(__file__).parent.parent / 'data' / 'turbomode_models' / 'meta_learner_v2'
    else:
        output_path = Path(output_path)

    # Load training data
    print(f"\n[STEP 1] Loading training data from {training_db_path}")

    # Get base model predictions and labels
    print("[STEP 2] Loading base model predictions...")

    # Model names in canonical order
    model_names = [
        'xgboost', 'xgboost_et', 'lightgbm', 'catboost',
        'xgboost_hist', 'xgboost_dart', 'xgboost_gblinear', 'xgboost_approx'
    ]

    # Load meta-learner training data
    # This assumes you have stored base model predictions
    # If not, you'll need to generate them first
    import sqlite3
    conn = sqlite3.connect(str(training_db_path))
    cursor = conn.cursor()

    # Check if meta_predictions table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='meta_predictions'
    """)

    if cursor.fetchone() is None:
        print("\n[ERROR] meta_predictions table not found!")
        print("You need to run the base models on training data first to generate predictions.")
        print("This creates the 24-feature matrix needed for meta-learner training.")
        return None

    # Load base model predictions
    query = """
        SELECT * FROM meta_predictions
        ORDER BY sample_id
    """

    df = pd.read_sql_query(query, conn)
    print(f"[OK] Loaded {len(df)} training samples")

    # Class distribution
    print(f"\nClass distribution:")
    print(df['label'].value_counts())
    print(f"  DOWN/SELL (0): {(df['label'] == 0).sum()} ({(df['label'] == 0).mean():.1%})")
    print(f"  NEUTRAL/HOLD (1): {(df['label'] == 1).sum()} ({(df['label'] == 1).mean():.1%})")
    print(f"  UP/BUY (2): {(df['label'] == 2).sum()} ({(df['label'] == 2).mean():.1%})")

    # Add override-aware features
    print("\n[STEP 3] Adding override-aware features...")
    df = add_override_aware_features(df, model_names)

    # Prepare feature matrix
    print("\n[STEP 4] Preparing feature matrix...")

    # Original 24 features
    feature_cols = []
    for model_name in model_names:
        feature_cols.extend([
            f'{model_name}_prob_down',
            f'{model_name}_prob_neutral',
            f'{model_name}_prob_up'
        ])

    # Override-aware features (per-model)
    for model_name in model_names:
        feature_cols.extend([
            f'{model_name}_asymmetry',
            f'{model_name}_max_directional',
            f'{model_name}_neutral_dominance'
        ])

    # Aggregate override-aware features
    feature_cols.extend([
        'avg_asymmetry',
        'max_asymmetry',
        'avg_max_directional',
        'avg_neutral_dominance',
        'models_favor_up',
        'models_favor_down',
        'directional_consensus'
    ])

    print(f"Total features: {len(feature_cols)}")
    print(f"  Original base model probs: 24")
    print(f"  Per-model override features: {8 * 3}")
    print(f"  Aggregate override features: 7")

    X = df[feature_cols].values
    y = df['label'].values

    # Train/validation split
    print(f"\n[STEP 5] Splitting data (test_size={test_size})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")

    # Calculate class weights
    if use_class_weights:
        from sklearn.utils.class_weight import compute_class_weight
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: w for i, w in enumerate(class_weights)}
        print(f"\nClass weights: {class_weight_dict}")
    else:
        class_weight_dict = None

    # Train LightGBM meta-learner
    print("\n[STEP 6] Training LightGBM meta-learner...")

    # Create feature names
    feature_names = feature_cols

    # Convert to DataFrame for LightGBM
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names)

    # LightGBM parameters
    params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'device': 'cpu'
    }

    # Create datasets
    train_data = lgb.Dataset(X_train_df, label=y_train, free_raw_data=False)
    val_data = lgb.Dataset(X_val_df, label=y_val, reference=train_data, free_raw_data=False)

    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=400,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )

    # Evaluate
    print("\n[STEP 7] Evaluating model...")

    # Training accuracy
    y_train_pred = np.argmax(model.predict(X_train_df), axis=1)
    train_acc = accuracy_score(y_train, y_train_pred)

    # Validation accuracy
    y_val_pred = np.argmax(model.predict(X_val_df), axis=1)
    val_acc = accuracy_score(y_val, y_val_pred)

    print(f"\nTraining accuracy: {train_acc:.2%}")
    print(f"Validation accuracy: {val_acc:.2%}")

    # Classification report
    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['SELL', 'HOLD', 'BUY']))

    # Confusion matrix
    print("\nValidation Confusion Matrix:")
    cm = confusion_matrix(y_val, y_val_pred)
    print("              Predicted")
    print("              SELL  HOLD  BUY")
    print(f"Actual SELL   {cm[0][0]:5d} {cm[0][1]:5d} {cm[0][2]:5d}")
    print(f"       HOLD   {cm[1][0]:5d} {cm[1][1]:5d} {cm[1][2]:5d}")
    print(f"       BUY    {cm[2][0]:5d} {cm[2][1]:5d} {cm[2][2]:5d}")

    # Feature importance
    print("\nTop 20 Most Important Features:")
    importance = model.feature_importance(importance_type='gain')
    feature_importance = list(zip(feature_names, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for i, (feat, imp) in enumerate(feature_importance[:20], 1):
        print(f"  {i:2d}. {feat:40s}: {imp:.0f}")

    # Save model
    if save_model:
        print(f"\n[STEP 8] Saving model to {output_path}")
        output_path.mkdir(parents=True, exist_ok=True)

        # Save LightGBM model
        model_file = output_path / 'meta_learner_v2.txt'
        model.save_model(str(model_file))
        print(f"  [OK] Saved LightGBM model: {model_file}")

        # Save metadata
        metadata = {
            'model_version': 'v2_override_aware',
            'created_at': datetime.now().isoformat(),
            'num_features': len(feature_cols),
            'feature_names': feature_names,
            'num_samples': len(df),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc),
            'hyperparameters': params,
            'class_distribution': {
                'down': int((y == 0).sum()),
                'hold': int((y == 1).sum()),
                'up': int((y == 2).sum())
            },
            'class_weights_used': use_class_weights,
            'feature_groups': {
                'base_model_probs': 24,
                'per_model_override': 24,
                'aggregate_override': 7,
                'total': len(feature_cols)
            }
        }

        metadata_file = output_path / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  [OK] Saved metadata: {metadata_file}")

        # Save feature importance
        importance_file = output_path / 'feature_importance.json'
        importance_dict = {feat: float(imp) for feat, imp in feature_importance}
        with open(importance_file, 'w') as f:
            json.dump(importance_dict, f, indent=2)
        print(f"  [OK] Saved feature importance: {importance_file}")

    print("\n" + "=" * 80)
    print("RETRAINING COMPLETE")
    print("=" * 80)
    print(f"Model saved to: {output_path}")
    print(f"Validation accuracy: {val_acc:.2%}")
    print("\nTo use this model, update the scanner to load 'meta_learner_v2' instead of 'meta_learner'")

    return {
        'model': model,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'feature_names': feature_names,
        'confusion_matrix': cm,
        'feature_importance': feature_importance
    }


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("META-LEARNER RETRAINING SCRIPT")
    print("This will retrain the final meta-learner with override-aware features")
    print("=" * 80)

    print("\n[WARNING] This requires base model predictions to be pre-computed.")
    print("If you haven't run the base models on training data yet, this will fail.")
    print("\nPress Ctrl+C to cancel, or wait 5 seconds to continue...")

    import time
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)

    # Run retraining
    result = retrain_meta_learner(
        use_class_weights=True,
        test_size=0.2,
        save_model=True
    )

    if result is not None:
        print(f"\n[OK] Retraining complete!")
        print(f"Validation accuracy: {result['val_accuracy']:.2%}")
    else:
        print("\n[FAIL] Retraining failed. Check error messages above.")
