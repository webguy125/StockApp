
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
MetaLearner-Only Retraining Script

CRITICAL FIX: Class mapping inversion correction
- Base models are trained correctly (0=SELL, 1=HOLD, 2=BUY)
- Only MetaLearner needs retraining after fixing class mapping in inference

This script:
1. Loads existing base models for each sector (5 models per sector)
2. Regenerates out-of-fold (OOF) predictions using the existing base models
3. Retrains ONLY the MetaLearner using corrected understanding
4. Saves new MetaLearner models for all 11 sectors

Base models remain unchanged - they were already trained with correct labels.
"""

import os
import time
import pickle
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import logging

# Import infrastructure
from backend.turbomode.core_engine.sector_batch_trainer import load_sector_data_once
from backend.turbomode.core_engine.training_symbols import TRAINING_SYMBOLS
from backend.turbomode.core_engine.model_registry import (
    BASE_MODELS,
    META_LEARNER,
    get_model_path
)
from backend.turbomode.core_engine.model_loader import load_base_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# All 11 sectors
ALL_SECTORS = [
    'technology',
    'financials',
    'healthcare',
    'consumer_discretionary',
    'communication_services',
    'industrials',
    'consumer_staples',
    'energy',
    'materials',
    'real_estate',
    'utilities'
]


def get_symbols_by_sector(sector: str):
    """Get all symbols for a sector."""
    if sector not in TRAINING_SYMBOLS:
        raise ValueError(f"Unknown sector: {sector}")

    sector_data = TRAINING_SYMBOLS[sector]
    symbols = []

    if 'large_cap' in sector_data:
        symbols.extend(sector_data['large_cap'])
    if 'mid_cap' in sector_data:
        symbols.extend(sector_data['mid_cap'])
    if 'small_cap' in sector_data:
        symbols.extend(sector_data['small_cap'])

    return symbols


def generate_oof_predictions_from_saved_models(
    sector: str,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5
) -> np.ndarray:
    """
    Generate out-of-fold predictions using EXISTING saved base models.

    Instead of training new base models, we load the existing ones and
    use cross-validation to generate OOF predictions for MetaLearner training.

    Args:
        sector: Sector name
        X: Feature matrix (N, n_features)
        y: Target labels (N,) - [0=SELL, 1=HOLD, 2=BUY]
        n_splits: Number of CV folds (default 5)

    Returns:
        oof_predictions: Stacked predictions (N, 15) - 5 models x 3 classes each
    """
    logger.info(f"  Generating OOF predictions using saved base models ({n_splits}-fold CV)...")

    # Load existing base models
    logger.info(f"  Loading saved base models for {sector}...")
    base_models_dict = load_base_models(sector)
    base_model_list = [base_models_dict[name] for name in BASE_MODELS]
    logger.info(f"    ✓ Loaded {len(base_model_list)} base models")

    n_samples = X.shape[0]
    n_classes = 3
    n_models = len(BASE_MODELS)

    # Initialize OOF prediction array (N, 15) - 5 models x 3 classes
    oof_predictions = np.zeros((n_samples, n_models * n_classes))

    # K-fold cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"    Fold {fold_idx + 1}/{n_splits}...")

        X_val_fold = X[val_idx]

        # Predict on validation fold using saved base models
        for model_idx, model in enumerate(base_model_list):
            probs = model.predict_proba(X_val_fold)  # shape: (n_val, 3)

            # Store in OOF array
            start_col = model_idx * n_classes
            end_col = start_col + n_classes
            oof_predictions[val_idx, start_col:end_col] = probs

    logger.info("    ✓ Out-of-fold predictions generated using saved models")

    return oof_predictions


def retrain_meta_learner(
    oof_predictions: np.ndarray,
    y: np.ndarray
) -> LogisticRegression:
    """
    Retrain MetaLearner on stacked out-of-fold predictions.

    Args:
        oof_predictions: Stacked OOF predictions (N, 15)
        y: Target labels (N,) - [0=SELL, 1=HOLD, 2=BUY]

    Returns:
        Trained LogisticRegression meta-learner
    """
    logger.info("  Retraining MetaLearner (LogisticRegression)...")

    meta_learner = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        random_state=42,
        verbose=0
    )

    meta_learner.fit(oof_predictions, y)
    logger.info("    ✓ MetaLearner retrained")

    return meta_learner


def retrain_sector_meta_learner(
    sector: str,
    db_path: str
) -> str:
    """
    Retrain ONLY the MetaLearner for a sector.

    Args:
        sector: Sector name
        db_path: Path to turbomode database

    Returns:
        Path to saved MetaLearner model
    """
    logger.info(f"\n[RETRAIN META] {sector.upper()}")

    # Get sector symbols
    sector_symbols = get_symbols_by_sector(sector)
    logger.info(f"  Symbols: {len(sector_symbols)}")

    # Load sector training data
    logger.info(f"  Loading training data...")
    X_sector, labels_dict, trade_ids = load_sector_data_once(db_path, sector_symbols)

    if len(X_sector) == 0:
        raise ValueError(f"No training data available for {sector}")

    # Build label vector
    y_sector = np.array([labels_dict[tid] for tid in trade_ids], dtype=np.int32)

    logger.info(f"  Training samples: {X_sector.shape[0]}")
    logger.info(f"  Features: {X_sector.shape[1]}")
    logger.info(f"  Classes: SELL={np.sum(y_sector==0)}, HOLD={np.sum(y_sector==1)}, BUY={np.sum(y_sector==2)}")

    # Generate OOF predictions using existing base models
    oof_predictions = generate_oof_predictions_from_saved_models(
        sector, X_sector, y_sector, n_splits=5
    )

    # Retrain MetaLearner
    meta_learner = retrain_meta_learner(oof_predictions, y_sector)

    # Save MetaLearner
    logger.info("  Saving MetaLearner...")
    meta_path = get_model_path(sector, META_LEARNER)
    with open(meta_path, 'wb') as f:
        pickle.dump(meta_learner, f)
    logger.info(f"    ✓ Saved {META_LEARNER} to {meta_path}")

    logger.info(f"[SUCCESS] MetaLearner retrained for {sector}")

    return meta_path


def retrain_all_meta_learners(db_path: str):
    """
    Retrain MetaLearner for all 11 sectors.

    Args:
        db_path: Path to turbomode database
    """
    print("\n" + "=" * 80)
    print("META-LEARNER RETRAINING (CLASS MAPPING FIX)")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sectors: {len(ALL_SECTORS)}")
    print(f"Action: Retrain MetaLearner only (base models unchanged)")
    print(f"Reason: Fix class mapping inversion (0=SELL, 1=HOLD, 2=BUY)")
    print("=" * 80)
    print()

    global_start = time.time()

    all_results = {}

    for i, sector in enumerate(ALL_SECTORS, 1):
        sector_start = time.time()

        try:
            meta_path = retrain_sector_meta_learner(sector, db_path)
            sector_time = time.time() - sector_start

            all_results[sector] = {
                'status': 'completed',
                'meta_path': meta_path,
                'time': sector_time
            }

            print(f"\n[{i}/{len(ALL_SECTORS)}] {sector.upper()} COMPLETE [OK]")
            print(f"Time: {sector_time:.1f} seconds")

        except Exception as e:
            import traceback
            print(f"\n[{i}/{len(ALL_SECTORS)}] {sector.upper()} FAILED [X]")
            print(f"Exception: {e}")
            traceback.print_exc()
            all_results[sector] = {'status': 'failed', 'error': str(e)}

        # Progress update
        elapsed = time.time() - global_start
        sectors_done = i
        sectors_remaining = len(ALL_SECTORS) - i
        avg_time_per_sector = elapsed / sectors_done
        estimated_remaining = avg_time_per_sector * sectors_remaining

        print(f"\n[PROGRESS] {sectors_done}/{len(ALL_SECTORS)} sectors complete")
        print(f"[PROGRESS] Elapsed: {elapsed/60:.1f} min | Estimated remaining: {estimated_remaining/60:.1f} min")
        print()

    # Final summary
    total_time = time.time() - global_start

    print("\n" + "=" * 80)
    print("META-LEARNER RETRAINING COMPLETE")
    print("=" * 80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print()

    # Results summary
    successful = sum(1 for r in all_results.values() if r.get('status') == 'completed')
    failed = sum(1 for r in all_results.values() if r.get('status') == 'failed')

    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Sectors processed: {len(ALL_SECTORS)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print()

    print("Per-Sector Results:")
    print("-" * 80)
    for sector in ALL_SECTORS:
        result = all_results.get(sector, {})
        status = result.get('status', 'unknown')

        if status == 'completed':
            time_sec = result['time']
            print(f"  {sector:30s} [OK] Completed in {time_sec:.1f} sec")
        else:
            error = result.get('error', 'Unknown error')
            print(f"  {sector:30s} [FAILED]: {error}")

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Run overnight_scanner.py to validate corrected predictions")
    print("2. Verify BUY/SELL distribution is no longer 99.6% SELL")
    print("3. Check that HOLD signals appear (should be ~95% of predictions)")
    print("=" * 80)

    return all_results


if __name__ == '__main__':
    # Database path
    backend_dir = str(project_root / "backend")
    db_path = os.path.join(backend_dir, "data", "turbomode.db")

    # Retrain all MetaLearners
    results = retrain_all_meta_learners(db_path)
