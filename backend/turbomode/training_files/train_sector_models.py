"""
Sector-Specific Model Training Orchestrator

Trains 11 separate model sets (one per GICS sector):
- technology (38 stocks)
- financials (26 stocks)
- healthcare (35 stocks)
- industrials (38 stocks)
- consumer_discretionary (29 stocks)
- communication_services (12 stocks)
- consumer_staples (7 stocks)
- energy (12 stocks)
- materials (9 stocks)
- utilities (12 stocks)
- real_estate (10 stocks)

Each sector gets:
- 8 base models (XGBoost variants, LightGBM, CatBoost)
- 1 meta-learner
- Models saved to: C:\StockApp\backend\turbomode\models\trained\{sector}\

Architecture:
- Market cap tier included as feature (0=small, 1=mid, 2=large)
- Each sector learns its own behavioral patterns
- Training time: ~25-35 min per sector = 4-5 hours total
"""
import os
import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from backend.turbomode.turbomode_training_loader import TurboModeTrainingDataLoader
from backend.turbomode.training_symbols import TRAINING_SYMBOLS, SECTOR_CODES

# Tree-based models
from backend.turbomode.models.xgboost_model import XGBoostModel
from backend.turbomode.models.xgboost_et_model import XGBoostETModel
from backend.turbomode.models.xgboost_hist_model import XGBoostHistModel
from backend.turbomode.models.xgboost_dart_model import XGBoostDARTModel
from backend.turbomode.models.xgboost_gblinear_model import XGBoostGBLinearModel
from backend.turbomode.models.xgboost_approx_model import XGBoostApproxModel
from backend.turbomode.models.lightgbm_model import LightGBMModel
from backend.turbomode.models.catboost_model import CatBoostModel
from backend.turbomode.models.meta_learner import MetaLearner

MODEL_BASE_PATH = r'C:\StockApp\backend\turbomode\models\trained'

BASE_MODELS = [
    ('xgboost', XGBoostModel, 'xgboost'),
    ('xgboost_et', XGBoostETModel, 'xgboost_et'),
    ('lightgbm', LightGBMModel, 'lightgbm'),
    ('catboost', CatBoostModel, 'catboost'),
    ('xgboost_hist', XGBoostHistModel, 'xgboost_hist'),
    ('xgboost_dart', XGBoostDARTModel, 'xgboost_dart'),
    ('xgboost_gblinear', XGBoostGBLinearModel, 'xgboost_gblinear'),
    ('xgboost_approx', XGBoostApproxModel, 'xgboost_approx'),
]


def get_sector_symbols(sector: str) -> List[str]:
    """Get all training symbols for a sector"""
    if sector not in TRAINING_SYMBOLS:
        return []

    symbols = []
    for cap_category, symbol_list in TRAINING_SYMBOLS[sector].items():
        symbols.extend(symbol_list)

    return symbols


def load_sector_training_data(sector: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load training data filtered by sector

    Returns:
        X_train, y_train, X_val, y_val, num_samples
    """
    print(f"\n{'='*80}")
    print(f"LOADING TRAINING DATA FOR SECTOR: {sector.upper()}")
    print(f"{'='*80}")

    sector_symbols = get_sector_symbols(sector)
    print(f"Sector symbols: {len(sector_symbols)}")
    print(f"  {', '.join(sorted(sector_symbols)[:10])}{'...' if len(sector_symbols) > 10 else ''}")

    loader = TurboModeTrainingDataLoader()
    X_train, y_train, X_val, y_val = loader.load_training_data(
        symbols_filter=sector_symbols,  # Only load data for this sector
        return_split=True  # Return train/val split (80/20)
    )

    print(f"\nSector {sector} data shape:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")

    return X_train, y_train, X_val, y_val, X_train.shape[0]


def train_sector_base_models(sector: str, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Train all base models for a sector

    Returns:
        Dict mapping model_name -> validation predictions
    """
    print(f"\n{'='*80}")
    print(f"TRAINING BASE MODELS FOR SECTOR: {sector.upper()}")
    print(f"{'='*80}\n")

    sector_model_path = os.path.join(MODEL_BASE_PATH, sector)
    os.makedirs(sector_model_path, exist_ok=True)

    base_predictions = {}

    for i, (model_name, model_class, model_dir) in enumerate(BASE_MODELS, 1):
        print(f"\n[{i}/{len(BASE_MODELS)}] Training {model_name}...")
        print("-" * 60)

        try:
            # Initialize model
            model = model_class()

            # Train
            start_time = datetime.now()
            model.train(X_train, y_train, X_val, y_val)
            train_time = (datetime.now() - start_time).total_seconds()

            # Get predictions
            val_preds = model.predict_proba(X_val)

            # Evaluate
            val_pred_classes = np.argmax(val_preds, axis=1)
            val_accuracy = np.mean(val_pred_classes == y_val)

            # Save model
            model_output_dir = os.path.join(sector_model_path, model_dir)
            os.makedirs(model_output_dir, exist_ok=True)
            model.save(model_output_dir)

            # Store predictions for meta-learner
            base_predictions[model_name] = val_preds

            print(f"  Training time: {train_time:.1f}s")
            print(f"  Val accuracy: {val_accuracy:.4f}")
            print(f"  Saved to: {model_output_dir}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    return base_predictions


def train_sector_meta_learner(sector: str, base_predictions: Dict[str, np.ndarray],
                               y_val: np.ndarray) -> None:
    """Train meta-learner for a sector"""
    print(f"\n{'='*80}")
    print(f"TRAINING META-LEARNER FOR SECTOR: {sector.upper()}")
    print(f"{'='*80}\n")

    # Stack base model predictions
    X_meta = np.concatenate([preds for preds in base_predictions.values()], axis=1)

    print(f"Meta-learner input shape: {X_meta.shape}")
    print(f"  {len(base_predictions)} base models × 2 classes = {X_meta.shape[1]} features")

    # Initialize meta-learner
    meta_learner = MetaLearner()

    # Train
    start_time = datetime.now()
    meta_learner.train(X_meta, y_val)
    train_time = (datetime.now() - start_time).total_seconds()

    # Evaluate
    meta_preds = meta_learner.predict(X_meta)
    meta_accuracy = np.mean(meta_preds == y_val)

    # Save
    meta_output_dir = os.path.join(MODEL_BASE_PATH, sector, 'meta_learner')
    os.makedirs(meta_output_dir, exist_ok=True)
    meta_learner.save(meta_output_dir)

    print(f"  Training time: {train_time:.1f}s")
    print(f"  Val accuracy: {meta_accuracy:.4f}")
    print(f"  Saved to: {meta_output_dir}")


def train_single_sector(sector: str) -> Dict:
    """Train all models for a single sector"""
    sector_start = datetime.now()

    print(f"\n\n{'#'*80}")
    print(f"# SECTOR: {sector.upper()}")
    print(f"{'#'*80}")

    # Load data
    X_train, y_train, X_val, y_val, num_samples = load_sector_training_data(sector)

    if num_samples < 100:
        print(f"\nWARNING: Only {num_samples} training samples for {sector}. Skipping.")
        return {'sector': sector, 'status': 'skipped', 'reason': 'insufficient_data'}

    # Train base models
    base_predictions = train_sector_base_models(sector, X_train, y_train, X_val, y_val)

    if len(base_predictions) < 3:
        print(f"\nWARNING: Only {len(base_predictions)} base models trained. Skipping meta-learner.")
        return {'sector': sector, 'status': 'partial', 'base_models': len(base_predictions)}

    # Train meta-learner
    train_sector_meta_learner(sector, base_predictions, y_val)

    sector_time = (datetime.now() - sector_start).total_seconds() / 60

    result = {
        'sector': sector,
        'status': 'completed',
        'num_symbols': len(get_sector_symbols(sector)),
        'num_samples': num_samples,
        'num_base_models': len(base_predictions),
        'training_time_minutes': round(sector_time, 2)
    }

    print(f"\n{'='*80}")
    print(f"SECTOR {sector.upper()} COMPLETE")
    print(f"  Time: {sector_time:.1f} minutes")
    print(f"  Models: {len(base_predictions)} base + 1 meta = {len(base_predictions) + 1} total")
    print(f"{'='*80}")

    return result


def train_all_sectors():
    """Train models for all 11 sectors"""
    overall_start = datetime.now()

    print("\n\n")
    print("="*80)
    print("SECTOR-SPECIFIC MODEL TRAINING")
    print("="*80)
    print(f"\nStarting: {overall_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total sectors: 11")
    print(f"Estimated time: 4-5 hours")
    print()

    sectors = list(TRAINING_SYMBOLS.keys())
    results = []

    for i, sector in enumerate(sectors, 1):
        print(f"\n\nPROGRESS: Sector {i}/{len(sectors)}")
        result = train_single_sector(sector)
        results.append(result)

    overall_time = (datetime.now() - overall_start).total_seconds() / 60

    # Save summary
    summary = {
        'started': overall_start.isoformat(),
        'completed': datetime.now().isoformat(),
        'total_time_minutes': round(overall_time, 2),
        'sectors_trained': len([r for r in results if r['status'] == 'completed']),
        'sectors_skipped': len([r for r in results if r['status'] != 'completed']),
        'results': results
    }

    summary_path = os.path.join(MODEL_BASE_PATH, 'sector_training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n\n")
    print("="*80)
    print("SECTOR TRAINING COMPLETE")
    print("="*80)
    print(f"\nTotal time: {overall_time:.1f} minutes ({overall_time/60:.1f} hours)")
    print(f"Sectors completed: {summary['sectors_trained']}/{len(sectors)}")
    print(f"\nSummary saved to: {summary_path}")
    print()

    for result in results:
        status_symbol = "✓" if result['status'] == 'completed' else "✗"
        sector_name = result['sector'].replace('_', ' ').title()
        if result['status'] == 'completed':
            print(f"  {status_symbol} {sector_name:30s} {result['training_time_minutes']:5.1f} min")
        else:
            print(f"  {status_symbol} {sector_name:30s} {result['status']}")

    print("\n" + "="*80)
    print("READY FOR INFERENCE")
    print("="*80)
    print("\nNext steps:")
    print("1. Update overnight_scanner.py to use sector-specific models")
    print("2. Run scanner to generate predictions")


if __name__ == '__main__':
    train_all_sectors()
