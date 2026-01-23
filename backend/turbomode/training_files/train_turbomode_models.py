"""
Sector-Specific Model Training Orchestrator - PARALLEL OPTIMIZED

OPTIMIZATIONS:
1. Load ALL data once at start (instead of 11 separate database queries)
2. Filter data by sector in memory (fast numpy indexing)
3. Train multiple sectors in parallel using multiprocessing
4. Limit concurrent processes to avoid GPU memory conflicts

Expected speedup: 3-4x faster (4-5 hours -> 1-2 hours)

Trains 11 separate model sets (one per GICS sector):
- technology, financials, healthcare, industrials, consumer_discretionary
- communication_services, consumer_staples, energy, materials, utilities, real_estate

Each sector gets:
- 8 base models (XGBoost variants, LightGBM, CatBoost)
- 1 meta-learner
- Models saved to: C:\StockApp\backend\turbomode\models\trained\{sector}\
"""
import os
import sys
import json
import numpy as np
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import multiprocessing as mp
from functools import partial

warnings.filterwarnings('ignore')

# Add project root to path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from backend.turbomode.training_symbols import TRAINING_SYMBOLS
from backend.turbomode.feature_list import FEATURE_LIST, features_to_array

# FAST MODE MODELS - Tree-based models optimized for speed
from backend.turbomode.models.xgboost_hist_model import XGBoostHistModel
from backend.turbomode.models.xgboost_gblinear_model import XGBoostGBLinearModel
from backend.turbomode.models.lightgbm_model import LightGBMModel
from backend.turbomode.models.catboost_model import CatBoostModel
from backend.turbomode.models.random_forest_model import RandomForestModel
from backend.turbomode.models.meta_learner import MetaLearner

MODEL_BASE_PATH = r'C:\StockApp\backend\turbomode\models\trained'
DB_PATH = r'C:\StockApp\backend\data\turbomode.db'

# FAST MODE: 5 base models (removed slow XGBoost variants)
BASE_MODELS = [
    ('lightgbm', LightGBMModel, 'lightgbm'),
    ('catboost', CatBoostModel, 'catboost'),
    ('xgboost_hist', XGBoostHistModel, 'xgboost_hist'),
    ('xgboost_gblinear', XGBoostGBLinearModel, 'xgboost_gblinear'),
    ('random_forest', RandomForestModel, 'random_forest'),
]


def get_sector_symbols(sector: str) -> List[str]:
    """Get all training symbols for a sector"""
    if sector not in TRAINING_SYMBOLS:
        return []

    symbols = []
    for cap_category, symbol_list in TRAINING_SYMBOLS[sector].items():
        symbols.extend(symbol_list)

    return symbols


def load_all_training_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ALL training data once from database

    Returns:
        X: Feature matrix (N, 179)
        y: Labels (N,)
        symbols: Symbol array (N,) for filtering by sector
    """
    print("\n" + "="*80, flush=True)
    print("LOADING ALL TRAINING DATA (1.24M samples)", flush=True)
    print("="*80, flush=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = """
        SELECT entry_features_json, outcome, symbol
        FROM trades
        WHERE trade_type = 'backtest'
        AND entry_features_json IS NOT NULL
        AND outcome IS NOT NULL
        AND outcome IN ('buy', 'hold', 'sell')
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    print(f"[DATA] Loaded {len(rows):,} samples from database")

    # Parse features and labels
    feature_list = []
    label_list = []
    symbol_list = []

    label_map = {'sell': 0, 'hold': 1, 'buy': 2}
    exclude_keys = {'feature_count', 'symbol', 'last_price', 'last_volume', 'timestamp', 'error'}

    for i, (features_json, outcome, symbol) in enumerate(rows):
        try:
            features_dict = json.loads(features_json)

            # Convert to feature array
            feature_array = features_to_array(features_dict)

            feature_list.append(feature_array)
            label_list.append(label_map[outcome])
            symbol_list.append(symbol)

        except Exception as e:
            continue

    X = np.array(feature_list, dtype=np.float32)
    y = np.array(label_list, dtype=np.int32)
    symbols = np.array(symbol_list)

    print(f"[OK] Parsed into arrays:")
    print(f"     X shape: {X.shape}")
    print(f"     y shape: {y.shape}")
    print(f"     symbols shape: {symbols.shape}")

    return X, y, symbols


def filter_data_by_sector(X: np.ndarray, y: np.ndarray, symbols: np.ndarray,
                          sector: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter data by sector using numpy indexing (FAST)

    Returns:
        X_train, y_train, X_val, y_val
    """
    sector_symbols = get_sector_symbols(sector)

    # Create boolean mask (vectorized, FAST)
    mask = np.isin(symbols, sector_symbols)

    X_sector = X[mask]
    y_sector = y[mask]

    # Split 80/20
    split_idx = int(0.8 * len(X_sector))

    X_train = X_sector[:split_idx]
    y_train = y_sector[:split_idx]
    X_val = X_sector[split_idx:]
    y_val = y_sector[split_idx:]

    return X_train, y_train, X_val, y_val


def train_sector_base_models(sector: str, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray, horizon_days: int = None) -> Dict[str, np.ndarray]:
    """
    Train all base models for a sector

    Args:
        sector: Sector name
        X_train, y_train, X_val, y_val: Training and validation data
        horizon_days: Horizon window (1 or 2 days). If None, uses default path structure.

    Returns:
        Dict mapping model_name -> validation predictions
    """
    # Build sector model path with optional horizon subdirectory
    if horizon_days is not None:
        sector_model_path = os.path.join(MODEL_BASE_PATH, sector, f"{horizon_days}d")
    else:
        sector_model_path = os.path.join(MODEL_BASE_PATH, sector)

    os.makedirs(sector_model_path, exist_ok=True)

    base_predictions = {}

    for i, (model_name, model_class, model_dir) in enumerate(BASE_MODELS, 1):
        try:
            # Create ABSOLUTE model output directory (uses sector_model_path which includes horizon if specified)
            model_output_dir = os.path.join(sector_model_path, model_dir)
            os.makedirs(model_output_dir, exist_ok=True)

            # Initialize model with ABSOLUTE output path
            model = model_class(model_path=model_output_dir)

            # Train
            start_time = datetime.now()
            model.train(X_train, y_train, X_val, y_val)
            train_time = (datetime.now() - start_time).total_seconds()

            # Get predictions
            val_preds = model.predict_proba(X_val)

            # Evaluate
            val_pred_classes = np.argmax(val_preds, axis=1)
            val_accuracy = np.mean(val_pred_classes == y_val)

            # Save model (no argument needed - path already set in constructor)
            model.save()

            # Store predictions for meta-learner
            base_predictions[model_name] = val_preds

            print(f"[{sector}] {model_name}: {val_accuracy:.4f} ({train_time:.1f}s)")

        except Exception as e:
            print(f"[{sector}] {model_name}: ERROR - {e}")
            continue

    return base_predictions


def add_override_features_to_predictions(predictions: List[Dict]) -> List[Dict]:
    """
    Add override-aware features to base predictions (31 additional features).

    Takes 24 base features (8 models x 3 probs) and adds:
    - 24 per-model features (asymmetry, max_directional, neutral_dominance for each model)
    - 7 aggregate features (avg_asymmetry, consensus metrics, etc.)

    Args:
        predictions: List of prediction dicts with format:
                    {'model_name': np.array([prob_down, prob_neutral, prob_up]), ...}

    Returns:
        List of enhanced prediction dicts with 55 features total
    """
    model_names = ['xgboost', 'xgboost_et', 'lightgbm', 'catboost',
                   'xgboost_hist', 'xgboost_dart', 'xgboost_gblinear', 'xgboost_approx']

    enhanced_predictions = []

    for pred_dict in predictions:
        enhanced = pred_dict.copy()

        # Add per-model override features
        for model_name in model_names:
            probs = pred_dict[model_name]
            prob_up = probs[2]
            prob_down = probs[0]
            prob_neutral = probs[1]

            # Asymmetry between buy and sell
            enhanced[f'{model_name}_asymmetry'] = np.abs(prob_up - prob_down)

            # Max directional confidence
            enhanced[f'{model_name}_max_directional'] = np.maximum(prob_up, prob_down)

            # Neutral dominance
            enhanced[f'{model_name}_neutral_dominance'] = prob_neutral - np.maximum(prob_up, prob_down)

        # Add aggregate override features
        asymmetries = [enhanced[f'{m}_asymmetry'] for m in model_names]
        max_directionals = [enhanced[f'{m}_max_directional'] for m in model_names]
        neutral_dominances = [enhanced[f'{m}_neutral_dominance'] for m in model_names]

        enhanced['avg_asymmetry'] = np.mean(asymmetries)
        enhanced['max_asymmetry'] = np.max(asymmetries)
        enhanced['avg_max_directional'] = np.mean(max_directionals)
        enhanced['avg_neutral_dominance'] = np.mean(neutral_dominances)

        # Consensus features
        models_favor_up = sum(1 for m in model_names if pred_dict[m][2] > pred_dict[m][0])
        models_favor_down = sum(1 for m in model_names if pred_dict[m][0] > pred_dict[m][2])

        enhanced['models_favor_up'] = models_favor_up
        enhanced['models_favor_down'] = models_favor_down
        enhanced['directional_consensus'] = np.abs(models_favor_up - models_favor_down) / len(model_names)

        enhanced_predictions.append(enhanced)

    return enhanced_predictions


def train_sector_meta_learner(sector: str, base_predictions: Dict[str, np.ndarray],
                               y_val: np.ndarray, horizon_days: int = None) -> float:
    """Train meta-learner v2 for a sector with 55 override-aware features

    Args:
        sector: Sector name
        base_predictions: Dict mapping model_name -> validation predictions
        y_val: Validation labels
        horizon_days: Horizon window (1 or 2 days). If None, uses default path structure.
    """

    # Convert base predictions to list of dicts format
    num_samples = len(y_val)
    predictions_list = []

    for i in range(num_samples):
        pred_dict = {}
        for model_name, preds in base_predictions.items():
            pred_dict[model_name] = preds[i]
        predictions_list.append(pred_dict)

    # Add override-aware features (31 additional, total 55)
    enhanced_predictions = add_override_features_to_predictions(predictions_list)

    # Initialize meta-learner v2 with ABSOLUTE path (includes horizon subdirectory if specified)
    if horizon_days is not None:
        meta_output_dir = os.path.join(MODEL_BASE_PATH, sector, f"{horizon_days}d", 'meta_learner_v2')
    else:
        meta_output_dir = os.path.join(MODEL_BASE_PATH, sector, 'meta_learner_v2')

    meta_learner = MetaLearner(model_path=meta_output_dir, use_gpu=True)

    # Train with enhanced predictions (55 features)
    start_time = datetime.now()
    metrics = meta_learner.train(enhanced_predictions, y_val)
    train_time = (datetime.now() - start_time).total_seconds()

    # Evaluate
    meta_accuracy = metrics.get('train_accuracy', 0.0)

    # Save
    os.makedirs(meta_output_dir, exist_ok=True)
    meta_learner.save()

    print(f"[{sector}] meta_learner_v2: {meta_accuracy:.4f} ({train_time:.1f}s) [55 features]")

    return meta_accuracy


def train_single_sector_worker(sector: str, X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray, horizon_days: int = None) -> Dict:
    """
    Worker function to train one sector (used by multiprocessing)
    Receives pre-filtered sector data - does NOT load from disk

    Args:
        sector: Sector name
        X_train: Training features for this sector only
        y_train: Training labels for this sector only
        X_val: Validation features for this sector only
        y_val: Validation labels for this sector only
        horizon_days: Horizon window (1 or 2 days). If None, uses default path structure.
    """
    sector_start = datetime.now()

    print(f"\n[{sector.upper()}] Starting training...")

    num_samples = len(X_train)
    print(f"[{sector.upper()}] Data: {num_samples:,} train, {len(X_val):,} val")

    if num_samples < 100:
        print(f"[{sector.upper()}] WARNING: Insufficient data, skipping")
        return {'sector': sector, 'status': 'skipped', 'reason': 'insufficient_data'}

    # Train base models
    base_predictions = train_sector_base_models(sector, X_train, y_train, X_val, y_val, horizon_days)

    if len(base_predictions) < 3:
        print(f"[{sector.upper()}] WARNING: Only {len(base_predictions)} models trained")
        return {'sector': sector, 'status': 'partial', 'base_models': len(base_predictions)}

    # Train meta-learner
    meta_accuracy = train_sector_meta_learner(sector, base_predictions, y_val, horizon_days)

    sector_time = (datetime.now() - sector_start).total_seconds() / 60

    result = {
        'sector': sector,
        'status': 'completed',
        'num_symbols': len(get_sector_symbols(sector)),
        'num_samples': num_samples,
        'num_base_models': len(base_predictions),
        'meta_accuracy': float(meta_accuracy),
        'training_time_minutes': round(sector_time, 2)
    }

    print(f"[{sector.upper()}] COMPLETE - {sector_time:.1f} min")

    return result


def train_all_sectors_parallel(max_workers: int = 1, horizon_days: int = None, thresholds: Dict[str, float] = None):
    """
    Train models for all 11 sectors with CORRECT architecture

    NOTE: Despite the function name, this uses SEQUENTIAL execution on Windows
          to avoid multiprocessing serialization issues with large numpy arrays.
          Since max_workers=1 anyway (to avoid GPU thrashing), no performance loss.

    CRITICAL: Data is loaded ONCE in parent, pre-split by sector, then trained sequentially

    Args:
        max_workers: Ignored (kept for API compatibility, always runs sequentially)
        horizon_days: Horizon window for label computation (1 or 2 days). If None, use pre-computed 'outcome' column.
        thresholds: Dict with 'buy' and 'sell' thresholds (e.g., {'buy': 0.10, 'sell': -0.10}).
                   Required if horizon_days is specified.
    """
    overall_start = datetime.now()

    print("\n\n")
    print("="*80)
    print("SECTOR-SPECIFIC MODEL TRAINING - SEQUENTIAL MODE (WINDOWS COMPATIBLE)")
    print("="*80)
    print(f"\nStarting: {overall_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total sectors: 11")
    print(f"Mode: Sequential (avoids Windows multiprocessing issues)")

    if horizon_days is not None:
        print(f"\nHorizon: {horizon_days}d (DYNAMIC LABEL COMPUTATION)")
        print(f"Thresholds: BUY >= {thresholds['buy']:.2%}, SELL <= {thresholds['sell']:.2%}")
    else:
        print(f"\nHorizon: Pre-computed (from 'outcome' column)")

    print()

    # STEP 1: Load ALL data ONCE in parent process
    print("\n" + "="*80)
    print("STEP 1: LOADING ALL TRAINING DATA (ONCE, IN PARENT)")
    print("="*80)

    from backend.turbomode.turbomode_training_loader import TurboModeTrainingDataLoader
    loader = TurboModeTrainingDataLoader()

    # Load ALL training data (no sector filtering, returns symbols)
    X_all, y_all, symbols_all = loader.load_training_data(
        return_split=False,
        horizon_days=horizon_days,
        thresholds=thresholds
    )

    print(f"[OK] Loaded {len(X_all):,} total samples")
    print(f"    Features: {X_all.shape[1]}")
    print()

    # STEP 2: Pre-split by sector ONCE in parent (VECTORIZED)
    print("="*80)
    print("STEP 2: PRE-SPLITTING DATA BY SECTOR (VECTORIZED, IN PARENT)")
    print("="*80)

    sectors = list(TRAINING_SYMBOLS.keys())
    sector_data = {}

    for sector in sectors:
        sector_symbols = get_sector_symbols(sector)

        # Vectorized boolean mask (NO disk I/O, NO per-sector loading)
        mask = np.isin(symbols_all, sector_symbols)

        # Filter by sector using vectorized indexing
        X_sector = X_all[mask]
        y_sector = y_all[mask]

        # Split into train/val (80/20) - vectorized
        split_idx = int(0.8 * len(X_sector))
        sector_data[sector] = {
            'X_train': X_sector[:split_idx],
            'y_train': y_sector[:split_idx],
            'X_val': X_sector[split_idx:],
            'y_val': y_sector[split_idx:]
        }

        print(f"  [{sector:25s}] {len(X_sector):,} samples ({len(sector_data[sector]['X_train']):,} train, {len(sector_data[sector]['X_val']):,} val)")

    print()

    # STEP 3: Train sectors SEQUENTIALLY (no multiprocessing - Windows compatible)
    print("="*80)
    print("STEP 3: TRAINING SECTORS SEQUENTIALLY")
    print("="*80)

    results = []
    for i, sector in enumerate(sectors, 1):
        print(f"\n[SECTOR {i}/11: {sector.upper()}]")

        try:
            result = train_single_sector_worker(
                sector,
                sector_data[sector]['X_train'],
                sector_data[sector]['y_train'],
                sector_data[sector]['X_val'],
                sector_data[sector]['y_val'],
                horizon_days
            )
            results.append(result)
        except Exception as e:
            print(f"[{sector.upper()}] FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'sector': sector,
                'status': 'failed',
                'error': str(e)
            })

    overall_time = (datetime.now() - overall_start).total_seconds() / 60

    # Save summary
    summary = {
        'started': overall_start.isoformat(),
        'completed': datetime.now().isoformat(),
        'total_time_minutes': round(overall_time, 2),
        'parallel_workers': max_workers,
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
    print(f"Speedup: {max_workers}x parallel workers")
    print(f"\nSummary saved to: {summary_path}")
    print()

    for result in results:
        sector_name = result['sector'].replace('_', ' ').title()
        if result['status'] == 'completed':
            print(f"  [{result['sector']:25s}] {result['training_time_minutes']:5.1f} min - {result['num_base_models']} models")
        else:
            print(f"  [{result['sector']:25s}] {result['status']}")

    print("\n" + "="*80)
    print("READY FOR INFERENCE")
    print("="*80)


if __name__ == '__main__':
    # Train with 1 worker (sequential to avoid GPU memory thrashing)
    train_all_sectors_parallel(max_workers=1)
