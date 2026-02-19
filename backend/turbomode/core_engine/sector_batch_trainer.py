
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
Sector-Level 14-Day Swing Trade Training Module
Multi-Model Ensemble for Directional Stock/Options Trading

ARCHITECTURE:
- Trains ONE ensemble per sector (5 base models + 1 MetaLearner)
- Uses label_14d_swing (14-day horizon, ±6% threshold)
- Model directory: models/<sector>/*.pkl
- Total models: 66 (6 per sector × 11 sectors)

LABEL SEMANTICS (14-Day Swing):
- 0 = SELL: 14-day return <= -6% (bearish swing)
- 1 = HOLD: 14-day return between -6% and +6% (no edge)
- 2 = BUY: 14-day return >= +6% (bullish swing)

PERFORMANCE:
- Training time: ~2-3 hours for all 11 sectors
- Clean, deterministic, swing trade architecture

Author: TurboMode Team
Date: 2026-01-21
"""

import os
import time
import numpy as np
import json
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging
import sqlite3

from backend.turbomode.core_engine.feature_list import FEATURE_LIST, features_to_array, features_to_array_vectorized
from backend.turbomode.canonical_ohlcv_loader import load_ohlcv_for_trades, CANONICAL_DB_PATH
from backend.turbomode.core_engine.train_turbomode_models_fastmode import (
    train_single_sector_worker_fastmode,
    save_fastmode_models
)

from sklearn.model_selection import train_test_split

# In-memory cache for sector data
_sector_cache = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('sector_batch_trainer')


def compute_labels_14d_swing(trades: List[Dict], ohlcv_data: Dict, horizon_days: int = 14, buy_threshold: float = 0.06, sell_threshold: float = -0.06) -> Dict:
    """
    Compute 14-day swing trade labels based on close-to-close returns.

    **SWING TRADE LABEL CONFIGURATION:**
    - Horizon: 14 trading days (2-3 weeks)
    - Thresholds: ±6% return over 14 days (configurable)
    - Method: Close-to-close return (not intraday TP/DD)

    **LABEL SEMANTICS:**
    - 0 = SELL: 14-day return <= -6% (bearish swing)
    - 1 = HOLD: 14-day return between -6% and +6% (no edge)
    - 2 = BUY: 14-day return >= +6% (bullish swing)

    Args:
        trades: List of trade dicts with id, symbol, entry_date, entry_price
        ohlcv_data: Dict mapping symbol -> OHLCV DataFrame with 'close' column
        horizon_days: Number of days for the trading horizon (default: 14)
        buy_threshold: Threshold for BUY label (default: 0.06 = +6%)
        sell_threshold: Threshold for SELL label (default: -0.06 = -6%)

    Returns:
        Dict mapping trade_id -> label (int)
        Labels: 0=SELL, 1=HOLD, 2=BUY
    """

    # Pre-process OHLCV data to numpy arrays
    price_data = {}
    for symbol, df in ohlcv_data.items():
        if len(df) > 0 and 'close' in df.columns:
            timestamps = np.array([datetime.fromisoformat(d.split()[0]) for d in df['timestamp'].values])
            closes = df['close'].values.astype(np.float32)
            price_data[symbol] = (timestamps, closes)

    # Group trades by symbol
    trades_by_symbol = {}
    for trade in trades:
        symbol = trade["symbol"]
        if symbol not in trades_by_symbol:
            trades_by_symbol[symbol] = []
        trades_by_symbol[symbol].append(trade)

    # Initialize results
    all_labels = {}

    # Process each symbol's trades
    for symbol, symbol_trades in trades_by_symbol.items():
        if symbol not in price_data:
            # No data - default to HOLD
            for trade in symbol_trades:
                all_labels[trade["id"]] = 1  # HOLD
            continue

        timestamps, closes = price_data[symbol]

        for trade in symbol_trades:
            trade_id = trade["id"]
            entry_price = trade["entry_price"]
            entry_dt = datetime.fromisoformat(trade["entry_date"])

            # Find entry index
            entry_idx = np.searchsorted(timestamps, entry_dt, side='right')

            if entry_idx >= len(timestamps):
                all_labels[trade_id] = 1  # HOLD (no data)
                continue

            # Find 14-day exit index
            exit_dt = entry_dt + timedelta(days=horizon_days)
            exit_idx = np.searchsorted(timestamps, exit_dt, side='right')

            # Ensure we have at least some data in the 14-day window
            if exit_idx <= entry_idx or exit_idx > len(timestamps):
                # Try to use last available close if within reasonable range
                if entry_idx < len(timestamps):
                    exit_idx = len(timestamps)
                else:
                    all_labels[trade_id] = 1  # HOLD (insufficient data)
                    continue

            # Get exit close price (use close at or near day 14)
            exit_price = closes[exit_idx - 1]  # Last available close in window

            # Handle NaN
            if np.isnan(exit_price) or exit_price <= 0:
                all_labels[trade_id] = 1  # HOLD (invalid price)
                continue

            # Compute 14-day return
            r_14d = (exit_price - entry_price) / entry_price

            # Classify outcome
            if r_14d >= buy_threshold:
                outcome = 2  # BUY
            elif r_14d <= sell_threshold:
                outcome = 0  # SELL
            else:
                outcome = 1  # HOLD

            all_labels[trade_id] = outcome

    return all_labels


def compute_labels_1d_5pct(trades: List[Dict], ohlcv_data: Dict) -> Dict:
    """
    **DEPRECATED: Legacy 1-day labeler, not used in TurboMode. Do not re-enable.**

    This function is DISABLED. TurboMode uses ONLY 14-day MFE/MAE path-dependent labels.

    Compute label_1d_5pct for all trades.

    Single label configuration:
    - 1d horizon: next 1 trading day
    - Thresholds: configurable (legacy used ±5%)

    Args:
        trades: List of trade dicts with id, symbol, entry_date, entry_price
        ohlcv_data: Dict mapping symbol -> OHLCV DataFrame

    Returns:
        Dict mapping trade_id -> label (int)
        Labels: 0=SELL, 1=HOLD, 2=BUY
    """
    # Legacy configuration: 1d (deprecated - not used in TurboMode)
    horizon_days = 1
    buy_threshold = 0.05  # Legacy value
    sell_threshold = -0.05  # Legacy value

    # Pre-process OHLCV data to numpy arrays
    price_data = {}
    for symbol, df in ohlcv_data.items():
        if len(df) > 0:
            timestamps = np.array([datetime.fromisoformat(d.split()[0]) for d in df['timestamp'].values])
            highs = df['high'].values.astype(np.float32)
            lows = df['low'].values.astype(np.float32)
            price_data[symbol] = (timestamps, highs, lows)

    # Group trades by symbol
    trades_by_symbol = {}
    for trade in trades:
        symbol = trade["symbol"]
        if symbol not in trades_by_symbol:
            trades_by_symbol[symbol] = []
        trades_by_symbol[symbol].append(trade)

    # Initialize results
    all_labels = {}

    # Process each symbol's trades in batch
    for symbol, symbol_trades in trades_by_symbol.items():
        if symbol not in price_data:
            # No data - default to HOLD
            for trade in symbol_trades:
                all_labels[trade["id"]] = 1  # HOLD
            continue

        timestamps, highs, lows = price_data[symbol]

        # Vectorized processing for all trades of this symbol
        for trade in symbol_trades:
            trade_id = trade["id"]
            entry_price = trade["entry_price"]
            entry_dt = datetime.fromisoformat(trade["entry_date"])

            # Compute label for 1d/5% configuration
            end_dt = entry_dt + timedelta(days=horizon_days)

            # Use searchsorted for efficient date range lookup
            start_idx = np.searchsorted(timestamps, entry_dt, side='right')
            end_idx = np.searchsorted(timestamps, end_dt, side='right')

            if start_idx >= end_idx or start_idx >= len(timestamps):
                all_labels[trade_id] = 1  # HOLD (no data)
                continue

            # Extract window
            window_highs = highs[start_idx:end_idx]
            window_lows = lows[start_idx:end_idx]

            # Remove NaNs
            valid_highs = window_highs[~np.isnan(window_highs)]
            valid_lows = window_lows[~np.isnan(window_lows)]

            if len(valid_highs) == 0 or len(valid_lows) == 0:
                all_labels[trade_id] = 1  # HOLD (NaN window)
                continue

            # Compute TP/DD
            y_tp = (np.max(valid_highs) - entry_price) / entry_price
            y_dd = (np.min(valid_lows) - entry_price) / entry_price

            # Classify outcome
            if y_tp >= buy_threshold:
                outcome = 2  # BUY
            elif y_dd <= sell_threshold:
                outcome = 0  # SELL
            else:
                outcome = 1  # HOLD

            all_labels[trade_id] = outcome

    return all_labels


def load_sector_data_once(db_path: str, sector_symbols: List[str], horizon_days: int = 14, buy_threshold: float = 0.06, sell_threshold: float = -0.06) -> Tuple[np.ndarray, Dict, List]:
    """
    Load sector data from database.

    Args:
        db_path: Path to turbomode.db
        sector_symbols: List of symbols in this sector
        horizon_days: Trading horizon in days (default: 14)
        buy_threshold: Threshold for BUY label (default: 0.06 = +6%)
        sell_threshold: Threshold for SELL label (default: -0.06 = -6%)

    Returns:
        (X_features, labels_dict, trade_ids)
        - X_features: (N, 179) feature matrix
        - labels_dict: Dict mapping trade_id -> label (int: 0=SELL, 1=HOLD, 2=BUY)
        - trade_ids: List of trade IDs (aligned with X_features rows)
    """
    logger.info(f"[CONFIG] Loading sector data with: horizon={horizon_days}d, buy_threshold={buy_threshold:+.1%}, sell_threshold={sell_threshold:+.1%}")
    # Check cache first
    cache_key = tuple(sorted(sector_symbols))
    if cache_key in _sector_cache:
        return _sector_cache[cache_key]

    start_time = time.time()

    # Query all backtest trades for this sector
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Enable WAL mode for concurrent read performance
    cursor.execute("PRAGMA journal_mode=WAL;")
    conn.commit()

    placeholders = ','.join(['?'] * len(sector_symbols))
    query = f"""
        SELECT id, symbol, entry_date, entry_price, entry_features_json, outcome
        FROM trades
        WHERE trade_type = 'backtest'
        AND entry_features_json IS NOT NULL
        AND entry_date IS NOT NULL
        AND entry_price IS NOT NULL
        AND outcome IS NOT NULL
        AND symbol IN ({placeholders})
    """

    cursor.execute(query, sector_symbols)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        logger.warning(f"[WARN] No data found for sector symbols: {sector_symbols}")
        return np.array([]), {}, []

    logger.info(f"[DATA] Loaded {len(rows):,} trades for sector")

    # Parse features and outcomes (labels are pre-computed in trades table)
    labels_dict = {}
    id_list = []

    # Collect JSON feature strings for vectorized conversion
    json_feature_list = []

    # Outcome mapping: 'sell' -> 0, 'hold' -> 1, 'buy' -> 2
    outcome_map = {'sell': 0, 'hold': 1, 'buy': 2}

    for row in rows:
        trade_id, symbol, entry_date, entry_price, features_json, outcome = row

        try:
            # Map outcome string to label integer
            if outcome not in outcome_map:
                logger.error(f"[ERROR] Invalid outcome '{outcome}' for trade {trade_id}")
                continue

            # Collect JSON string for vectorized processing
            json_feature_list.append(features_json)
            labels_dict[trade_id] = outcome_map[outcome]
            id_list.append(trade_id)

        except Exception as e:
            logger.error(f"[ERROR] Failed to parse trade {trade_id}: {e}")
            continue

    # Vectorized conversion of all JSON feature strings to NumPy array
    X_features = features_to_array_vectorized(json_feature_list, fill_value=0.0)

    # Validate feature matrix shape
    from backend.turbomode.core_engine.feature_list import FEATURE_COUNT
    if X_features.shape[1] != FEATURE_COUNT:
        logger.error(f"[ERROR] Expected {FEATURE_COUNT} features, got {X_features.shape[1]}")
        return np.array([]), {}, []

    parse_time = time.time() - start_time
    logger.info(f"[PARSE] Features and labels parsed in {parse_time:.2f}s ({len(X_features):,} samples)")

    # Log label distribution
    label_counts = {0: 0, 1: 0, 2: 0}
    for label in labels_dict.values():
        label_counts[label] += 1

    total = sum(label_counts.values())
    if total > 0:
        logger.info(f"[LABELS] Distribution: SELL={label_counts[0]:,} ({label_counts[0]/total*100:.1f}%), "
                   f"HOLD={label_counts[1]:,} ({label_counts[1]/total*100:.1f}%), "
                   f"BUY={label_counts[2]:,} ({label_counts[2]/total*100:.1f}%)")

    total_time = time.time() - start_time
    logger.info(f"[TOTAL] Sector data loaded in {total_time:.2f}s")

    # Store in cache
    _sector_cache[cache_key] = (X_features, labels_dict, id_list)

    return X_features, labels_dict, id_list


def run_sector_training(
    sector_name: str,
    sector_symbols: List[str],
    db_path: str,
    save_dir: str
) -> Dict:
    """
    Train exactly ONE ensemble (5 base + 1 meta) for a single sector.

    **14-DAY SWING TRADE ARCHITECTURE:**
    This function trains models for 14-day directional swing trades.

    Pipeline:
    1. Loads sector data from database
    2. Preprocesses 179 features
    3. Computes label_14d_swing (±6% over 14 days)
    4. Trains 5 base models + MetaLearner
    5. Saves ensemble to: models/<sector>/*.pkl

    **LABEL SEMANTICS:**
    - 0 = SELL: 14-day return <= -6%
    - 1 = HOLD: 14-day return between -6% and +6%
    - 2 = BUY: 14-day return >= +6%

    Args:
        sector_name: Sector name (e.g., 'technology')
        sector_symbols: List of symbols in this sector
        db_path: Path to turbomode.db
        save_dir: Save directory for models

    Returns:
        Dict with training results and timing statistics
    """
    sector_start = time.time()

    logger.info("=" * 80)
    logger.info(f"SECTOR: {sector_name.upper()}")
    logger.info(f"Symbols: {len(sector_symbols)}")
    logger.info("=" * 80)

    # STEP 1: Load sector data
    X_sector, labels_dict, trade_ids = load_sector_data_once(db_path, sector_symbols)

    if len(X_sector) == 0:
        logger.error(f"[FAILED] No valid data for sector {sector_name}")
        return {'status': 'failed', 'error': 'No valid data'}

    # STEP 2: Build label vector (14-day MFE/MAE labels only)
    y_sector = np.array([labels_dict[tid] for tid in trade_ids], dtype=np.int32)

    # Log label distribution
    sell_count = np.sum(y_sector == 0)
    hold_count = np.sum(y_sector == 1)
    buy_count = np.sum(y_sector == 2)

    logger.info(f"Label distribution: SELL={sell_count:,} ({sell_count/len(y_sector)*100:.1f}%), "
               f"HOLD={hold_count:,} ({hold_count/len(y_sector)*100:.1f}%), "
               f"BUY={buy_count:,} ({buy_count/len(y_sector)*100:.1f}%)")

    # STEP 3: Split into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_sector, y_sector, test_size=0.2, random_state=42, stratify=y_sector
    )

    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}")

    # STEP 4: Train single model
    try:
        train_start = time.time()

        result = train_single_sector_worker_fastmode(
            sector=sector_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            horizon_days=14,  # ENFORCED: 14d (TurboMode uses only 14-day MFE/MAE labels)
            save_models=True,
            save_dir=save_dir
        )

        train_time = time.time() - train_start

        logger.info(f"[OK] Training complete in {train_time:.1f}s")
        logger.info(f"Model accuracy: {result.get('meta_accuracy', result.get('accuracy', 'N/A'))}")
        logger.info(f"Saved to: {save_dir}/{sector_name}/")

    except Exception as e:
        logger.error(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'failed', 'error': str(e)}

    sector_time = time.time() - sector_start

    logger.info("=" * 80)
    logger.info(f"SECTOR {sector_name.upper()} COMPLETE")
    logger.info(f"Total time: {sector_time/60:.1f} minutes")
    logger.info("=" * 80)

    return {
        'status': 'completed',
        'sector': sector_name,
        'total_time': sector_time,
        'result': result
    }


if __name__ == '__main__':
    # Test with a single sector
    from backend.turbomode.core_engine.training_symbols import get_symbols_by_sector

    test_sector = 'technology'
    test_symbols = get_symbols_by_sector(test_sector)

    backend_dir = str(project_root / "backend")
    db_path = os.path.join(backend_dir, "data", "turbomode.db")
    save_dir = os.path.join(backend_dir, "turbomode", "models", "trained")

    print(f"\nTesting single-model training with {test_sector} sector...")
    print(f"Symbols: {len(test_symbols)}")
    print(f"Label: 14-day MFE/MAE path-dependent (±6% threshold)")
    print(f"Expected output: 1 model file")
    print()

    result = run_sector_training(
        sector_name=test_sector,
        sector_symbols=test_symbols,
        db_path=db_path,
        save_dir=save_dir
    )

    print("\nResult:")
    print(json.dumps(result, indent=2, default=str))
