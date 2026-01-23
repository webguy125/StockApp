"""
TurboMode Training Data Loader
100% Pure TurboMode - NO AdvancedML Dependencies

Loads training data from turbomode.db for model training.
Replaces HistoricalBacktest.prepare_training_data() with a pure TurboMode implementation.

Author: TurboMode Core Engine
Date: 2026-01-06
"""

import sys
import os
import sqlite3
import json
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(backend_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import canonical feature list
from backend.turbomode.feature_list import FEATURE_LIST, features_to_array

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('turbomode_training_loader')


def _load_price_data_for_symbols(conn, symbols):
    """Preload price_data for all symbols into memory, grouped by symbol."""
    cursor = conn.cursor()
    placeholders = ",".join(["?"] * len(symbols))
    cursor.execute(
        f"SELECT symbol, date, high, low FROM price_data WHERE symbol IN ({placeholders}) ORDER BY symbol, date",
        tuple(symbols)
    )
    data_by_symbol = {}
    for symbol, ts, high, low in cursor.fetchall():
        data_by_symbol.setdefault(symbol, []).append((ts, high, low))
    # Convert to numpy arrays per symbol for fast slicing
    for symbol, rows in data_by_symbol.items():
        timestamps = np.array([r[0] for r in rows])
        highs = np.array([r[1] for r in rows], dtype=np.float32)
        lows = np.array([r[2] for r in rows], dtype=np.float32)
        data_by_symbol[symbol] = (timestamps, highs, lows)
    return data_by_symbol


def compute_labels_for_trades(conn, trades, horizon_days, thresholds):
    """
    Compute labels dynamically from price_data table.

    Args:
        conn: Database connection
        trades: List of trade dicts with id, symbol, entry_date, entry_price
        horizon_days: Horizon window (1 or 2 days)
        thresholds: Dict with 'buy' and 'sell' thresholds (e.g., {'buy': 0.10, 'sell': -0.10})

    Returns:
        Dict mapping trade_id -> {'y_tp': float, 'y_dd': float, 'outcome': int}
        outcome: 0=SELL, 1=HOLD, 2=BUY
    """
    labels = {}

    # Preload price_data for all symbols used in trades
    symbols = sorted({t["symbol"] for t in trades})
    price_data = _load_price_data_for_symbols(conn, symbols)

    buy_th = thresholds["buy"]
    sell_th = thresholds["sell"]

    for trade in trades:
        trade_id = trade["id"]
        symbol = trade["symbol"]
        entry_price = trade["entry_price"]
        entry_dt = datetime.fromisoformat(trade["entry_date"])
        end_dt = entry_dt + timedelta(days=horizon_days)

        if symbol not in price_data:
            labels[trade_id] = {"y_tp": 0.0, "y_dd": 0.0, "outcome": 1}
            continue

        timestamps, highs, lows = price_data[symbol]

        # Boolean mask for horizon window
        mask = (timestamps > entry_dt.isoformat()) & (timestamps <= end_dt.isoformat())
        if not np.any(mask):
            labels[trade_id] = {"y_tp": 0.0, "y_dd": 0.0, "outcome": 1}
            continue

        window_highs = highs[mask]
        window_lows = lows[mask]

        y_tp = (np.max(window_highs) - entry_price) / entry_price
        y_dd = (np.min(window_lows) - entry_price) / entry_price

        if y_tp >= buy_th:
            outcome = 2
        elif y_dd <= sell_th:
            outcome = 0
        else:
            outcome = 1

        labels[trade_id] = {"y_tp": float(y_tp), "y_dd": float(y_dd), "outcome": outcome}

    return labels


class TurboModeTrainingDataLoader:
    """
    Pure TurboMode training data loader

    Responsibilities:
    - Load training samples from turbomode.db
    - Parse entry_features_json into feature matrix X
    - Map outcome labels to integers (SELL=0, HOLD=1, BUY=2 in 3-class; SELL=0, BUY=1 in binary)
    - Return (X, y) for model training

    NO DEPENDENCIES ON:
    - AdvancedMLDatabase
    - HistoricalBacktest
    - Any AdvancedML schema or tables
    - TurboMode DC components
    """

    def __init__(self, db_path: str = None, cache_dir: str = None):
        """
        Initialize training data loader

        Args:
            db_path: Path to turbomode.db (defaults to backend/data/turbomode.db)
            cache_dir: Directory for .npy cache files (defaults to backend/data/training_cache)
        """
        if db_path is None:
            db_path = os.path.join(backend_dir, "data", "turbomode.db")

        if cache_dir is None:
            cache_dir = os.path.join(backend_dir, "data", "training_cache")

        self.db_path = db_path
        self.cache_dir = cache_dir

        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        logger.info("[INIT] TurboMode Training Data Loader initialized")
        logger.info(f"       Database: {db_path}")
        logger.info(f"       Cache dir: {cache_dir}")

    def load_training_data(self, include_hold: bool = True, symbols_filter: Optional[List[str]] = None,
                          return_split: bool = False, test_size: float = 0.2,
                          random_state: int = 42, horizon_days: int = None,
                          thresholds: Dict[str, float] = None) -> Tuple[np.ndarray, ...]:
        """
        Load training data from turbomode.db

        Args:
            include_hold: Whether to include HOLD samples (default: True)
                         Binary classification: BUY vs SELL only
                         Multi-class: BUY vs HOLD vs SELL
            symbols_filter: Optional list of symbols to filter by (for sector-specific training)
            return_split: If True, return train/val split. If False, return full dataset
            test_size: Validation split size (default: 0.2 = 20%)
            random_state: Random seed for reproducible splits (default: 42)
            horizon_days: Horizon window for label computation (1 or 2 days). If None, use pre-computed 'outcome' column.
            thresholds: Dict with 'buy' and 'sell' thresholds (e.g., {'buy': 0.10, 'sell': -0.10}).
                       Required if horizon_days is specified.

        Returns:
            If symbols_filter is None (loading ALL data):
                - return_split=False: (X, y, symbols)
                - return_split=True: (X_train, y_train, X_val, y_val, symbols)
            If symbols_filter is provided (sector-specific):
                - return_split=False: (X, y)
                - return_split=True: (X_train, y_train, X_val, y_val)

        Label mapping:
            - Binary (include_hold=False): SELL/DOWN=0, BUY/UP=1
            - Multi-class (include_hold=True): SELL/DOWN=0, HOLD/NEUTRAL=1, BUY/UP=2
        """
        # Check for cached data (only for full dataset loads with no filtering)
        if symbols_filter is None and not return_split:
            cache_suffix = "3class" if include_hold else "binary"
            cache_files = {
                'X': os.path.join(self.cache_dir, f"X_all_{cache_suffix}.npy"),
                'y': os.path.join(self.cache_dir, f"y_all_{cache_suffix}.npy"),
                'symbols': os.path.join(self.cache_dir, f"symbols_all_{cache_suffix}.npy"),
                'meta': os.path.join(self.cache_dir, f"cache_meta_{cache_suffix}.json")
            }

            # Check if cache is valid
            cache_valid = False
            if all(os.path.exists(f) for f in [cache_files['X'], cache_files['y'], cache_files['symbols'], cache_files['meta']]):
                # Load cache metadata
                with open(cache_files['meta'], 'r') as f:
                    cache_meta = json.load(f)

                # Check if sample count matches database
                conn_check = sqlite3.connect(self.db_path)
                cursor_check = conn_check.cursor()

                if include_hold:
                    cursor_check.execute("""
                        SELECT COUNT(*) FROM trades
                        WHERE trade_type = 'backtest'
                        AND entry_features_json IS NOT NULL
                        AND outcome IN ('buy', 'hold', 'sell')
                    """)
                else:
                    cursor_check.execute("""
                        SELECT COUNT(*) FROM trades
                        WHERE trade_type = 'backtest'
                        AND entry_features_json IS NOT NULL
                        AND outcome IN ('buy', 'sell')
                    """)

                db_sample_count = cursor_check.fetchone()[0]
                conn_check.close()

                cached_sample_count = cache_meta.get('sample_count', 0)

                if db_sample_count == cached_sample_count:
                    cache_valid = True
                elif db_sample_count > cached_sample_count:
                    # NEW: Incremental cache append (DB has more samples than cache)
                    new_sample_count = db_sample_count - cached_sample_count
                    logger.info(f"[CACHE] Incremental update detected: {new_sample_count:,} new samples")
                    logger.info(f"[CACHE] Loading existing cache + appending new data (FAST!)")

                    # Try incremental append
                    try:
                        X_cached, y_cached, symbols_cached, success = self._load_and_append_cache(
                            cache_files, cached_sample_count, include_hold
                        )
                        if success:
                            return X_cached, y_cached, symbols_cached
                        else:
                            # Append failed, fall through to full rebuild
                            logger.info(f"[CACHE] Append failed, rebuilding from scratch...")
                    except Exception as e:
                        logger.info(f"[CACHE] Append error: {e}, rebuilding from scratch...")
                else:
                    # DB has FEWER samples than cache (data was deleted, corruption, etc.)
                    logger.info(f"[CACHE] Cache corruption: DB has {db_sample_count:,} samples, cache has {cached_sample_count:,}")
                    logger.info(f"[CACHE] Invalidating cache and rebuilding from database...")

            # Load from cache if valid
            if cache_valid:
                logger.info("[CACHE] Loading from .npy cache files (NO JSON PARSING)")
                X = np.load(cache_files['X'])
                y = np.load(cache_files['y'])
                symbols = np.load(cache_files['symbols'])

                logger.info(f"[CACHE] Loaded {len(X):,} samples from cache")
                logger.info(f"        Features: {X.shape[1]}")

                # Print statistics
                if include_hold:
                    sell_count = np.sum(y == 0)
                    hold_count = np.sum(y == 1)
                    buy_count = np.sum(y == 2)
                    logger.info(f"        SELL/DOWN (0): {sell_count:,} ({sell_count/len(y)*100:.1f}%)")
                    logger.info(f"        HOLD/NEUTRAL (1): {hold_count:,} ({hold_count/len(y)*100:.1f}%)")
                    logger.info(f"        BUY/UP (2): {buy_count:,} ({buy_count/len(y)*100:.1f}%)")
                else:
                    sell_count = np.sum(y == 0)
                    buy_count = np.sum(y == 1)
                    logger.info(f"        SELL/DOWN (0): {sell_count:,} ({sell_count/len(y)*100:.1f}%)")
                    logger.info(f"        BUY/UP (1): {buy_count:,} ({buy_count/len(y)*100:.1f}%)")

                return X, y, symbols

        # No cache or filtered query - proceed with database load
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Build query with optional symbol filtering
        # If horizon_days is specified, load trades with entry_date/entry_price for dynamic labeling
        # Otherwise, use pre-computed 'outcome' column
        if horizon_days is not None:
            # Dynamic label computation - load id, entry_date, entry_price, features, symbol
            base_query = """
                SELECT id, symbol, entry_date, entry_price, entry_features_json
                FROM trades
                WHERE trade_type = 'backtest'
                AND entry_features_json IS NOT NULL
                AND entry_date IS NOT NULL
                AND entry_price IS NOT NULL
            """
        elif include_hold:
            # Multi-class: BUY, HOLD, SELL (pre-computed outcomes)
            base_query = """
                SELECT entry_features_json, outcome, symbol
                FROM trades
                WHERE trade_type = 'backtest'
                AND entry_features_json IS NOT NULL
                AND outcome IS NOT NULL
                AND outcome IN ('buy', 'hold', 'sell')
            """
        else:
            # Binary: BUY vs SELL only (pre-computed outcomes)
            base_query = """
                SELECT entry_features_json, outcome, symbol
                FROM trades
                WHERE trade_type = 'backtest'
                AND entry_features_json IS NOT NULL
                AND outcome IS NOT NULL
                AND outcome IN ('buy', 'sell')
            """

        # Add symbol filter if provided
        if symbols_filter:
            placeholders = ','.join(['?'] * len(symbols_filter))
            query = base_query + f" AND symbol IN ({placeholders})"
            cursor.execute(query, symbols_filter)
            logger.info(f"[FILTER] Loading data for {len(symbols_filter)} symbols: {', '.join(symbols_filter[:5])}{'...' if len(symbols_filter) > 5 else ''}")
        else:
            query = base_query
            cursor.execute(query)

        rows = cursor.fetchall()

        if not rows:
            logger.warning("[WARNING] No training data found in database")
            conn.close()
            return np.array([]), np.array([])

        logger.info(f"[DATA] Loading {len(rows):,} samples from turbomode.db")

        # Extract features, labels, and symbols
        feature_list = []
        label_list = []
        symbol_list = []

        if horizon_days is not None:
            # Dynamic label computation mode
            if thresholds is None:
                raise ValueError("thresholds parameter is required when horizon_days is specified")

            logger.info(f"[LABELS] Computing labels dynamically for {horizon_days}d horizon")
            logger.info(f"[LABELS] Thresholds: BUY >= {thresholds['buy']:.2%}, SELL <= {thresholds['sell']:.2%}")

            # Build trades list for label computation
            trades = []
            id_to_features = {}

            for row in rows:
                trade_id, symbol, entry_date, entry_price, features_json = row
                trades.append({
                    "id": trade_id,
                    "symbol": symbol,
                    "entry_date": entry_date,
                    "entry_price": entry_price
                })
                id_to_features[trade_id] = features_json

            # Compute labels dynamically from price_data
            labels_by_id = compute_labels_for_trades(conn, trades, horizon_days, thresholds)

            # Extract features and labels
            for trade in trades:
                trade_id = trade["id"]
                if trade_id not in labels_by_id:
                    continue

                try:
                    # Parse features JSON
                    features = json.loads(id_to_features[trade_id])

                    # Extract feature values using CANONICAL FEATURE_LIST order
                    feature_values = features_to_array(features, fill_value=0.0)

                    # Validate feature count
                    if len(feature_values) != 179:
                        logger.error(f"[ERROR] Expected 179 features, got {len(feature_values)}")
                        continue

                    # Get dynamically computed label
                    label = labels_by_id[trade_id]["outcome"]

                    # Filter by include_hold setting
                    if not include_hold and label == 1:
                        continue  # Skip HOLD samples in binary mode

                    # Remap labels for binary mode (0=SELL, 1=BUY)
                    if not include_hold:
                        label = 0 if label == 0 else 1  # Map 2 (BUY) -> 1

                    feature_list.append(feature_values)
                    label_list.append(label)
                    symbol_list.append(trade["symbol"])

                except Exception as e:
                    logger.error(f"[ERROR] Failed to parse sample: {e}")
                    continue

        else:
            # Pre-computed outcome mode
            # Define label mapping
            if include_hold:
                # 3-class mode: 0=down/sell, 1=neutral/hold, 2=up/buy
                label_map = {'sell': 0, 'hold': 1, 'buy': 2}
            else:
                # Binary mode: 0=down/sell, 1=up/buy
                label_map = {'sell': 0, 'buy': 1}

            for row in rows:
                features_json = row[0]
                outcome = row[1]
                symbol = row[2]

                try:
                    # Parse features JSON
                    features = json.loads(features_json)

                    # Extract feature values using CANONICAL FEATURE_LIST order
                    # This ensures deterministic, fixed-length vectors (179 features)
                    feature_values = features_to_array(features, fill_value=0.0)

                    # Validate feature count
                    if len(feature_values) != 179:
                        logger.error(f"[ERROR] Expected 179 features, got {len(feature_values)}")
                        continue

                    # Map outcome to label integer
                    label = label_map.get(outcome, -1)

                    if label == -1:
                        logger.warning(f"[WARN] Unknown outcome '{outcome}', skipping sample")
                        continue

                    feature_list.append(feature_values)
                    label_list.append(label)
                    symbol_list.append(symbol)

                except Exception as e:
                    logger.error(f"[ERROR] Failed to parse sample: {e}")
                    continue

        conn.close()

        # Convert to numpy arrays
        X = np.array(feature_list, dtype=np.float32)
        y = np.array(label_list, dtype=np.int32)
        symbols = np.array(symbol_list, dtype=str)

        # Validation
        if len(X) == 0:
            logger.error("[ERROR] No valid samples loaded")
            return (np.array([]), np.array([]), np.array([])) if symbols_filter is None else (np.array([]), np.array([]))

        # Print statistics
        logger.info(f"\n[DATA] Training data loaded successfully")
        logger.info(f"  Samples: {X.shape[0]:,}")
        logger.info(f"  Features: {X.shape[1]}")

        if include_hold:
            # 3-class: 0=sell/down, 1=hold/neutral, 2=buy/up
            sell_count = np.sum(y == 0)
            hold_count = np.sum(y == 1)
            buy_count = np.sum(y == 2)
            logger.info(f"  SELL/DOWN (0): {sell_count:,} ({sell_count/len(y)*100:.1f}%)")
            logger.info(f"  HOLD/NEUTRAL (1): {hold_count:,} ({hold_count/len(y)*100:.1f}%)")
            logger.info(f"  BUY/UP (2): {buy_count:,} ({buy_count/len(y)*100:.1f}%)")
        else:
            # Binary: 0=sell/down, 1=buy/up
            sell_count = np.sum(y == 0)
            buy_count = np.sum(y == 1)
            logger.info(f"  SELL/DOWN (0): {sell_count:,} ({sell_count/len(y)*100:.1f}%)")
            logger.info(f"  BUY/UP (1): {buy_count:,} ({buy_count/len(y)*100:.1f}%)")

        # Save to cache if this is a full dataset load (no filtering)
        if symbols_filter is None and not return_split:
            cache_suffix = "3class" if include_hold else "binary"
            cache_files = {
                'X': os.path.join(self.cache_dir, f"X_all_{cache_suffix}.npy"),
                'y': os.path.join(self.cache_dir, f"y_all_{cache_suffix}.npy"),
                'symbols': os.path.join(self.cache_dir, f"symbols_all_{cache_suffix}.npy"),
                'meta': os.path.join(self.cache_dir, f"cache_meta_{cache_suffix}.json")
            }

            logger.info(f"[CACHE] Saving parsed arrays to .npy cache...")
            np.save(cache_files['X'], X)
            np.save(cache_files['y'], y)
            np.save(cache_files['symbols'], symbols)

            # Save metadata for cache validation
            cache_meta = {
                'sample_count': len(X),
                'feature_count': X.shape[1],
                'created_at': datetime.now().isoformat(),
                'include_hold': include_hold
            }
            with open(cache_files['meta'], 'w') as f:
                json.dump(cache_meta, f, indent=2)

            logger.info(f"[CACHE] Saved to {self.cache_dir}")
            logger.info(f"[CACHE] Future loads will skip JSON parsing (instant load)")
            logger.info(f"[CACHE] Cache will auto-invalidate if database sample count changes")

        # Return train/val split if requested
        if return_split:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            logger.info(f"\n[SPLIT] Train/validation split:")
            logger.info(f"  Training: {X_train.shape[0]:,} samples ({(1-test_size)*100:.0f}%)")
            logger.info(f"  Validation: {X_val.shape[0]:,} samples ({test_size*100:.0f}%)")
            # If no symbol filter, we're loading all data for parent process - return symbols
            if symbols_filter is None:
                return X_train, y_train, X_val, y_val, symbols
            else:
                return X_train, y_train, X_val, y_val
        else:
            # If no symbol filter, we're loading all data for parent process - return symbols
            if symbols_filter is None:
                return X, y, symbols
            else:
                return X, y

    def _load_and_append_cache(self, cache_files: Dict[str, str], cached_count: int,
                                include_hold: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Load existing cache and append only new samples (incremental update)

        Args:
            cache_files: Dictionary of cache file paths
            cached_count: Number of samples in existing cache
            include_hold: Whether to include HOLD samples

        Returns:
            (X, y, symbols, success): Updated arrays and success flag
        """
        logger.info(f"[CACHE APPEND] Loading existing cache...")

        # Load existing cache
        X_cached = np.load(cache_files['X'])
        y_cached = np.load(cache_files['y'])
        symbols_cached = np.load(cache_files['symbols'], allow_pickle=True)

        logger.info(f"[CACHE APPEND] Loaded {len(X_cached):,} existing samples")
        logger.info(f"[CACHE APPEND] Querying database for new samples...")

        # Query database for NEW samples only (LIMIT + OFFSET)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if include_hold:
            query = """
                SELECT entry_features_json, outcome, symbol
                FROM trades
                WHERE trade_type = 'backtest'
                AND entry_features_json IS NOT NULL
                AND outcome IN ('buy', 'hold', 'sell')
                ORDER BY id
                LIMIT -1 OFFSET ?
            """
        else:
            query = """
                SELECT entry_features_json, outcome, symbol
                FROM trades
                WHERE trade_type = 'backtest'
                AND entry_features_json IS NOT NULL
                AND outcome IN ('buy', 'sell')
                ORDER BY id
                LIMIT -1 OFFSET ?
            """

        cursor.execute(query, (cached_count,))
        new_rows = cursor.fetchall()
        conn.close()

        logger.info(f"[CACHE APPEND] Found {len(new_rows):,} new samples to parse")

        if len(new_rows) == 0:
            logger.info(f"[CACHE APPEND] No new samples, using existing cache")
            return X_cached, y_cached, symbols_cached, True

        # Parse new samples
        feature_list = []
        label_list = []
        symbol_list = []

        label_map = {'sell': 0, 'hold': 1, 'buy': 2}
        exclude_keys = {'feature_count', 'symbol', 'last_price', 'last_volume', 'timestamp', 'error'}

        for i, (features_json, outcome, symbol) in enumerate(new_rows):
            try:
                features_dict = json.loads(features_json)
                feature_array = features_to_array(features_dict)
                feature_list.append(feature_array)
                label_list.append(label_map[outcome])
                symbol_list.append(symbol)
            except Exception as e:
                continue

        if len(feature_list) == 0:
            logger.info(f"[CACHE APPEND] No valid new samples parsed")
            return X_cached, y_cached, symbols_cached, True

        X_new = np.array(feature_list, dtype=np.float32)
        y_new = np.array(label_list, dtype=np.int32)
        symbols_new = np.array(symbol_list)

        logger.info(f"[CACHE APPEND] Parsed {len(X_new):,} new samples")

        # Concatenate old + new
        X_combined = np.concatenate([X_cached, X_new], axis=0)
        y_combined = np.concatenate([y_cached, y_new], axis=0)
        symbols_combined = np.concatenate([symbols_cached, symbols_new], axis=0)

        logger.info(f"[CACHE APPEND] Combined: {len(X_combined):,} total samples")

        # Save updated cache
        logger.info(f"[CACHE APPEND] Saving updated cache files...")
        np.save(cache_files['X'], X_combined)
        np.save(cache_files['y'], y_combined)
        np.save(cache_files['symbols'], symbols_combined)

        # Update metadata
        cache_meta = {
            'sample_count': len(X_combined),
            'feature_count': X_combined.shape[1],
            'created_at': datetime.now().isoformat(),
            'include_hold': include_hold
        }
        with open(cache_files['meta'], 'w') as f:
            json.dump(cache_meta, f, indent=2)

        logger.info(f"[CACHE APPEND] Cache updated successfully!")

        # Print statistics
        if include_hold:
            sell_count = np.sum(y_combined == 0)
            hold_count = np.sum(y_combined == 1)
            buy_count = np.sum(y_combined == 2)
            logger.info(f"        SELL/DOWN (0): {sell_count:,} ({sell_count/len(y_combined)*100:.1f}%)")
            logger.info(f"        HOLD/NEUTRAL (1): {hold_count:,} ({hold_count/len(y_combined)*100:.1f}%)")
            logger.info(f"        BUY/UP (2): {buy_count:,} ({buy_count/len(y_combined)*100:.1f}%)")
        else:
            sell_count = np.sum(y_combined == 0)
            buy_count = np.sum(y_combined == 1)
            logger.info(f"        SELL/DOWN (0): {sell_count:,} ({sell_count/len(y_combined)*100:.1f}%)")
            logger.info(f"        BUY/UP (1): {buy_count:,} ({buy_count/len(y_combined)*100:.1f}%)")

        return X_combined, y_combined, symbols_combined, True

    def get_sample_count(self, with_features: bool = True) -> Dict[str, int]:
        """
        Get sample counts from database

        Args:
            with_features: Count only samples with features (default: True)

        Returns:
            Dictionary with sample counts by label
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if with_features:
            query = """
                SELECT outcome, COUNT(*)
                FROM trades
                WHERE trade_type = 'backtest'
                AND entry_features_json IS NOT NULL
                GROUP BY outcome
            """
        else:
            query = """
                SELECT outcome, COUNT(*)
                FROM trades
                WHERE trade_type = 'backtest'
                GROUP BY outcome
            """

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        counts = {}
        for row in rows:
            outcome, count = row
            counts[outcome] = count

        counts['total'] = sum(counts.values())

        return counts


if __name__ == '__main__':
    # Test the training data loader
    print("Testing TurboMode Training Data Loader")
    print("=" * 80)

    loader = TurboModeTrainingDataLoader()

    # Get sample counts
    print("\nSample counts in database:")
    counts = loader.get_sample_count(with_features=True)
    for outcome, count in sorted(counts.items()):
        print(f"  {outcome}: {count:,}")

    print("\n" + "=" * 80)
    print("Loading training data (multi-class: BUY vs HOLD vs SELL)...")
    print("=" * 80)

    # Load multi-class classification data (all 169,400 samples)
    X, y = loader.load_training_data(include_hold=True)

    if len(X) > 0:
        print(f"\n[SUCCESS] Loaded {len(X):,} samples")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label vector shape: {y.shape}")
        print(f"\nSample features (first 10):")
        print(X[0][:10])
        print(f"\nSample label: {y[0]}")
    else:
        print("[ERROR] No data loaded")

    print("\n" + "=" * 80)
    print("[OK] TurboMode Training Data Loader test complete!")
