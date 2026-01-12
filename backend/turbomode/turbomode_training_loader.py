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
from typing import Tuple, Dict, Any
import logging

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

    def __init__(self, db_path: str = None):
        """
        Initialize training data loader

        Args:
            db_path: Path to turbomode.db (defaults to backend/data/turbomode.db)
        """
        if db_path is None:
            db_path = os.path.join(backend_dir, "data", "turbomode.db")

        self.db_path = db_path

        logger.info("[INIT] TurboMode Training Data Loader initialized")
        logger.info(f"       Database: {db_path}")

    def load_training_data(self, include_hold: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load training data from turbomode.db

        Args:
            include_hold: Whether to include HOLD samples (default: False)
                         Binary classification: BUY vs SELL only
                         Multi-class: BUY vs HOLD vs SELL

        Returns:
            Tuple of (X, y) where:
            - X is feature matrix (n_samples, n_features)
            - y is label vector (n_samples,)

        Label mapping:
            - Binary (include_hold=False): SELL/DOWN=0, BUY/UP=1
            - Multi-class (include_hold=True): SELL/DOWN=0, HOLD/NEUTRAL=1, BUY/UP=2
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Query for samples with features and labels
        if include_hold:
            # Multi-class: BUY, HOLD, SELL
            query = """
                SELECT entry_features_json, outcome
                FROM trades
                WHERE trade_type = 'backtest'
                AND entry_features_json IS NOT NULL
                AND outcome IS NOT NULL
                AND outcome IN ('buy', 'hold', 'sell')
            """
        else:
            # Binary: BUY vs SELL only
            query = """
                SELECT entry_features_json, outcome
                FROM trades
                WHERE trade_type = 'backtest'
                AND entry_features_json IS NOT NULL
                AND outcome IS NOT NULL
                AND outcome IN ('buy', 'sell')
            """

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            logger.warning("[WARNING] No training data found in database")
            return np.array([]), np.array([])

        logger.info(f"[DATA] Loading {len(rows):,} samples from turbomode.db")

        # Extract features and labels
        feature_list = []
        label_list = []

        # Define label mapping
        if include_hold:
            # 3-class mode: 0=down/sell, 1=neutral/hold, 2=up/buy
            label_map = {'sell': 0, 'hold': 1, 'buy': 2}
        else:
            # Binary mode: 0=down/sell, 1=up/buy
            label_map = {'sell': 0, 'buy': 1}

        # Metadata fields to exclude from features
        exclude_keys = {'feature_count', 'symbol', 'last_price', 'last_volume', 'timestamp', 'error'}

        for row in rows:
            features_json = row[0]
            outcome = row[1]

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

            except Exception as e:
                logger.error(f"[ERROR] Failed to parse sample: {e}")
                continue

        # Convert to numpy arrays
        X = np.array(feature_list, dtype=np.float32)
        y = np.array(label_list, dtype=np.int32)

        # Validation
        if len(X) == 0:
            logger.error("[ERROR] No valid samples loaded")
            return np.array([]), np.array([])

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

        return X, y

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
