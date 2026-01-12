"""
TurboMode Feature Extractor
100% Pure TurboMode - NO AdvancedML Dependencies

Extracts 179 technical features from price data for ML training.
Uses GPUFeatureEngineer from backend/advanced_ml/features/ ONLY for feature calculation.
Does NOT use any AdvancedML database, schema, or backtest components.

Author: TurboMode Core Engine
Date: 2026-01-06
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json
import logging

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(backend_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Master Market Data API (read-only price data source)
from master_market_data.market_data_api import get_market_data_api

# Import GPU Feature Engineer (computational utility only - NOT AdvancedML contamination)
# This is a PURE UTILITY for calculating technical indicators
from backend.advanced_ml.features.gpu_feature_engineer import GPUFeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('turbomode_feature_extractor')


class TurboModeFeatureExtractor:
    """
    Pure TurboMode feature extractor

    Responsibilities:
    - Load candle data from Master Market Data DB
    - Calculate 179 technical features using GPUFeatureEngineer
    - Return deterministic, ordered feature dict
    - Serialize features to JSON for storage in turbomode.db

    NO DEPENDENCIES ON:
    - AdvancedMLDatabase
    - HistoricalBacktest
    - Any AdvancedML schema or tables
    - TurboMode DC components
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize TurboMode feature extractor

        Args:
            use_gpu: Whether to use GPU acceleration for feature computation
        """
        # Connect to Master Market Data API (read-only)
        self.market_data_api = get_market_data_api()

        # Initialize GPU Feature Engineer (computational utility)
        # This is NOT an AdvancedML dependency - it's a pure math library
        # for calculating technical indicators (SMA, RSI, MACD, etc.)
        try:
            self.feature_engineer = GPUFeatureEngineer(use_gpu=use_gpu, use_feature_selection=False)
            if use_gpu and hasattr(self.feature_engineer, 'using_gpu') and not self.feature_engineer.using_gpu:
                logger.warning("[GPU] GPU not available, using CPU feature extraction")
        except Exception as e:
            logger.warning(f"[GPU] Failed to initialize GPU feature engineer: {e}")
            self.feature_engineer = GPUFeatureEngineer(use_gpu=False, use_feature_selection=False)

        logger.info("[INIT] TurboMode Feature Extractor initialized")
        logger.info(f"       GPU Acceleration: {use_gpu}")

    def extract_features_for_date(self,
                                   symbol: str,
                                   target_date: str,
                                   lookback_days: int = 365) -> Optional[Dict[str, Any]]:
        """
        Extract 179 features for a specific date

        Args:
            symbol: Stock ticker (canonical format)
            target_date: Date to extract features for (YYYY-MM-DD)
            lookback_days: Days of history needed for feature calculation

        Returns:
            Dictionary of features, or None if data unavailable
        """
        # Calculate start date (need history for indicators)
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        start_dt = target_dt - timedelta(days=lookback_days)
        start_date = start_dt.strftime('%Y-%m-%d')

        # Fetch price data from Master Market Data DB
        try:
            price_data = self.market_data_api.get_candles(
                symbol=symbol,
                start_date=start_date,
                end_date=target_date,
                timeframe='1d'
            )
        except Exception as e:
            logger.error(f"[ERROR] Failed to fetch price data for {symbol} on {target_date}: {e}")
            return None

        if price_data is None or len(price_data) == 0:
            logger.warning(f"[WARN] No price data for {symbol} on {target_date}")
            return None

        # Convert DataFrame index (timestamp) to columns
        price_data = price_data.reset_index()
        if 'timestamp' in price_data.columns:
            price_data.rename(columns={'timestamp': 'date'}, inplace=True)

        # Ensure we have data for the target date (or nearest trading day before it)
        price_data['date'] = pd.to_datetime(price_data['date'])
        target_date_dt = pd.to_datetime(target_date)

        # Find the last trading day on or before the target date
        valid_dates = price_data[price_data['date'] <= target_date_dt]['date']

        if len(valid_dates) == 0:
            logger.warning(f"[WARN] No price data before {target_date} for {symbol}")
            return None

        # Use the last available trading day (handles weekends/holidays)
        actual_date = valid_dates.max()

        # Filter data up to the actual date (in case API returned extra future data)
        price_data = price_data[price_data['date'] <= actual_date].copy()

        # Extract features using GPU Feature Engineer
        try:
            features_dict = self.feature_engineer.extract_features(price_data)

            if features_dict is None or len(features_dict) == 0:
                logger.error(f"[ERROR] Feature extraction returned empty for {symbol} on {target_date}")
                return None

            # Add metadata
            features_dict['symbol'] = symbol
            features_dict['timestamp'] = target_date
            features_dict['feature_count'] = len([k for k in features_dict.keys()
                                                   if k not in ['symbol', 'timestamp', 'feature_count']])

            return features_dict

        except Exception as e:
            logger.error(f"[ERROR] Feature extraction failed for {symbol} on {target_date}: {e}")
            return None

    def features_to_json(self, features_dict: Dict[str, Any]) -> str:
        """
        Serialize features to JSON string

        Args:
            features_dict: Dictionary of features

        Returns:
            JSON string
        """
        # Convert numpy types to Python types for JSON serialization
        clean_dict = {}
        for key, value in features_dict.items():
            if isinstance(value, (np.integer, np.floating)):
                clean_dict[key] = float(value)
            elif isinstance(value, np.ndarray):
                clean_dict[key] = value.tolist()
            elif isinstance(value, (int, float, str, bool, type(None))):
                clean_dict[key] = value
            else:
                clean_dict[key] = str(value)

        return json.dumps(clean_dict)

    def json_to_features(self, json_str: str) -> Dict[str, Any]:
        """
        Deserialize JSON string to features dict

        Args:
            json_str: JSON string

        Returns:
            Dictionary of features
        """
        return json.loads(json_str)


if __name__ == '__main__':
    # Test the feature extractor
    print("Testing TurboMode Feature Extractor")
    print("=" * 80)

    extractor = TurboModeFeatureExtractor(use_gpu=True)

    # Test with AAPL on a recent date
    test_symbol = 'AAPL'
    test_date = '2024-01-15'

    print(f"\nExtracting features for {test_symbol} on {test_date}...")

    features = extractor.extract_features_for_date(
        symbol=test_symbol,
        target_date=test_date,
        lookback_days=365
    )

    if features:
        print(f"\n[SUCCESS] Extracted {len(features)} feature keys")
        print(f"\nFeature keys (first 20):")
        for i, key in enumerate(list(features.keys())[:20], 1):
            print(f"  {i}. {key}: {features[key]}")

        # Test JSON serialization
        json_str = extractor.features_to_json(features)
        print(f"\n[OK] JSON serialization successful ({len(json_str)} bytes)")

        # Test deserialization
        features_recovered = extractor.json_to_features(json_str)
        print(f"[OK] JSON deserialization successful ({len(features_recovered)} keys)")
    else:
        print("[ERROR] Feature extraction failed")

    print("\n" + "=" * 80)
    print("[OK] TurboMode Feature Extractor test complete!")
