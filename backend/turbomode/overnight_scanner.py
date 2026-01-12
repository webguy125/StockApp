"""
TurboMode Overnight Scanner
Scans entire S&P 500 and generates ML-based trading signals

Run this script nightly to generate fresh signals for the next trading day
"""

import sys
import os

# Add project root to path for master_market_data and backend imports
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(backend_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
import torch

# Import Master Market Data DB API (read-only, shared data source)
from master_market_data.market_data_api import get_market_data_api

# Import curated 80-stock symbol list from ML training system
from backend.turbomode.core_symbols import (
    CORE_SYMBOLS,
    SECTOR_CODES
)

# Import database
from turbomode.database_schema import TurboModeDB

# Import ML system (NEW 8-model ensemble)
from backend.turbomode.models.xgboost_model import XGBoostModel
from backend.turbomode.models.xgboost_et_model import XGBoostETModel
from backend.turbomode.models.lightgbm_model import LightGBMModel
from backend.turbomode.models.catboost_model import CatBoostModel
from backend.turbomode.models.xgboost_hist_model import XGBoostHistModel
from backend.turbomode.models.xgboost_dart_model import XGBoostDARTModel
from backend.turbomode.models.xgboost_gblinear_model import XGBoostGBLinearModel
from backend.turbomode.models.xgboost_approx_model import XGBoostApproxModel
from backend.turbomode.models.meta_learner import MetaLearner

# Import feature engine
from backend.turbomode.turbomode_vectorized_feature_engine import TurboModeVectorizedFeatureEngine


class OvernightScanner:
    """
    Overnight scanner for 80 curated stocks using 8-model GPU ensemble

    Features:
    - Scans 80 curated stocks (balanced across sectors and market caps)
    - Uses 8-model GPU ensemble (71.29% test accuracy on 10%/10% thresholds)
    - Excludes symbols with active signals
    - Generates BUY/SELL predictions with confidence
    - Saves top signals to database
    - Updates sector statistics
    """

    def __init__(self, db_path: str = "backend/data/turbomode.db",
                 model_path: str = "backend/data/turbomode_models"):
        """
        Initialize scanner

        Args:
            db_path: Path to TurboMode database
            model_path: Path to TurboMode models directory
        """
        # Convert paths to absolute if they're relative
        if not os.path.isabs(db_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            db_path = os.path.join(project_root, db_path)

        if not os.path.isabs(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            model_path = os.path.join(project_root, model_path)

        self.db = TurboModeDB(db_path=db_path)
        self.model_path = model_path

        # Initialize Master Market Data DB API (read-only, shared data source)
        self.market_data_api = get_market_data_api()
        print(f"[INIT] Connected to Master Market Data DB (read-only)")

        # Initialize ML models (TurboMode-specific)
        print(f"[INIT] Loading TurboMode ML models from {model_path}...")
        self._load_models()

        # Initialize feature engineering (ALL 179 features)
        # Models were trained on all 179 features, so we must extract all 179
        self.feature_engineer = TurboModeVectorizedFeatureEngine()

        print(f"[OK] Scanner initialized with TurboMode models (179 features)")

    def _get_all_symbols(self) -> List[str]:
        """Get list of all 80 curated symbols from CORE_SYMBOLS"""
        all_symbols = []
        for sector, caps in CORE_SYMBOLS.items():
            for cap_size, symbols in caps.items():
                all_symbols.extend(symbols)
        return sorted(all_symbols)

    def _get_sector_for_symbol(self, symbol: str) -> Optional[str]:
        """Get sector for a symbol"""
        for sector, caps in CORE_SYMBOLS.items():
            for cap_size, symbols in caps.items():
                if symbol in symbols:
                    return sector
        return None

    def _get_cap_size_for_symbol(self, symbol: str) -> Optional[str]:
        """Get market cap category for a symbol"""
        for sector, caps in CORE_SYMBOLS.items():
            for cap_size, symbols in caps.items():
                if symbol in symbols:
                    return cap_size
        return None

    def _load_models(self):
        """Load all trained ML models from TurboMode directory (NEW 8-model GPU ensemble)"""
        import os

        # Initialize NEW 8-model GPU ensemble with TurboMode paths
        # Note: All models use GPU via device='cuda' in their hyperparameters
        self.xgb_model = XGBoostModel(model_path=os.path.join(self.model_path, "xgboost"), use_gpu=True)
        self.xgb_et_model = XGBoostETModel(model_path=os.path.join(self.model_path, "xgboost_et"))
        self.lgbm_model = LightGBMModel(model_path=os.path.join(self.model_path, "lightgbm"))
        self.catboost_model = CatBoostModel(model_path=os.path.join(self.model_path, "catboost"))
        self.xgb_hist_model = XGBoostHistModel(model_path=os.path.join(self.model_path, "xgboost_hist"))
        self.xgb_dart_model = XGBoostDARTModel(model_path=os.path.join(self.model_path, "xgboost_dart"))
        self.xgb_gblinear_model = XGBoostGBLinearModel(model_path=os.path.join(self.model_path, "xgboost_gblinear"))
        self.xgb_approx_model = XGBoostApproxModel(model_path=os.path.join(self.model_path, "xgboost_approx"))
        self.meta_learner = MetaLearner(model_path=os.path.join(self.model_path, "meta_learner_v2"))

        # Load all 8 base models (load() raises exceptions on failure)
        self.xgb_model.load()
        self.xgb_et_model.load()
        self.lgbm_model.load()
        self.catboost_model.load()
        self.xgb_hist_model.load()
        self.xgb_dart_model.load()
        self.xgb_gblinear_model.load()
        self.xgb_approx_model.load()

        # Load meta-learner
        self.meta_learner.load()

        # All models loaded successfully
        print(f"[OK] Loaded 8 base models + meta-learner from {self.model_path}")

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current/latest closing price for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Latest close price, or None if failed
        """
        try:
            # Get last 5 days from Master Market Data DB
            df = self.market_data_api.get_candles(symbol, timeframe='1d', days_back=5)

            if df is None or df.empty:
                return None

            # Get most recent close
            return float(df['close'].iloc[-1])

        except Exception as e:
            print(f"  [WARNING] Failed to get price for {symbol}: {e}")
            return None

    def extract_features(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Extract 179 features for ML prediction (176 technical + 3 metadata)

        Args:
            symbol: Stock symbol

        Returns:
            Feature dictionary with 179 features, or None if failed
        """
        try:
            # Get historical data from Master Market Data DB (need at least 400 rows for indicators)
            df = self.market_data_api.get_candles(symbol, timeframe='1d', days_back=730)  # 2 years

            if df is None or df.empty or len(df) < 400:
                return None

            # Normalize column names to match feature engineer expectations
            # Master DB uses lowercase (timestamp, open, high, low, close, volume)
            # Feature engineer expects lowercase (date, open, high, low, close, volume)
            df = df.reset_index()
            df.rename(columns={
                'timestamp': 'date'
            }, inplace=True)

            # Extract features for entire DataFrame (vectorized)
            features_df = self.feature_engineer.extract_features(df)

            if features_df is None or features_df.empty:
                return None

            # Get the LAST row (most recent features) and convert to dict
            features = features_df.iloc[-1].to_dict()

            # Add 3 metadata features (sector_code, market_cap_tier, symbol_hash)
            from backend.turbomode.core_symbols import get_symbol_metadata
            metadata = get_symbol_metadata(symbol)
            features.update(metadata)

            # Now we have 176 + 3 = 179 features total
            return features

        except Exception as e:
            print(f"  [WARNING] Failed to extract features for {symbol}: {e}")
            return None

    def get_prediction(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Get ML prediction from meta-learner (8-model GPU ensemble)

        Args:
            features: Feature dictionary (179 technical features + metadata)

        Returns:
            Prediction dictionary with:
            - prediction: 'buy', 'hold', or 'sell'
            - buy_prob: float
            - sell_prob: float
            - confidence: float
        """
        # Convert feature dict to numpy array
        # Extract only the 179 technical features (exclude metadata like sector_code, market_cap_tier, symbol_hash)
        from backend.turbomode.feature_list import FEATURE_LIST

        # Build feature array in canonical order
        feature_array = np.array([features.get(f, 0.0) for f in FEATURE_LIST], dtype=np.float32)

        # OPTION B2 PIPELINE: 8 Base Models -> Directional Override (per-model) -> Final Meta-Learner

        # Step 1: Get predictions from all 8 base models
        xgb_pred = self.xgb_model.predict(feature_array)
        xgb_et_pred = self.xgb_et_model.predict(feature_array)
        lgbm_pred = self.lgbm_model.predict(feature_array)
        catboost_pred = self.catboost_model.predict(feature_array)
        xgb_hist_pred = self.xgb_hist_model.predict(feature_array)
        xgb_dart_pred = self.xgb_dart_model.predict(feature_array)
        xgb_gblinear_pred = self.xgb_gblinear_model.predict(feature_array)
        xgb_approx_pred = self.xgb_approx_model.predict(feature_array)

        # Step 2: Convert to format for directional override layer
        ensemble_outputs = [
            {'prob_down': float(xgb_pred[0]), 'prob_neutral': float(xgb_pred[1]), 'prob_up': float(xgb_pred[2])},
            {'prob_down': float(xgb_et_pred[0]), 'prob_neutral': float(xgb_et_pred[1]), 'prob_up': float(xgb_et_pred[2])},
            {'prob_down': float(lgbm_pred[0]), 'prob_neutral': float(lgbm_pred[1]), 'prob_up': float(lgbm_pred[2])},
            {'prob_down': float(catboost_pred[0]), 'prob_neutral': float(catboost_pred[1]), 'prob_up': float(catboost_pred[2])},
            {'prob_down': float(xgb_hist_pred[0]), 'prob_neutral': float(xgb_hist_pred[1]), 'prob_up': float(xgb_hist_pred[2])},
            {'prob_down': float(xgb_dart_pred[0]), 'prob_neutral': float(xgb_dart_pred[1]), 'prob_up': float(xgb_dart_pred[2])},
            {'prob_down': float(xgb_gblinear_pred[0]), 'prob_neutral': float(xgb_gblinear_pred[1]), 'prob_up': float(xgb_gblinear_pred[2])},
            {'prob_down': float(xgb_approx_pred[0]), 'prob_neutral': float(xgb_approx_pred[1]), 'prob_up': float(xgb_approx_pred[2])},
        ]

        # Step 3: Apply directional override to each model individually
        from backend.turbomode.directional_override import apply_override_to_each_model
        override_result = apply_override_to_each_model(ensemble_outputs)
        adjusted_outputs = override_result['adjusted_outputs']

        # Step 4: Convert adjusted outputs back to format for final meta-learner (55 features)
        adjusted_predictions = {}
        model_names = ['xgboost', 'xgboost_et', 'lightgbm', 'catboost',
                      'xgboost_hist', 'xgboost_dart', 'xgboost_gblinear', 'xgboost_approx']

        # First, add the 24 base probability features
        for i, model_name in enumerate(model_names):
            adj_out = adjusted_outputs[i]
            # Convert to numpy array [prob_down, prob_neutral, prob_up]
            adjusted_predictions[model_name] = np.array([
                adj_out['prob_down'],
                adj_out['prob_neutral'],
                adj_out['prob_up']
            ])

        # Step 4b: Add 31 override-aware features (24 per-model + 7 aggregate)
        # Per-model features (24 total: 8 models × 3)
        for i, model_name in enumerate(model_names):
            adj_out = adjusted_outputs[i]
            prob_up = adj_out['prob_up']
            prob_down = adj_out['prob_down']
            prob_neutral = adj_out['prob_neutral']

            # Asymmetry between buy and sell
            adjusted_predictions[f'{model_name}_asymmetry'] = np.abs(prob_up - prob_down)

            # Maximum directional probability
            adjusted_predictions[f'{model_name}_max_directional'] = np.maximum(prob_up, prob_down)

            # Neutral dominance (how much neutral exceeds directional)
            adjusted_predictions[f'{model_name}_neutral_dominance'] = prob_neutral - np.maximum(prob_up, prob_down)

        # Aggregate features (7 total)
        asymmetries = [adjusted_predictions[f'{m}_asymmetry'] for m in model_names]
        max_directionals = [adjusted_predictions[f'{m}_max_directional'] for m in model_names]
        neutral_dominances = [adjusted_predictions[f'{m}_neutral_dominance'] for m in model_names]

        adjusted_predictions['avg_asymmetry'] = np.mean(asymmetries)
        adjusted_predictions['max_asymmetry'] = np.max(asymmetries)
        adjusted_predictions['avg_max_directional'] = np.mean(max_directionals)
        adjusted_predictions['avg_neutral_dominance'] = np.mean(neutral_dominances)

        # Consensus features (how many models favor each direction)
        up_probs = [adjusted_outputs[i]['prob_up'] for i in range(8)]
        down_probs = [adjusted_outputs[i]['prob_down'] for i in range(8)]

        models_favor_up = sum(1 for i in range(8) if up_probs[i] > down_probs[i])
        models_favor_down = sum(1 for i in range(8) if down_probs[i] > up_probs[i])

        adjusted_predictions['models_favor_up'] = models_favor_up
        adjusted_predictions['models_favor_down'] = models_favor_down
        adjusted_predictions['directional_consensus'] = np.abs(models_favor_up - models_favor_down) / 8.0

        # Step 5: Feed 55 features to final meta-learner_v2
        ensemble_pred = self.meta_learner.predict(adjusted_predictions)

        # Step 6: Convert to scanner format
        prob_down = ensemble_pred['prob_down']
        prob_neutral = ensemble_pred['prob_neutral']
        prob_up = ensemble_pred['prob_up']

        probs = {
            'sell': prob_down,
            'hold': prob_neutral,
            'buy': prob_up
        }

        prediction = max(probs, key=probs.get)
        confidence = probs[prediction]

        return {
            'prediction': prediction,
            'confidence': confidence,
            'prob_down': prob_down,
            'prob_neutral': prob_neutral,
            'prob_up': prob_up,
            'sell_prob': prob_down,  # Alias for scanner compatibility
            'buy_prob': prob_up,     # Alias for scanner compatibility
            'override_count': override_result['override_count']
        }

    def is_stock_tradeable(self, symbol: str, df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Pre-filter to detect dead/flatlined/capped stocks before model prediction

        Filters out:
        - Low volume stocks (avg < 100K shares/day)
        - Flatlined stocks (price volatility < 2% over 30 days)
        - Stocks with collapsing volume (>50% drop in 30d vs 90d average)
        - Range-bound stocks stuck at resistance (< 3% from ceiling for 10+ days)

        Args:
            symbol: Stock symbol
            df: Optional pre-downloaded DataFrame (if None, will download)

        Returns:
            Dictionary with 'tradeable' (bool) and 'reason' (str)
        """
        try:
            # Get data from Master DB if not provided
            if df is None:
                df = self.market_data_api.get_candles(symbol, timeframe='1d', days_back=180)  # 6 months

            if df is None or df.empty or len(df) < 30:
                return {'tradeable': False, 'reason': 'Insufficient data'}

            # Filter 1: Average volume check (last 30 days)
            recent_volume = df['volume'].tail(30).mean()
            if recent_volume < 100_000:
                return {
                    'tradeable': False,
                    'reason': f'Low volume ({recent_volume:,.0f} shares/day)'
                }

            # Filter 2: Price volatility check (30-day range)
            recent_prices = df['close'].tail(30)
            price_min = recent_prices.min()
            price_max = recent_prices.max()
            volatility_pct = ((price_max - price_min) / price_min) * 100

            if volatility_pct < 2.0:
                return {
                    'tradeable': False,
                    'reason': f'Flatlined ({volatility_pct:.1f}% range in 30d)'
                }

            # Filter 3: Volume collapse check (30d vs 90d)
            if len(df) >= 90:
                volume_30d = df['volume'].tail(30).mean()
                volume_90d = df['volume'].tail(90).mean()

                volume_change_pct = ((volume_30d - volume_90d) / volume_90d) * 100

                if volume_change_pct < -50:  # Volume dropped >50%
                    return {
                        'tradeable': False,
                        'reason': f'Volume collapse ({volume_change_pct:.1f}%)'
                    }

            # Filter 4: Range-bound at resistance check (buyout targets, strong resistance)
            # Detect stocks stuck near a ceiling for extended periods
            # Example: EXAS at $101-102 with $105 buyout cap (3% max gain)
            if len(df) >= 20:
                # Look at last 20 days for pattern detection
                last_20_days = df.tail(20)
                last_20_high = last_20_days['high'].max()
                last_20_low = last_20_days['low'].min()

                # Also check last 10 days specifically
                last_10_days = df.tail(10)
                last_10_high = last_10_days['high'].max()
                last_10_low = last_10_days['low'].min()
                last_10_range_pct = ((last_10_high - last_10_low) / last_10_low) * 100

                current_price = df['close'].iloc[-1]

                # Key indicator: Stock is stuck if:
                # 1. The 20-day high and 10-day high are VERY close (< 1% apart)
                #    This means the ceiling hasn't moved in 20 days
                # 2. The 10-day range is VERY tight (< 3%)
                #    This means it's consolidating, not breaking out
                # 3. Current price is near that ceiling (< 2% away)

                ceiling_stability = abs(last_20_high - last_10_high) / last_20_high * 100

                if ceiling_stability < 1.0 and last_10_range_pct < 3.0:
                    distance_from_ceiling = ((last_10_high - current_price) / current_price) * 100

                    if distance_from_ceiling < 2.0:
                        # Calculate max possible gain to ceiling
                        max_gain_to_ceiling = ((last_10_high - current_price) / current_price) * 100

                        # If max gain is < 6% (can't reach 10% target), filter out
                        if max_gain_to_ceiling < 6.0:
                            return {
                                'tradeable': False,
                                'reason': f'Stuck at resistance (${last_10_high:.2f}, only {max_gain_to_ceiling:.1f}% upside, {last_10_range_pct:.1f}% 10d range)'
                            }

            # All filters passed
            return {
                'tradeable': True,
                'reason': f'OK (vol: {recent_volume:,.0f}, volatility: {volatility_pct:.1f}%)'
            }

        except Exception as e:
            return {'tradeable': False, 'reason': f'Filter error: {e}'}

    def get_prediction_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get prediction for a symbol WITHOUT filtering by confidence threshold
        Used for the "All Predictions" page

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with prediction details (prediction, confidence, price, sector, market_cap)
        """
        try:
            # Get current price
            current_price = self.get_current_price(symbol)
            if current_price is None:
                return None

            # Extract features
            features = self.extract_features(symbol)
            if features is None:
                return None

            # Get prediction
            prediction = self.get_prediction(features)

            # Get metadata
            sector = self._get_sector_for_symbol(symbol)
            market_cap = self._get_cap_size_for_symbol(symbol)

            return {
                'symbol': symbol,
                'prediction': prediction['prediction'],  # 'buy', 'sell', or 'hold'
                'confidence': float(prediction['confidence']),
                'prob_down': float(prediction['prob_down']),
                'prob_neutral': float(prediction['prob_neutral']),
                'prob_up': float(prediction['prob_up']),
                'current_price': float(current_price),
                'sector': sector if sector else 'unknown',
                'market_cap_category': market_cap if market_cap else 'unknown'
            }

        except Exception as e:
            print(f"[ERROR] Failed to get prediction for {symbol}: {e}")
            return None

    def scan_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Scan a single symbol and generate signal if applicable

        Args:
            symbol: Stock symbol

        Returns:
            Signal dictionary if BUY/SELL with high confidence, else None
        """
        # PRE-FILTER: Check if stock is tradeable (not dead/flatlined)
        # Get data once from Master DB and reuse for both filter and feature extraction
        try:
            df_6mo = self.market_data_api.get_candles(symbol, timeframe='1d', days_back=180)  # 6 months

            # Run pre-filter
            tradeable_check = self.is_stock_tradeable(symbol, df=df_6mo)

            if not tradeable_check['tradeable']:
                print(f"  [FILTERED] {symbol}: {tradeable_check['reason']}")
                return None

        except Exception as e:
            print(f"  [WARNING] Failed to check tradeability for {symbol}: {e}")
            return None

        # Get current price
        current_price = self.get_current_price(symbol)
        if current_price is None:
            return None

        # Extract features
        features = self.extract_features(symbol)
        if features is None:
            return None

        # Get prediction
        prediction = self.get_prediction(features)

        # Log override decision to audit file (non-blocking)
        try:
            from backend.turbomode.override_audit_logger import log_override_decision
            from datetime import datetime

            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'prob_buy': prediction['prob_up'],
                'prob_hold': prediction['prob_neutral'],
                'prob_sell': prediction['prob_down'],
                'override_triggered': prediction['override_count'] > 0,
                'final_prediction': prediction['prediction'],
                'override_count': prediction['override_count'],
                'entry_price': current_price
            }
            log_override_decision(audit_entry)
        except Exception as e:
            # Don't let logging errors break the scanner
            pass

        # Symmetric confidence thresholds (testing if market truly has no bearish setups)
        BUY_CONFIDENCE = 0.65   # ≥65% confidence for BUY signals (long positions)
        SELL_CONFIDENCE = 0.65  # ≥65% confidence for SELL signals (short/puts - same as BUY for testing)

        signal_type = None
        if prediction['prediction'] == 'buy' and prediction['confidence'] >= BUY_CONFIDENCE:
            signal_type = 'BUY'
        elif prediction['prediction'] == 'sell' and prediction['confidence'] >= SELL_CONFIDENCE:
            signal_type = 'SELL'
        else:
            # HOLD or low confidence - skip
            return None

        # Calculate entry range (±2% tolerance for morning gaps)
        # Signal price is based on close, but morning open can gap
        # Accept entries within 2% of signal price
        entry_min = current_price * 0.98  # -2% (acceptable gap down for BUY)
        entry_max = current_price * 1.02  # +2% (acceptable gap up for BUY)

        # Calculate targets and stops
        if signal_type == 'BUY':
            target_price = current_price * 1.10  # +10%
            stop_price = current_price * 0.95    # -5%
        else:  # SELL
            target_price = current_price * 0.90  # -10%
            stop_price = current_price * 1.05    # +5%

        # Get classifications from 80-stock curated list
        market_cap = self._get_cap_size_for_symbol(symbol)
        sector = self._get_sector_for_symbol(symbol)

        if market_cap is None or sector is None:
            # Symbol not in our 80-stock curated list (shouldn't happen)
            return None

        return {
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': prediction['confidence'],
            'entry_date': datetime.now().strftime('%Y-%m-%d'),
            'entry_price': current_price,
            'entry_min': entry_min,   # Minimum acceptable entry (2% below signal)
            'entry_max': entry_max,   # Maximum acceptable entry (2% above signal)
            'target_price': target_price,
            'stop_price': stop_price,
            'market_cap': market_cap,
            'sector': sector,

            # Additional info for logging (binary classification: BUY/SELL only)
            'buy_prob': prediction['buy_prob'],
            'sell_prob': prediction['sell_prob']
        }

    def scan_all(self, max_signals_per_type: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan all 80 curated stocks

        Args:
            max_signals_per_type: Maximum BUY and SELL signals to save (default 100)

        Returns:
            Dictionary with 'buy_signals' and 'sell_signals' lists
        """
        print("\n" + "=" * 70)
        print("TURBOMODE OVERNIGHT SCANNER (80 Curated Stocks)")
        print("=" * 70)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Update age of existing signals
        print("\n[STEP 1] Updating existing signal ages...")
        expired_count = self.db.update_signal_age()

        # Get symbols to exclude (already have active signals)
        print("\n[STEP 2] Getting active signals to exclude...")
        active_symbols = set(self.db.get_active_symbols())
        print(f"  Excluding {len(active_symbols)} symbols with active signals")

        # Get all 80 curated symbols
        print("\n[STEP 3] Loading 80 curated stock list...")
        all_symbols = self._get_all_symbols()
        print(f"  Total symbols: {len(all_symbols)}")

        # Filter out active symbols
        symbols_to_scan = [s for s in all_symbols if s not in active_symbols]
        print(f"  Symbols to scan: {len(symbols_to_scan)}")

        # Scan all symbols
        print(f"\n[STEP 4] Scanning {len(symbols_to_scan)} symbols...")
        print("  This will take approximately 3-5 minutes (GPU-accelerated)...")

        buy_signals = []
        sell_signals = []
        scanned = 0
        failed = 0

        for i, symbol in enumerate(symbols_to_scan, 1):
            # Progress indicator every 50 symbols
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(symbols_to_scan)} ({i/len(symbols_to_scan)*100:.1f}%) - "
                      f"BUY: {len(buy_signals)}, SELL: {len(sell_signals)}")

                # Clear GPU memory every 50 symbols to prevent memory exhaustion
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print(f"    [GPU] Cleared GPU cache at symbol {i}")

            signal = self.scan_symbol(symbol)

            if signal is None:
                failed += 1
                continue

            scanned += 1

            if signal['signal_type'] == 'BUY':
                buy_signals.append(signal)
            else:
                sell_signals.append(signal)

        print(f"\n[STEP 5] Scan complete!")
        print(f"  Scanned: {scanned}/{len(symbols_to_scan)}")
        print(f"  Failed: {failed}")
        print(f"  BUY signals: {len(buy_signals)}")
        print(f"  SELL signals: {len(sell_signals)}")

        # Sort by confidence
        buy_signals.sort(key=lambda x: x['confidence'], reverse=True)
        sell_signals.sort(key=lambda x: x['confidence'], reverse=True)

        # Limit to top N signals
        buy_signals = buy_signals[:max_signals_per_type]
        sell_signals = sell_signals[:max_signals_per_type]

        print(f"\n[STEP 6] Saving signals to database (with smart replacement)...")
        print(f"  Processing {len(buy_signals)} BUY signals")
        print(f"  Processing {len(sell_signals)} SELL signals")

        # Save to database with smart replacement logic
        saved_buy = 0
        replaced_buy = 0
        saved_sell = 0
        replaced_sell = 0

        # Process BUY signals
        for signal in buy_signals:
            market_cap = signal['market_cap']
            signal_type = signal['signal_type']

            # Try to add signal normally
            if self.db.add_signal(signal):
                saved_buy += 1
                continue

            # Signal exists or limit reached - check if we should replace
            count = self.db.count_active_signals(market_cap, signal_type)

            if count >= 20:  # At limit for this market cap + type
                # Get weakest signal (time-decayed confidence)
                weakest = self.db.get_weakest_signal(market_cap, signal_type)

                if weakest:
                    # Calculate effective confidence for new signal (age=0)
                    new_effective = signal['confidence']  # Fresh signal, no decay

                    # Compare to weakest signal's effective confidence
                    if new_effective > weakest['effective_confidence']:
                        # Replace weakest with new signal
                        if self.db.replace_signal(weakest['symbol'], weakest['signal_type'], signal):
                            replaced_buy += 1
                            print(f"    [REPLACE] {signal['symbol']} ({new_effective:.1%}) replaces "
                                  f"{weakest['symbol']} (eff: {weakest['effective_confidence']:.1%}, "
                                  f"age: {weakest['age_days']}d)")

        # Process SELL signals
        for signal in sell_signals:
            market_cap = signal['market_cap']
            signal_type = signal['signal_type']

            # Try to add signal normally
            if self.db.add_signal(signal):
                saved_sell += 1
                continue

            # Signal exists or limit reached - check if we should replace
            count = self.db.count_active_signals(market_cap, signal_type)

            if count >= 20:  # At limit for this market cap + type
                # Get weakest signal (time-decayed confidence)
                weakest = self.db.get_weakest_signal(market_cap, signal_type)

                if weakest:
                    # Calculate effective confidence for new signal (age=0)
                    new_effective = signal['confidence']  # Fresh signal, no decay

                    # Compare to weakest signal's effective confidence
                    if new_effective > weakest['effective_confidence']:
                        # Replace weakest with new signal
                        if self.db.replace_signal(weakest['symbol'], weakest['signal_type'], signal):
                            replaced_sell += 1
                            print(f"    [REPLACE] {signal['symbol']} ({new_effective:.1%}) replaces "
                                  f"{weakest['symbol']} (eff: {weakest['effective_confidence']:.1%}, "
                                  f"age: {weakest['age_days']}d)")

        print(f"\n  Results:")
        print(f"    BUY:  {saved_buy} added, {replaced_buy} replaced")
        print(f"    SELL: {saved_sell} added, {replaced_sell} replaced")

        # Update sector statistics
        print(f"\n[STEP 7] Updating sector statistics...")
        self._update_sector_stats(buy_signals + sell_signals)

        # NEW: Save ALL predictions (not just signals above threshold) for "All Predictions" page
        print(f"\n[STEP 8] Generating complete predictions file for all 80 stocks...")
        self._save_all_predictions()

        print("\n" + "=" * 70)
        print("SCAN COMPLETE")
        print("=" * 70)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'stats': {
                'total_scanned': scanned,
                'total_failed': failed,
                'buy_count': len(buy_signals),
                'sell_count': len(sell_signals),
                'saved_buy': saved_buy,
                'replaced_buy': replaced_buy,
                'saved_sell': saved_sell,
                'replaced_sell': replaced_sell
            }
        }

    def _save_all_predictions(self):
        """
        Generate and save predictions for ALL 80 stocks (regardless of confidence threshold)
        Saves to JSON file for fast loading by "All Predictions" webpage
        """
        import json
        import os

        print("  Getting predictions for all 80 stocks...")
        all_symbols = self._get_all_symbols()
        all_predictions = []

        for i, symbol in enumerate(all_symbols, 1):
            if i % 10 == 0:
                print(f"    Progress: {i}/{len(all_symbols)} stocks...")

            pred = self.get_prediction_for_symbol(symbol)
            if pred:
                all_predictions.append(pred)

        # Sort by confidence descending
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        # Calculate statistics
        buy_count = len([p for p in all_predictions if p['prediction'] == 'buy'])
        sell_count = len([p for p in all_predictions if p['prediction'] == 'sell'])
        hold_count = len([p for p in all_predictions if p['prediction'] == 'hold'])
        threshold_count = len([p for p in all_predictions if p['confidence'] >= 0.65])

        # Save to JSON file
        output_path = os.path.join(os.path.dirname(__file__), 'data', 'all_predictions.json')
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'total': len(all_predictions),
            'statistics': {
                'buy_count': buy_count,
                'sell_count': sell_count,
                'hold_count': hold_count,
                'threshold_met': threshold_count,
                'threshold_not_met': len(all_predictions) - threshold_count
            },
            'predictions': all_predictions
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"  [OK] Saved {len(all_predictions)} predictions to all_predictions.json")
        print(f"    BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}")
        print(f"    Above 65% threshold: {threshold_count}")

    def _update_sector_stats(self, signals: List[Dict[str, Any]]):
        """
        Update sector statistics based on new signals

        Args:
            signals: List of signal dictionaries
        """
        # Aggregate by sector
        sector_data = defaultdict(lambda: {
            'buy_signals': [],
            'sell_signals': []
        })

        for signal in signals:
            sector = signal['sector']
            if signal['signal_type'] == 'BUY':
                sector_data[sector]['buy_signals'].append(signal)
            else:
                sector_data[sector]['sell_signals'].append(signal)

        # Calculate stats for each sector
        today = datetime.now().strftime('%Y-%m-%d')

        for sector, data in sector_data.items():
            buy_sigs = data['buy_signals']
            sell_sigs = data['sell_signals']

            avg_buy_conf = np.mean([s['confidence'] for s in buy_sigs]) if buy_sigs else 0.0
            avg_sell_conf = np.mean([s['confidence'] for s in sell_sigs]) if sell_sigs else 0.0

            # Determine sentiment
            if len(buy_sigs) > len(sell_sigs) * 1.5:
                sentiment = 'BULLISH'
            elif len(sell_sigs) > len(buy_sigs) * 1.5:
                sentiment = 'BEARISH'
            else:
                sentiment = 'NEUTRAL'

            stats = {
                'total_buy_signals': len(buy_sigs),
                'total_sell_signals': len(sell_sigs),
                'avg_buy_confidence': float(avg_buy_conf),
                'avg_sell_confidence': float(avg_sell_conf),
                'sentiment': sentiment
            }

            self.db.update_sector_stats(today, sector, stats)

        print(f"  Updated stats for {len(sector_data)} sectors")

    def print_top_signals(self, signals: Dict[str, List[Dict[str, Any]]], top_n: int = 10):
        """
        Print top N signals for review

        Args:
            signals: Dictionary with 'buy_signals' and 'sell_signals'
            top_n: Number of top signals to display
        """
        print(f"\n{'=' * 70}")
        print(f"TOP {top_n} BUY SIGNALS")
        print(f"{'=' * 70}")
        print(f"{'Symbol':<8} {'Sector':<25} {'Cap':<10} {'Confidence':<12} {'Entry':<10}")
        print("-" * 70)

        for signal in signals['buy_signals'][:top_n]:
            print(f"{signal['symbol']:<8} {signal['sector']:<25} {signal['market_cap']:<10} "
                  f"{signal['confidence']:>6.2%}      ${signal['entry_price']:>7.2f}")

        print(f"\n{'=' * 70}")
        print(f"TOP {top_n} SELL SIGNALS")
        print(f"{'=' * 70}")
        print(f"{'Symbol':<8} {'Sector':<25} {'Cap':<10} {'Confidence':<12} {'Entry':<10}")
        print("-" * 70)

        for signal in signals['sell_signals'][:top_n]:
            print(f"{signal['symbol']:<8} {signal['sector']:<25} {signal['market_cap']:<10} "
                  f"{signal['confidence']:>6.2%}      ${signal['entry_price']:>7.2f}")


if __name__ == '__main__':
    # Run overnight scan
    scanner = OvernightScanner()
    results = scanner.scan_all(max_signals_per_type=100)
    scanner.print_top_signals(results, top_n=20)

    print("\n[OK] Overnight scan complete!")
