"""
TurboMode Overnight Scanner
Scans entire S&P 500 and generates ML-based trading signals

Run this script nightly to generate fresh signals for the next trading day
"""

import sys
import os

# Add backend to path if not already there
if 'backend' not in [os.path.basename(p) for p in sys.path]:
    backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import yfinance as yf
import pandas as pd
import numpy as np
from collections import defaultdict
import torch

# Import curated 80-stock symbol list from ML training system
from advanced_ml.config.core_symbols import (
    CORE_SYMBOLS,
    SECTOR_CODES
)

# Import database
from turbomode.database_schema import TurboModeDB

# Import ML system (NEW 8-model GPU ensemble)
from advanced_ml.features.gpu_feature_engineer import GPUFeatureEngineer
from advanced_ml.models.xgboost_model import XGBoostModel
from advanced_ml.models.xgboost_et_model import XGBoostETModel
from advanced_ml.models.lightgbm_model import LightGBMModel
from advanced_ml.models.catboost_model import CatBoostModel
from advanced_ml.models.xgboost_hist_model import XGBoostHistModel
from advanced_ml.models.xgboost_dart_model import XGBoostDartModel
from advanced_ml.models.xgboost_gblinear_model import XGBoostGBLinearModel
from advanced_ml.models.xgboost_approx_model import XGBoostApproxModel
from advanced_ml.models.meta_learner import MetaLearner


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

        # Initialize ML models (TurboMode-specific)
        print(f"[INIT] Loading TurboMode ML models from {model_path}...")
        self._load_models()

        # Initialize feature engineering (GPU features - ALL 179 features, no selection)
        # Models were trained on all 179 features, so we must extract all 179
        self.feature_engineer = GPUFeatureEngineer(use_gpu=True, use_feature_selection=False)

        print(f"[OK] Scanner initialized with TurboMode models (179 features, GPU-accelerated)")

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
        self.xgb_dart_model = XGBoostDartModel(model_path=os.path.join(self.model_path, "xgboost_dart"))
        self.xgb_gblinear_model = XGBoostGBLinearModel(model_path=os.path.join(self.model_path, "xgboost_gblinear"))
        self.xgb_approx_model = XGBoostApproxModel(model_path=os.path.join(self.model_path, "xgboost_approx"))
        self.meta_learner = MetaLearner(model_path=os.path.join(self.model_path, "meta_learner"))

        # Load all 8 base models
        if not self.xgb_model.load():
            raise RuntimeError("Failed to load XGBoost model")
        if not self.xgb_et_model.load():
            raise RuntimeError("Failed to load XGBoost ET model")
        if not self.lgbm_model.load():
            raise RuntimeError("Failed to load LightGBM model")
        if not self.catboost_model.load():
            raise RuntimeError("Failed to load CatBoost model")
        if not self.xgb_hist_model.load():
            raise RuntimeError("Failed to load XGBoost Hist model")
        if not self.xgb_dart_model.load():
            raise RuntimeError("Failed to load XGBoost DART model")
        if not self.xgb_gblinear_model.load():
            raise RuntimeError("Failed to load XGBoost GBLinear model")
        if not self.xgb_approx_model.load():
            raise RuntimeError("Failed to load XGBoost Approx model")

        # Load meta-learner
        if not self.meta_learner.load():
            raise RuntimeError("Failed to load Meta-Learner model")

        # Register base models with meta-learner (MUST match training order!)
        self.meta_learner.register_base_model('xgboost', self.xgb_model)
        self.meta_learner.register_base_model('xgboost_et', self.xgb_et_model)
        self.meta_learner.register_base_model('lightgbm', self.lgbm_model)
        self.meta_learner.register_base_model('catboost', self.catboost_model)
        self.meta_learner.register_base_model('xgboost_hist', self.xgb_hist_model)
        self.meta_learner.register_base_model('xgboost_dart', self.xgb_dart_model)
        self.meta_learner.register_base_model('xgboost_gblinear', self.xgb_gblinear_model)
        self.meta_learner.register_base_model('xgboost_approx', self.xgb_approx_model)

        print(f"[OK] Loaded 8 GPU-accelerated base models + meta-learner from {self.model_path}")

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current/latest closing price for a symbol

        Args:
            symbol: Stock symbol

        Returns:
            Latest close price, or None if failed
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d")  # Last 5 days

            if hist.empty:
                return None

            # Get most recent close
            return float(hist['Close'].iloc[-1])

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
            # Download historical data (need at least 500 rows for vectorized approach)
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="2y", interval="1d")

            if df.empty or len(df) < 500:
                return None

            # Prepare data for feature engineering
            df = df.reset_index()

            # GPU batch feature extraction uses the LAST window (most recent data)
            # Extract features for index = len(df) - 1 (the most recent complete window)
            start_indices = [len(df) - 1]

            features_list = self.feature_engineer.extract_features_batch(df, start_indices, symbol)

            if not features_list or len(features_list) == 0:
                return None

            # Get the 176 technical features
            features = features_list[0]

            # Add 3 metadata features (sector_code, market_cap_tier, symbol_hash)
            from advanced_ml.config.symbol_metadata import get_symbol_metadata
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
            features: Feature dictionary (179 features)

        Returns:
            Prediction dictionary with:
            - prediction: 'buy', 'hold', or 'sell'
            - buy_prob: float
            - sell_prob: float
            - confidence: float
        """
        # Get predictions from all 8 base models
        base_predictions = {}

        base_predictions['xgboost'] = self.xgb_model.predict(features)
        base_predictions['xgboost_et'] = self.xgb_et_model.predict(features)
        base_predictions['lightgbm'] = self.lgbm_model.predict(features)
        base_predictions['catboost'] = self.catboost_model.predict(features)
        base_predictions['xgboost_hist'] = self.xgb_hist_model.predict(features)
        base_predictions['xgboost_dart'] = self.xgb_dart_model.predict(features)
        base_predictions['xgboost_gblinear'] = self.xgb_gblinear_model.predict(features)
        base_predictions['xgboost_approx'] = self.xgb_approx_model.predict(features)

        # Get meta-learner ensemble prediction
        ensemble_pred = self.meta_learner.predict(base_predictions)

        return ensemble_pred

    def scan_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Scan a single symbol and generate signal if applicable

        Args:
            symbol: Stock symbol

        Returns:
            Signal dictionary if BUY/SELL with high confidence, else None
        """
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
