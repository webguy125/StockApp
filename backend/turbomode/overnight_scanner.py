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

# Import S&P 500 symbols
from turbomode.sp500_symbols import (
    get_all_symbols,
    get_sector_for_symbol,
    get_cap_size_for_symbol
)

# Import database
from turbomode.database_schema import TurboModeDB

# Import ML system
from advanced_ml.training.training_pipeline import TrainingPipeline
from advanced_ml.features.feature_engineer import FeatureEngineer


class OvernightScanner:
    """
    Overnight S&P 500 scanner using ML meta-learner

    Features:
    - Scans all 500 S&P symbols
    - Excludes symbols with active signals
    - Generates BUY/SELL predictions with confidence
    - Saves top signals to database
    - Updates sector statistics
    """

    def __init__(self, db_path: str = "backend/data/turbomode.db",
                 ml_db_path: str = "backend/backend/data/advanced_ml_system.db"):
        """
        Initialize scanner

        Args:
            db_path: Path to TurboMode database
            ml_db_path: Path to ML system database (for feature engineering)
        """
        self.db = TurboModeDB(db_path=db_path)

        # Initialize ML pipeline
        print("[INIT] Loading ML models...")
        self.pipeline = TrainingPipeline(db_path=ml_db_path)

        # Load trained models
        self._load_models()

        # Initialize feature engineering (no events, baseline 179 features)
        self.feature_engineer = FeatureEngineer(enable_events=False)

        print(f"[OK] Scanner initialized")

    def _load_models(self):
        """Load all trained ML models"""
        # Load base models
        if not self.pipeline.rf_model.load():
            raise RuntimeError("Failed to load Random Forest model")
        if not self.pipeline.xgb_model.load():
            raise RuntimeError("Failed to load XGBoost model")
        if not self.pipeline.lgbm_model.load():
            raise RuntimeError("Failed to load LightGBM model")
        if not self.pipeline.et_model.load():
            raise RuntimeError("Failed to load Extra Trees model")
        if not self.pipeline.gb_model.load():
            raise RuntimeError("Failed to load Gradient Boosting model")
        if not self.pipeline.nn_model.load():
            raise RuntimeError("Failed to load Neural Network model")
        if not self.pipeline.lr_model.load():
            raise RuntimeError("Failed to load Logistic Regression model")
        if not self.pipeline.svm_model.load():
            raise RuntimeError("Failed to load SVM model")

        # Load meta-learner
        if not self.pipeline.meta_learner.load():
            raise RuntimeError("Failed to load Meta-Learner model")

        # Register base models with meta-learner
        self.pipeline.meta_learner.register_base_model('random_forest', self.pipeline.rf_model)
        self.pipeline.meta_learner.register_base_model('xgboost', self.pipeline.xgb_model)
        self.pipeline.meta_learner.register_base_model('lightgbm', self.pipeline.lgbm_model)
        self.pipeline.meta_learner.register_base_model('extratrees', self.pipeline.et_model)
        self.pipeline.meta_learner.register_base_model('gradientboost', self.pipeline.gb_model)
        self.pipeline.meta_learner.register_base_model('neural_network', self.pipeline.nn_model)
        self.pipeline.meta_learner.register_base_model('logistic_regression', self.pipeline.lr_model)
        self.pipeline.meta_learner.register_base_model('svm', self.pipeline.svm_model)

        print(f"[OK] Loaded 8 base models + meta-learner")

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
        Extract 179 features for ML prediction

        Args:
            symbol: Stock symbol

        Returns:
            Feature dictionary, or None if failed
        """
        try:
            # Download historical data (need enough for indicators)
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1y", interval="1d")

            if df.empty or len(df) < 200:
                return None

            # Prepare data for feature engineering
            df = df.reset_index()

            # Rename columns to lowercase (required by FeatureEngineer)
            df.columns = [c.lower() for c in df.columns]

            # Extract features using FeatureEngineer
            # Returns 179 technical features (events disabled)
            features = self.feature_engineer.extract_features(
                df,
                symbol=symbol
            )

            if not features:
                return None

            return features

        except Exception as e:
            print(f"  [WARNING] Failed to extract features for {symbol}: {e}")
            return None

    def get_prediction(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Get ML prediction from meta-learner

        Args:
            features: Feature dictionary (179 features)

        Returns:
            Prediction dictionary with:
            - prediction: 'buy', 'hold', or 'sell'
            - buy_prob: float
            - hold_prob: float
            - sell_prob: float
            - confidence: float
        """
        # Get predictions from all base models
        base_predictions = {}

        base_predictions['random_forest'] = self.pipeline.rf_model.predict(features)
        base_predictions['xgboost'] = self.pipeline.xgb_model.predict(features)
        base_predictions['lightgbm'] = self.pipeline.lgbm_model.predict(features)
        base_predictions['extratrees'] = self.pipeline.et_model.predict(features)
        base_predictions['gradientboost'] = self.pipeline.gb_model.predict(features)
        base_predictions['neural_network'] = self.pipeline.nn_model.predict(features)
        base_predictions['logistic_regression'] = self.pipeline.lr_model.predict(features)
        base_predictions['svm'] = self.pipeline.svm_model.predict(features)

        # Get meta-learner ensemble prediction
        ensemble_pred = self.pipeline.meta_learner.predict(base_predictions)

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

        # Filter for BUY or SELL with confidence >= 75%
        MIN_CONFIDENCE = 0.75

        signal_type = None
        if prediction['prediction'] == 'buy' and prediction['confidence'] >= MIN_CONFIDENCE:
            signal_type = 'BUY'
        elif prediction['prediction'] == 'sell' and prediction['confidence'] >= MIN_CONFIDENCE:
            signal_type = 'SELL'
        else:
            # HOLD or low confidence - skip
            return None

        # Calculate targets and stops
        if signal_type == 'BUY':
            target_price = current_price * 1.10  # +10%
            stop_price = current_price * 0.95    # -5%
        else:  # SELL
            target_price = current_price * 0.90  # -10%
            stop_price = current_price * 1.05    # +5%

        # Get classifications
        market_cap = get_cap_size_for_symbol(symbol)
        sector = get_sector_for_symbol(symbol)

        if market_cap is None or sector is None:
            # Symbol not in our S&P 500 list (shouldn't happen)
            return None

        return {
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': prediction['confidence'],
            'entry_date': datetime.now().strftime('%Y-%m-%d'),
            'entry_price': current_price,
            'target_price': target_price,
            'stop_price': stop_price,
            'market_cap': market_cap,
            'sector': sector,

            # Additional info for logging
            'buy_prob': prediction['buy_prob'],
            'hold_prob': prediction['hold_prob'],
            'sell_prob': prediction['sell_prob']
        }

    def scan_all(self, max_signals_per_type: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan all S&P 500 symbols

        Args:
            max_signals_per_type: Maximum BUY and SELL signals to save (default 100)

        Returns:
            Dictionary with 'buy_signals' and 'sell_signals' lists
        """
        print("\n" + "=" * 70)
        print("TURBOMODE OVERNIGHT SCANNER")
        print("=" * 70)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Update age of existing signals
        print("\n[STEP 1] Updating existing signal ages...")
        expired_count = self.db.update_signal_age()

        # Get symbols to exclude (already have active signals)
        print("\n[STEP 2] Getting active signals to exclude...")
        active_symbols = set(self.db.get_active_symbols())
        print(f"  Excluding {len(active_symbols)} symbols with active signals")

        # Get all S&P 500 symbols
        print("\n[STEP 3] Loading S&P 500 symbol list...")
        all_symbols = get_all_symbols()
        print(f"  Total symbols: {len(all_symbols)}")

        # Filter out active symbols
        symbols_to_scan = [s for s in all_symbols if s not in active_symbols]
        print(f"  Symbols to scan: {len(symbols_to_scan)}")

        # Scan all symbols
        print(f"\n[STEP 4] Scanning {len(symbols_to_scan)} symbols...")
        print("  This will take approximately 20-30 minutes...")

        buy_signals = []
        sell_signals = []
        scanned = 0
        failed = 0

        for i, symbol in enumerate(symbols_to_scan, 1):
            # Progress indicator every 50 symbols
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(symbols_to_scan)} ({i/len(symbols_to_scan)*100:.1f}%) - "
                      f"BUY: {len(buy_signals)}, SELL: {len(sell_signals)}")

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
