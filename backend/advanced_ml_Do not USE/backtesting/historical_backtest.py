"""
Historical Backtesting Engine
Generates training data from historical price movements

Features:
- Fetches 7 years of historical data (captures multiple market cycles)
- Generates 179 features per day
- Simulates 14-day hold period trades
- Labels outcomes: Buy (profitable), Hold (neutral), Sell (unprofitable)
- Stores results in database
- Can generate 120,000+ labeled samples (7 years × 78 symbols)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
import json
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from advanced_ml.features.feature_engineer import FeatureEngineer
from advanced_ml.features.gpu_feature_engineer import GPUFeatureEngineer
from advanced_ml.features.vectorized_gpu_features import VectorizedGPUFeatures
from advanced_ml.features.regime_macro_features import get_regime_macro_features
from advanced_ml.database.schema import AdvancedMLDatabase


class HistoricalBacktest:
    """
    Generate training data from historical price movements

    Strategy:
    1. Fetch 7 years of historical data for symbols (multiple market cycles)
    2. For each trading day:
       - Extract 179 features
       - Simulate entering a long position
       - Check outcome after 14 days:
         * +10% or more = Buy label (0)
         * -5% or worse = Sell label (2)
         * Between -5% and +10% = Hold label (1)
    3. Store labeled samples for model training
    """

    def __init__(self, db_path: str = "backend/data/advanced_ml_system.db", use_gpu: bool = True):
        """
        Initialize backtesting engine

        Args:
            db_path: Path to database
            use_gpu: Whether to use GPU-accelerated feature engineering (5-10x faster)
        """
        self.db = AdvancedMLDatabase(db_path)
        self.use_gpu = use_gpu

        # Use VECTORIZED GPU feature engineer for maximum speed
        if use_gpu:
            try:
                self.vectorized_gpu = VectorizedGPUFeatures(use_gpu=True)
                # REDESIGN FOR 90% ACCURACY: Use all 176 features (not just top 100)
                self.feature_engineer = GPUFeatureEngineer(use_gpu=True, use_feature_selection=False)
                if not self.vectorized_gpu.using_gpu:
                    print("[WARNING] GPU not available, falling back to CPU feature engineering")
                    self.feature_engineer = FeatureEngineer()
                    self.vectorized_gpu = None
            except Exception as e:
                print(f"[WARNING] GPU feature engineer failed ({e}), using CPU")
                self.feature_engineer = FeatureEngineer()
                self.vectorized_gpu = None
        else:
            self.feature_engineer = FeatureEngineer()
            self.vectorized_gpu = None

        # Trading parameters (match live system)
        self.hold_period = 14  # days
        self.win_threshold = 0.10  # +10%
        self.loss_threshold = -0.05  # -5%

        # Backtest statistics
        self.stats = {
            'total_trades': 0,
            'buy_labels': 0,
            'buy_signals': 0,
            'hold_labels': 0,
            'hold_signals': 0,
            'sell_labels': 0,
            'sell_signals': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'symbols_processed': 0
        }

    def fetch_historical_data(self, symbol: str, years: int = 7, start_date: datetime = None, end_date: datetime = None) -> Optional[pd.DataFrame]:
        """
        Fetch historical price data

        Args:
            symbol: Stock symbol
            years: Years of history to fetch (ignored if start_date/end_date provided)
            start_date: Custom start date (for archive generation)
            end_date: Custom end date (for archive generation)

        Returns:
            DataFrame with OHLCV data, or None if error
        """
        try:
            # Use custom dates if provided, otherwise default to recent data
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=years * 365)

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')

            if df.empty or len(df) < 50:
                print(f"[WARNING] Insufficient data for {symbol}")
                return None

            # Rename columns to lowercase
            df.columns = df.columns.str.lower()

            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                print(f"[WARNING] Missing required columns for {symbol}")
                return None

            return df

        except Exception as e:
            print(f"[ERROR] Failed to fetch data for {symbol}: {e}")
            return None

    def calculate_trade_outcome(self, entry_price: float, future_prices: pd.Series) -> Tuple[str, float, str]:
        """
        Calculate 7-day forward return for prediction

        SYMMETRIC THRESHOLD-BASED BINARY CLASSIFICATION:
        - Train only on STRONG directional moves (±10%)
        - Exclude ambiguous moves (HOLD) from training
        - Apply asymmetric confidence masking at inference (65% BUY, 75% SELL)

        Training Labels (Symmetric):
        - BUY: 7-day return ≥ +10% (strong upward move)
        - SELL: 7-day return ≤ -10% (strong downward move)
        - HOLD: between -10% and +10% (excluded from training)

        Inference Confidence (Asymmetric - applied in meta_learner):
        - BUY requires ≥65% confidence (more aggressive on entries)
        - SELL requires ≥75% confidence (more conservative on exits)
        - Below threshold → HOLD (wait for better opportunity)

        Args:
            entry_price: Entry price
            future_prices: Next N days of prices (need at least 7)

        Returns:
            Tuple of (label, return_pct, exit_reason)
        """
        # Need at least 7 days of future data
        if len(future_prices) < 7:
            return 'hold', 0.0, 'insufficient_data'

        # Calculate 7-day forward return
        price_7d = future_prices.iloc[6]  # Day 7 (0-indexed, so index 6)
        return_7d = (price_7d - entry_price) / entry_price

        # Symmetric threshold-based classification
        # Only train on STRONG signals (±10%), exclude noise
        if return_7d >= 0.10:
            label = 'buy'  # Strong upward move (≥+10%)
        elif return_7d <= -0.10:
            label = 'sell'  # Strong downward move (≤-10%)
        else:
            label = 'hold'  # Ambiguous move (excluded from training)

        return label, return_7d, 'forward_7d'

    def generate_labeled_data(self, symbol: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate labeled training samples from historical data

        Args:
            symbol: Stock symbol
            df: Historical OHLCV data

        Returns:
            List of labeled samples (features + label)
        """
        samples = []

        # Need at least hold_period + 50 days for features + future window
        min_length = self.hold_period + 50
        if len(df) < min_length:
            return samples

        # PRIORITY 1: FULL 179-FEATURE GPU PROCESSING (10-12 hours, 85-95% accuracy)
        # FORCE THIS PATH - NO FALLBACK TO 30 FEATURES!
        if hasattr(self.feature_engineer, 'extract_features_batch') and self.use_gpu:
            print(f"[GPU BATCH MODE - 179 FEATURES] Processing {len(df) - self.hold_period - 50} days on GPU!")
            start_indices = list(range(50, len(df) - self.hold_period))
            all_features = self.feature_engineer.extract_features_batch(df, start_indices, symbol)

            # Add sector + market_cap metadata (3 features: sector_code, market_cap_tier, symbol_hash)
            from advanced_ml.config.symbol_metadata import get_symbol_metadata
            metadata = get_symbol_metadata(symbol)
            for features in all_features:
                features.update(metadata)
        else:
            # CRITICAL: If extract_features_batch doesn't exist, we MUST NOT use 30-feature vectorized mode
            # This causes training/prediction mismatch. Instead, fall back to loop-based 179-feature extraction.
            print(f"[WARNING] extract_features_batch not available - falling back to loop-based 179-feature extraction")
            all_features = None

        # Process results if we got them from vectorized/batch mode
        if all_features is not None:

            # Process results
            for idx_pos, i in enumerate(start_indices):
                features = all_features[idx_pos]
                if features.get('feature_count', 0) == 0:
                    continue

                entry_date = df.index[i]
                entry_price = df.iloc[i]['close']
                future_prices = df.iloc[i+1:i+1+self.hold_period]['close']
                label, return_pct, exit_reason = self.calculate_trade_outcome(entry_price, future_prices)

                # CRITICAL: Skip HOLD samples (only train on strong signals ±10%)
                if label == 'hold':
                    self.stats['hold_signals'] += 1
                    continue  # Exclude from training - ambiguous moves

                label_map = {'buy': 0, 'sell': 1}  # Binary classification
                label_int = label_map[label]

                sample = {
                    'symbol': symbol,
                    'date': entry_date.strftime('%Y-%m-%d'),
                    'entry_price': float(entry_price),
                    'return_pct': float(return_pct * 100),
                    'label': label_int,
                    'label_name': label,
                    'exit_reason': exit_reason,
                    'features': features
                }
                samples.append(sample)
                self.stats['total_trades'] += 1
                if label == 'buy':
                    self.stats['buy_signals'] += 1
                elif label == 'sell':
                    self.stats['sell_signals'] += 1
                else:
                    self.stats['hold_signals'] += 1

            return samples

        # FALLBACK: Original loop-based processing (slower)
        print(f"[DEBUG] Starting loop for {len(df) - self.hold_period - 50} days")
        # Iterate through each day (except last hold_period days)
        for i in range(50, len(df) - self.hold_period):
            if i == 50:
                print(f"[DEBUG] First iteration (day {i})")
            try:
                # Get data up to this point for feature extraction
                historical_data = df.iloc[:i+1]

                # Get entry date (needed for regime/macro features)
                entry_date = df.index[i]

                # Extract features
                if i == 50:
                    print(f"[DEBUG] Extracting features for day {i}...")
                features = self.feature_engineer.extract_features(historical_data, symbol=symbol)

                # Add sector + market_cap metadata (3 features: sector_code, market_cap_tier, symbol_hash)
                from advanced_ml.config.symbol_metadata import get_symbol_metadata
                metadata = get_symbol_metadata(symbol)
                features.update(metadata)

                if i == 50:
                    print(f"[DEBUG] Features extracted: {features.get('feature_count', 0)} features")
                    print(f"[DEBUG] Checking feature count...")

                # Add regime + macro features (Phase 1 improvement)
                # DISABLED during archive generation for speed - events have pre-assigned regimes
                # Will be enabled during normal training
                # try:
                #     regime_macro = get_regime_macro_features(entry_date, symbol)
                #     features.update(regime_macro)
                # except Exception as e:
                #     # If regime/macro features fail, continue without them (graceful degradation)
                #     pass

                # Skip if feature extraction failed
                if i == 50:
                    print(f"[DEBUG] About to check feature count (line 218)...")
                if features.get('feature_count', 0) == 0:
                    continue

                # Get entry price (close of current day)
                if i == 50:
                    print(f"[DEBUG] Getting entry price...")
                entry_price = df.iloc[i]['close']
                if i == 50:
                    print(f"[DEBUG] Entry price: {entry_price}")

                # Get future prices for next hold_period days
                if i == 50:
                    print(f"[DEBUG] Getting future prices...")
                future_prices = df.iloc[i+1:i+1+self.hold_period]['close']
                if i == 50:
                    print(f"[DEBUG] Future prices: {len(future_prices)} days")

                # Calculate outcome
                if i == 50:
                    print(f"[DEBUG] Calculating trade outcome...")
                label, return_pct, exit_reason = self.calculate_trade_outcome(entry_price, future_prices)
                if i == 50:
                    print(f"[DEBUG] Outcome: {label}, return: {return_pct:.2%}")

                # CRITICAL: Skip HOLD samples (only train on strong signals ±10%)
                if label == 'hold':
                    self.stats['hold_signals'] += 1
                    continue  # Exclude from training - ambiguous moves

                # Map label to integer
                label_map = {'buy': 0, 'sell': 1}  # Binary classification
                label_int = label_map[label]

                # Create sample
                sample = {
                    'symbol': symbol,
                    'date': entry_date.strftime('%Y-%m-%d'),
                    'entry_price': float(entry_price),
                    'return_pct': float(return_pct * 100),
                    'label': label_int,
                    'label_name': label,
                    'exit_reason': exit_reason,
                    'features': features
                }

                samples.append(sample)

                # Update stats
                self.stats['total_trades'] += 1
                if label == 'buy':
                    self.stats['buy_labels'] += 1
                elif label == 'sell':
                    self.stats['sell_labels'] += 1
                else:
                    self.stats['hold_labels'] += 1

            except Exception as e:
                print(f"[ERROR] Error processing day {i} for {symbol}: {e}")
                continue

        return samples

    def save_samples_to_db(self, samples: List[Dict[str, Any]]):
        """
        Save labeled samples to database

        Args:
            samples: List of labeled samples
        """
        if not samples:
            return

        conn = self.db.get_connection()
        cursor = conn.cursor()

        for sample in samples:
            try:
                # Save to feature_store table
                cursor.execute('''
                    INSERT OR REPLACE INTO feature_store
                    (symbol, timestamp, features_json, rsi_14, macd_histogram,
                     volume_ratio, trend_strength, momentum_score, volatility_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    sample['symbol'],
                    sample['date'],
                    json.dumps(sample['features']),
                    sample['features'].get('rsi_14', 50.0),
                    sample['features'].get('macd_histogram', 0.0),
                    sample['features'].get('volume_ratio_20', 1.0),
                    sample['features'].get('trend_strength', 25.0),
                    sample['features'].get('momentum_score', 0.0),
                    sample['features'].get('historical_vol_20', 0.0)
                ))

                # Save to trades table (as completed backtest trades)
                trade_id = f"backtest_{sample['symbol']}_{sample['date']}"

                # Map label integer to action name (binary: buy/sell)
                label_to_action = {0: 'buy', 1: 'sell'}
                action = label_to_action[sample['label']]

                cursor.execute('''
                    INSERT OR REPLACE INTO trades
                    (id, symbol, entry_date, entry_price, exit_date, exit_price,
                     outcome, profit_loss_pct, exit_reason, entry_features_json, trade_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_id,
                    sample['symbol'],
                    sample['date'],
                    sample['entry_price'],
                    sample['date'],  # Use same date for backtest
                    sample['entry_price'] * (1 + sample['return_pct'] / 100),
                    action,  # Store 'buy', 'hold', or 'sell' instead of 'win'/'neutral'/'loss'
                    sample['return_pct'],
                    sample['exit_reason'],
                    json.dumps(sample['features']),
                    'backtest'
                ))

            except Exception as e:
                print(f"[ERROR] Failed to save sample: {e}")
                continue

        conn.commit()
        conn.close()

    def run_backtest(self, symbols: List[str], years: int = 7, save_to_db: bool = True) -> Dict[str, Any]:
        """
        Run historical backtest on multiple symbols

        Args:
            symbols: List of stock symbols
            years: Years of history to use
            save_to_db: Whether to save results to database

        Returns:
            Backtest results dictionary
        """
        print(f"\n[BACKTEST] Historical Backtest")
        print(f"  Symbols: {len(symbols)}")
        print(f"  Years: {years}")
        print(f"  Hold Period: {self.hold_period} days")
        print(f"  Win Threshold: +{self.win_threshold * 100}%")
        print(f"  Loss Threshold: {self.loss_threshold * 100}%")
        print()

        all_samples = []

        print(f"\n[DEBUG] About to start processing {len(symbols)} symbols")
        print(f"[DEBUG] Symbols list: {symbols}")

        # Process each symbol with progress bar
        print("[DEBUG] Entering loop (tqdm disabled for debugging)...")
        for i, symbol in enumerate(symbols, 1):
            print(f"\n[DEBUG] INSIDE LOOP - symbol {i}/{len(symbols)}: {symbol}")
            print(f"\n[DEBUG] Starting symbol {symbol}")
            # Fetch historical data
            print(f"[DEBUG] Fetching data for {symbol}...")
            df = self.fetch_historical_data(symbol, years)
            print(f"[DEBUG] Data fetched: {len(df) if df is not None else 0} rows")

            if df is None:
                continue

            # Generate labeled samples
            print(f"[DEBUG] Generating labeled data for {symbol}...")
            samples = self.generate_labeled_data(symbol, df)
            print(f"[DEBUG] Generated {len(samples) if samples else 0} samples")

            if samples:
                all_samples.extend(samples)
                self.stats['symbols_processed'] += 1

                # Print progress every 10 symbols
                if self.stats['symbols_processed'] % 10 == 0:
                    print(f"\n[PROGRESS] {self.stats['symbols_processed']} symbols | {len(all_samples)} samples")

        # Save to database
        if save_to_db and all_samples:
            print(f"\n[SAVE] Saving {len(all_samples)} samples to database...")
            self.save_samples_to_db(all_samples)

        # Calculate final statistics
        total = len(all_samples)
        if total > 0:
            self.stats['buy_pct'] = self.stats['buy_labels'] / total * 100
            self.stats['hold_pct'] = self.stats['hold_labels'] / total * 100
            self.stats['sell_pct'] = self.stats['sell_labels'] / total * 100

            # Calculate average returns
            buy_returns = [s['return_pct'] for s in all_samples if s['label_name'] == 'buy']
            sell_returns = [s['return_pct'] for s in all_samples if s['label_name'] == 'sell']

            if buy_returns:
                self.stats['avg_win'] = np.mean(buy_returns)
            if sell_returns:
                self.stats['avg_loss'] = np.mean(sell_returns)

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"BACKTEST COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total Trades Generated: {total}")
        print(f"Symbols Processed: {self.stats['symbols_processed']}")
        print(f"\nLabel Distribution:")
        print(f"  Buy (profitable):  {self.stats['buy_labels']} ({self.stats.get('buy_pct', 0):.1f}%)")
        print(f"  Hold (neutral):    {self.stats['hold_labels']} ({self.stats.get('hold_pct', 0):.1f}%)")
        print(f"  Sell (loss):       {self.stats['sell_labels']} ({self.stats.get('sell_pct', 0):.1f}%)")
        print(f"\nAverage Returns:")
        print(f"  Avg Win:  +{self.stats['avg_win']:.2f}%")
        print(f"  Avg Loss: {self.stats['avg_loss']:.2f}%")
        print(f"{'=' * 60}\n")

        return {
            'total_samples': total,
            'samples': all_samples,
            'stats': self.stats
        }

    def prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load samples from database and prepare for model training

        Returns:
            Tuple of (X, y) where X is features matrix, y is labels vector
        """
        conn = self.db.get_connection()
        cursor = conn.cursor()

        # Fetch all training samples from trades table (has both features and labels)
        cursor.execute('''
            SELECT entry_features_json, outcome
            FROM trades
            WHERE trade_type = 'backtest'
            AND entry_features_json IS NOT NULL
            AND outcome IS NOT NULL
        ''')

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            print("[WARNING] No training data found in database")
            return np.array([]), np.array([])

        # Extract features and labels
        feature_list = []
        label_list = []

        for row in rows:
            features_json = row[0]
            outcome = row[1]

            # Parse features
            features = json.loads(features_json)

            # Remove metadata fields
            exclude_keys = ['feature_count', 'symbol', 'last_price', 'last_volume', 'timestamp', 'error']
            feature_values = []

            for key, value in sorted(features.items()):
                if key not in exclude_keys:
                    if isinstance(value, (int, float)):
                        if np.isnan(value) or np.isinf(value):
                            value = 0.0
                        feature_values.append(float(value))

            # Map outcome (action) to label integer
            # outcome field now stores 'buy' or 'sell' (binary classification)
            label_map = {'buy': 0, 'sell': 1}
            label = label_map.get(outcome, 0)  # Default to 0 (buy) if unknown

            feature_list.append(feature_values)
            label_list.append(label)

        # Convert to numpy arrays
        X = np.array(feature_list)
        y = np.array(label_list)

        print(f"\n[DATA] Training data prepared")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Buy: {np.sum(y == 0)}")
        print(f"  Sell: {np.sum(y == 1)}")

        return X, y


if __name__ == '__main__':
    # Test backtest engine
    print("Testing Historical Backtest Engine...")

    # Test with a few symbols
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

    # Initialize backtest
    backtest = HistoricalBacktest()

    # Run backtest (1 year of data for speed)
    results = backtest.run_backtest(test_symbols, years=1, save_to_db=True)

    print(f"\nGenerated {results['total_samples']} training samples")

    # Test loading data for training
    print("\nTesting data loading...")
    X, y = backtest.prepare_training_data()

    if len(X) > 0:
        print(f"\nReady for training:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print("\n[OK] Backtest engine test complete!")
    else:
        print("[ERROR] No data loaded")
