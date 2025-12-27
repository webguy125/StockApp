"""
Historical Backtesting Engine
Generates training data from historical price movements

Features:
- Fetches 2+ years of historical data
- Generates 300+ features per day
- Simulates 14-day hold period trades
- Labels outcomes: Buy (profitable), Hold (neutral), Sell (unprofitable)
- Stores results in database
- Can generate 10,000+ labeled samples instantly
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
from advanced_ml.features.regime_macro_features import get_regime_macro_features
from advanced_ml.database.schema import AdvancedMLDatabase


class HistoricalBacktest:
    """
    Generate training data from historical price movements

    Strategy:
    1. Fetch 2+ years of historical data for symbols
    2. For each trading day:
       - Extract 300+ features
       - Simulate entering a long position
       - Check outcome after 14 days:
         * +10% or more = Buy label (0)
         * -5% or worse = Sell label (2)
         * Between -5% and +10% = Hold label (1)
    3. Store labeled samples for model training
    """

    def __init__(self, db_path: str = "backend/data/advanced_ml_system.db"):
        """
        Initialize backtesting engine

        Args:
            db_path: Path to database
        """
        self.db = AdvancedMLDatabase(db_path)
        self.feature_engineer = FeatureEngineer()

        # Trading parameters (match live system)
        self.hold_period = 14  # days
        self.win_threshold = 0.10  # +10%
        self.loss_threshold = -0.05  # -5%

        # Backtest statistics
        self.stats = {
            'total_trades': 0,
            'buy_labels': 0,
            'hold_labels': 0,
            'sell_labels': 0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'symbols_processed': 0
        }

    def fetch_historical_data(self, symbol: str, years: int = 2, start_date: datetime = None, end_date: datetime = None) -> Optional[pd.DataFrame]:
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
        Calculate trade outcome for 14-day hold period

        Args:
            entry_price: Entry price
            future_prices: Next 14 days of prices

        Returns:
            Tuple of (label, return_pct, exit_reason)
        """
        if future_prices.empty:
            return 'hold', 0.0, 'insufficient_data'

        # Check each day for exit conditions
        for i, price in enumerate(future_prices):
            return_pct = (price - entry_price) / entry_price

            # Win condition: +10% or more
            if return_pct >= self.win_threshold:
                return 'buy', return_pct, f'target_hit_day_{i+1}'

            # Loss condition: -5% or worse
            if return_pct <= self.loss_threshold:
                return 'sell', return_pct, f'stop_hit_day_{i+1}'

        # If no exit condition hit, use final price
        final_return = (future_prices.iloc[-1] - entry_price) / entry_price

        # Label based on final return
        if final_return >= self.win_threshold:
            label = 'buy'
        elif final_return <= self.loss_threshold:
            label = 'sell'
        else:
            label = 'hold'

        return label, final_return, f'time_expired_{len(future_prices)}d'

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

        # Iterate through each day (except last hold_period days)
        for i in range(50, len(df) - self.hold_period):
            try:
                # Get data up to this point for feature extraction
                historical_data = df.iloc[:i+1]

                # Get entry date (needed for regime/macro features)
                entry_date = df.index[i]

                # Extract features
                features = self.feature_engineer.extract_features(historical_data, symbol=symbol)

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
                if features.get('feature_count', 0) == 0:
                    continue

                # Get entry price (close of current day)
                entry_price = df.iloc[i]['close']

                # Get future prices for next hold_period days
                future_prices = df.iloc[i+1:i+1+self.hold_period]['close']

                # Calculate outcome
                label, return_pct, exit_reason = self.calculate_trade_outcome(entry_price, future_prices)

                # Map label to integer
                label_map = {'buy': 0, 'hold': 1, 'sell': 2}
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
                    'win' if sample['label'] == 0 else 'loss' if sample['label'] == 2 else 'neutral',
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

    def run_backtest(self, symbols: List[str], years: int = 2, save_to_db: bool = True) -> Dict[str, Any]:
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

        # Process each symbol with progress bar
        for symbol in tqdm(symbols, desc="Processing symbols"):
            # Fetch historical data
            df = self.fetch_historical_data(symbol, years)

            if df is None:
                continue

            # Generate labeled samples
            samples = self.generate_labeled_data(symbol, df)

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

        # Fetch all backtest trades with features
        cursor.execute('''
            SELECT entry_features_json, outcome
            FROM trades
            WHERE trade_type = 'backtest'
            AND entry_features_json IS NOT NULL
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

            # Map outcome to label
            label_map = {'win': 0, 'neutral': 1, 'loss': 2}
            label = label_map.get(outcome, 1)

            feature_list.append(feature_values)
            label_list.append(label)

        # Convert to numpy arrays
        X = np.array(feature_list)
        y = np.array(label_list)

        print(f"\n[DATA] Training data prepared")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Buy: {np.sum(y == 0)}")
        print(f"  Hold: {np.sum(y == 1)}")
        print(f"  Sell: {np.sum(y == 2)}")

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
