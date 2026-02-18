
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
TurboMode Backtest Engine
Clean backtest implementation with NO AdvancedML dependencies
Generates training samples from historical data using canonical label logic

Author: TurboMode Core Engine
Date: 2026-01-06
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import uuid

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
turbomode_dir = os.path.dirname(current_dir)
backend_dir = os.path.dirname(turbomode_dir)
project_root = os.path.dirname(backend_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Master Market Data API (read-only)
from master_market_data.market_data_api import get_market_data_api

# Import feature engine for computing entry features
from backend.turbomode.core_engine.turbomode_vectorized_feature_engine import TurboModeVectorizedFeatureEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('turbomode_backtest')


class TurboModeBacktest:
    """
    TurboMode-only backtest engine

    NO dependencies on:
    - backend.advanced_ml.*
    - HistoricalBacktest
    - AdvancedMLDatabase

    Only uses TurboMode schema tables:
    - price_data
    - feature_store
    - trades
    - training_runs
    - model_metadata
    """

    def __init__(self, turbomode_db_path: str = "backend/data/turbomode.db"):
        """
        Initialize TurboMode backtest engine

        Args:
            turbomode_db_path: Path to turbomode.db
        """
        # Convert to absolute path
        if not os.path.isabs(turbomode_db_path):
            turbomode_db_path = os.path.join(project_root, turbomode_db_path)

        self.turbomode_db_path = turbomode_db_path

        # SCHEMA GUARDRAIL: Validate schema before any operations
        from backend.turbomode.schema_guardrail import validate_schema
        try:
            validation_result = validate_schema(turbomode_db_path, strict=True)
            if validation_result['status'] == 'OK':
                logger.info("[GUARDRAIL] Schema validation passed")
        except Exception as e:
            logger.error(f"[GUARDRAIL] Schema validation failed: {e}")
            raise

        # Connect to Master Market Data DB (read-only)
        self.market_data_api = get_market_data_api()

        # Initialize feature engine for computing entry features
        self.feature_engine = TurboModeVectorizedFeatureEngine()

        logger.info("[INIT] TurboMode Backtest Engine initialized")
        logger.info(f"       TurboMode DB: {turbomode_db_path}")
        logger.info("       Feature engine initialized (179 features)")

    def get_last_entry_date(self, symbol: str) -> Optional[str]:
        """
        Get the last entry_date for a symbol in the database (for incremental backtest)

        Args:
            symbol: Stock ticker

        Returns:
            Last entry_date as 'YYYY-MM-DD' string, or None if no data exists
        """
        conn = sqlite3.connect(self.turbomode_db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT MAX(entry_date) FROM trades
            WHERE trade_type = 'backtest'
            AND symbol = ?
        """, (symbol,))

        result = cursor.fetchone()[0]
        conn.close()

        return result  # Returns 'YYYY-MM-DD' or None

    def generate_backtest_samples(self,
                                  symbol: str,
                                  start_date: str = None,
                                  end_date: str = None,
                                  lookback_days: int = 3650,
                                  incremental: bool = False) -> Dict[str, Any]:
        """
        Generate backtest training samples for a symbol

        Args:
            symbol: Stock ticker (canonical format)
            start_date: Start date (YYYY-MM-DD) - optional
            end_date: End date (YYYY-MM-DD) - optional
            lookback_days: Days to look back (default 3650 = ~10 years)
            incremental: If True, only process dates after last entry_date (default: False)

        Returns:
            Dictionary with results
        """
        logger.info(f"[BACKTEST] Generating samples for {symbol} (incremental={incremental})")

        # Calculate date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None:
            if incremental:
                # Get last processed date from database
                last_date = self.get_last_entry_date(symbol)
                if last_date:
                    # Start from day after last entry
                    last_dt = datetime.strptime(last_date, '%Y-%m-%d')
                    start_dt = last_dt + timedelta(days=1)
                    start_date = start_dt.strftime('%Y-%m-%d')
                    logger.info(f"[INCREMENTAL] Last entry: {last_date}, starting from {start_date}")
                else:
                    # No existing data, fall back to full lookback
                    logger.info(f"[INCREMENTAL] No existing data, using full lookback")
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    start_dt = end_dt - timedelta(days=lookback_days)
                    start_date = start_dt.strftime('%Y-%m-%d')
            else:
                # Standard full lookback
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                start_dt = end_dt - timedelta(days=lookback_days)
                start_date = start_dt.strftime('%Y-%m-%d')

        logger.info(f"[BACKTEST] Date range: {start_date} to {end_date}")

        # Fetch historical price data from Master Market Data DB
        price_data = self.market_data_api.get_candles(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe='1d'
        )

        if price_data is None or len(price_data) == 0:
            logger.warning(f"[BACKTEST] No price data found for {symbol}")
            return {
                'symbol': symbol,
                'total_samples': 0,
                'buy_samples': 0,
                'sell_samples': 0,
                'hold_samples': 0
            }

        # Convert timestamp index to 'date' column for backtest processing
        price_data = price_data.reset_index()
        price_data.rename(columns={'timestamp': 'date'}, inplace=True)
        if incremental:
            last_date = self.get_last_entry_date(symbol)
            if last_date:
                price_data = price_data[price_data['date'] > last_date].reset_index(drop=True)
                logger.info(f"[INCREMENTAL] Filtered to {len(price_data)} new days after {last_date}")
                if len(price_data) == 0:
                    logger.info("[INCREMENTAL] No new days to process")
                    return {
                        'symbol': symbol,
                        'total_samples': 0,
                        'buy_samples': 0,
                        'sell_samples': 0,
                        'hold_samples': 0
                    }

        logger.info(f"[BACKTEST] Loaded {len(price_data)} days of price data")

        # Generate training samples with canonical labels AND features
        samples = self._generate_samples_with_canonical_labels(symbol, price_data)

        # Save samples to trades table
        saved_count = self._save_samples_to_trades_table(samples)

        # Calculate statistics
        buy_count = sum(1 for s in samples if s['outcome'] == 'buy')
        sell_count = sum(1 for s in samples if s['outcome'] == 'sell')
        hold_count = sum(1 for s in samples if s['outcome'] == 'hold')

        result = {
            'symbol': symbol,
            'total_samples': len(samples),
            'saved_samples': saved_count,
            'buy_samples': buy_count,
            'sell_samples': sell_count,
            'hold_samples': hold_count,
            'buy_pct': (buy_count / len(samples) * 100) if len(samples) > 0 else 0,
            'sell_pct': (sell_count / len(samples) * 100) if len(samples) > 0 else 0,
            'hold_pct': (hold_count / len(samples) * 100) if len(samples) > 0 else 0
        }

        logger.info(f"[BACKTEST] {symbol}: {len(samples)} samples generated")
        logger.info(f"           BUY: {buy_count} ({result['buy_pct']:.1f}%)")
        logger.info(f"           SELL: {sell_count} ({result['sell_pct']:.1f}%)")
        logger.info(f"           HOLD: {hold_count} ({result['hold_pct']:.1f}%)")

        return result

    def _generate_samples_with_canonical_labels(self,
                                                 symbol: str,
                                                 price_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate training samples with canonical label logic AND entry features

        Canonical label logic:
        - Look forward 14 days (holding period)
        - Calculate return_pct = (future_price - entry_price) / entry_price
        - if return_pct >= +6%: label = 'buy'
        - if return_pct <= -6%: label = 'sell'
        - else: label = 'hold'

        Features:
        - Computes 179 features for each entry point using TurboModeVectorizedFeatureEngine
        - Stores features as JSON in entry_features_json column

        Args:
            symbol: Stock ticker
            price_data: DataFrame with columns [date, open, high, low, close, volume]

        Returns:
            List of sample dictionaries with features
        """
        import json

        samples = []
        holding_period = 14  # days (aligned with 14-day swing trading strategy)
        buy_threshold = 0.06  # +6%
        sell_threshold = -0.06  # -6%

        # Ensure price_data is sorted by date
        price_data = price_data.sort_values('date').reset_index(drop=True)

        # Compute features for ALL dates at once (vectorized)
        # Feature engine expects DataFrame with columns: open, high, low, close, volume
        try:
            features_df = self.feature_engine.extract_features(price_data)
            # Convert DataFrame to list of dicts (one per row)
            all_features = features_df.to_dict('records')
        except Exception as e:
            logger.error(f"[FEATURES] Failed to compute features for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            all_features = None

        # === VECTORIZED LABEL + MFE/MAE PRECOMPUTATION (Phase 2 Restore) ===
        n = len(price_data)
        hp = holding_period

        closes = price_data['close'].values
        highs = price_data['high'].values
        lows = price_data['low'].values

        # Forward exit prices (vectorized)
        exit_prices = closes[hp:]
        entry_prices = closes[:-hp]

        # Forward returns
        return_pct_arr = (exit_prices - entry_prices) / entry_prices

        # Precompute MFE/MAE windows
        mfe_arr = np.zeros(n - hp)
        mae_arr = np.zeros(n - hp)

        for i in range(n - hp):
            window_high = highs[i+1:i+hp+1]
            window_low = lows[i+1:i+hp+1]
            mfe_arr[i] = (window_high.max() - entry_prices[i]) / entry_prices[i]
            mae_arr[i] = (window_low.min() - entry_prices[i]) / entry_prices[i]

        # Threshold hits
        hit_target_arr = mfe_arr >= buy_threshold
        hit_stop_arr = mae_arr <= sell_threshold

        # Vectorized outcome initialization
        outcome_arr = np.full(n - hp, 'hold', dtype=object)

        # BUY only
        buy_mask = hit_target_arr & ~hit_stop_arr
        outcome_arr[buy_mask] = 'buy'

        # SELL only
        sell_mask = hit_stop_arr & ~hit_target_arr
        outcome_arr[sell_mask] = 'sell'

        # BOTH hit → determine first hit
        both_mask = hit_target_arr & hit_stop_arr

        first_target_day_arr = np.full(n - hp, None, dtype=object)
        first_stop_day_arr = np.full(n - hp, None, dtype=object)

        both_idx = np.where(both_mask)[0]

        for idx in both_idx:
            entry_price = entry_prices[idx]
            for j in range(1, hp + 1):
                high_pct = (highs[idx + j] - entry_price) / entry_price
                low_pct = (lows[idx + j] - entry_price) / entry_price

                if first_target_day_arr[idx] is None and high_pct >= buy_threshold:
                    first_target_day_arr[idx] = j

                if first_stop_day_arr[idx] is None and low_pct <= sell_threshold:
                    first_stop_day_arr[idx] = j

                if first_target_day_arr[idx] is not None and first_stop_day_arr[idx] is not None:
                    break

            # FIRST-HIT DECISION
            if first_stop_day_arr[idx] is None or (first_target_day_arr[idx] is not None and first_target_day_arr[idx] < first_stop_day_arr[idx]):
                outcome_arr[idx] = 'buy'
            else:
                outcome_arr[idx] = 'sell'

        # === SAMPLE CONSTRUCTION (kept identical to your logic) ===
        samples = []
        for i in range(n - hp):
            entry_row = price_data.iloc[i]
            exit_row = price_data.iloc[i + hp]

            entry_date = entry_row['date']
            exit_date = exit_row['date']

            entry_price = float(entry_row['close'])
            exit_price = float(exit_row['close'])

            outcome = outcome_arr[i]
            mfe = float(mfe_arr[i])
            mae = float(mae_arr[i])

            target_day = first_target_day_arr[i] if hit_target_arr[i] else None
            stop_day = first_stop_day_arr[i] if hit_stop_arr[i] else None

            features_dict = all_features[i] if all_features is not None else None
            entry_features_json = json.dumps(features_dict) if features_dict else None

            sample = {
                'id': str(uuid.uuid4()),
                'symbol': symbol,
                'entry_date': entry_date.strftime('%Y-%m-%d') if hasattr(entry_date, 'strftime') else str(entry_date),
                'entry_price': entry_price,
                'exit_date': exit_date.strftime('%Y-%m-%d') if hasattr(exit_date, 'strftime') else str(exit_date),
                'exit_price': exit_price,
                'position_size': 1.0,
                'outcome': outcome,
                'profit_loss': exit_price - entry_price,
                'profit_loss_pct': float(return_pct_arr[i]),
                'exit_reason': 'backtest',
                'entry_features_json': entry_features_json,
                'trade_type': 'backtest',
                'strategy': 'turbomode',
                'notes': f'{holding_period}d swing trade',
                'created_at': datetime.now().isoformat(),
                'mfe': mfe,
                'mae': mae,
                'target_hit_day': target_day,
                'stop_hit_day': stop_day
            }

            samples.append(sample)

        return samples

    def _save_samples_to_trades_table(self, samples: List[Dict[str, Any]]) -> int:
        """
        Save training samples to trades table in turbomode.db

        Args:
            samples: List of sample dictionaries

        Returns:
            Number of samples saved
        """
        if not samples:
            return 0

        conn = sqlite3.connect(self.turbomode_db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")

        params = []
        for s in samples:
            params.append((
                s['id'], s['symbol'], s['entry_date'], s['entry_price'],
                s['exit_date'], s['exit_price'], s['position_size'], s['outcome'],
                s['profit_loss'], s['profit_loss_pct'], s['exit_reason'],
                s['entry_features_json'], s['trade_type'], s['strategy'],
                s['notes'], s['created_at']
            ))

        try:
            cursor.executemany("""
                INSERT OR REPLACE INTO trades (
                    id, symbol, entry_date, entry_price, exit_date, exit_price,
                    position_size, outcome, profit_loss, profit_loss_pct,
                    exit_reason, entry_features_json, trade_type, strategy,
                    notes, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, params)
            saved_count = len(params)
        except Exception as e:
            logger.error(f"[ERROR] Batch insert failed: {e}")
            saved_count = 0

        conn.commit()
        conn.close()

        return saved_count

    def get_existing_sample_count(self, symbol: str = None) -> int:
        """
        Get count of existing backtest samples

        Args:
            symbol: Optional symbol filter

        Returns:
            Count of samples
        """
        conn = sqlite3.connect(self.turbomode_db_path)
        cursor = conn.cursor()

        if symbol:
            cursor.execute("""
                SELECT COUNT(*) FROM trades
                WHERE trade_type = 'backtest' AND symbol = ?
            """, (symbol,))
        else:
            cursor.execute("""
                SELECT COUNT(*) FROM trades
                WHERE trade_type = 'backtest'
            """)

        count = cursor.fetchone()[0]
        conn.close()

        return count


def main():
    """
    Production entrypoint for full 230-symbol backtest generation.

    This delegates to generate_backtest_data.py which processes all 230 CORE_230
    symbols and writes backtest samples to backend/data/turbomode.db.

    For testing a single symbol, use the __main__ block below.
    """
    print("[PRODUCTION] Running full 230-symbol batch backtest generation")
    print("This function delegates to generate_backtest_data.py")
    print()

    import subprocess
    import os

    # Get path to generate_backtest_data.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    batch_script = os.path.join(current_dir, "generate_backtest_data.py")

    if not os.path.exists(batch_script):
        raise FileNotFoundError(f"Batch backtest generator not found: {batch_script}")

    # Run the batch generator
    result = subprocess.run([sys.executable, batch_script], cwd=current_dir)

    if result.returncode != 0:
        raise RuntimeError(f"Batch backtest generation failed with exit code {result.returncode}")

    print()
    print("[OK] Full backtest generation completed")
    return 0


# ===================================================================================
# TEST HARNESS (for development/testing only)
# ===================================================================================
# NOTE: This __main__ block is for TESTING ONLY with a single symbol.
# Production code should call main() or use generate_backtest_data.py directly.
# ===================================================================================

if __name__ == '__main__':
    # Test the backtest engine
    print("Testing TurboMode Backtest Engine")
    print("=" * 80)

    backtest = TurboModeBacktest()

    # Test with a single symbol
    test_symbol = 'AAPL'
    result = backtest.generate_backtest_samples(
        symbol=test_symbol,
        lookback_days=365  # 1 year for testing
    )

    print("\nTest Results:")
    print(f"  Symbol: {result['symbol']}")
    print(f"  Total samples: {result['total_samples']}")
    print(f"  BUY: {result['buy_samples']} ({result['buy_pct']:.1f}%)")
    print(f"  SELL: {result['sell_samples']} ({result['sell_pct']:.1f}%)")
    print(f"  HOLD: {result['hold_samples']} ({result['hold_pct']:.1f}%)")
    print("\n[OK] TurboMode Backtest Engine test complete!")
