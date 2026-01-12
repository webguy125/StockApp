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
backend_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(backend_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Master Market Data API (read-only)
from master_market_data.market_data_api import get_market_data_api

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
        from turbomode.schema_guardrail import validate_schema
        try:
            validation_result = validate_schema(turbomode_db_path, strict=True)
            if validation_result['status'] == 'OK':
                logger.info("[GUARDRAIL] Schema validation passed")
        except Exception as e:
            logger.error(f"[GUARDRAIL] Schema validation failed: {e}")
            raise

        # Connect to Master Market Data DB (read-only)
        self.market_data_api = get_market_data_api()
        logger.info("[INIT] TurboMode Backtest Engine initialized")
        logger.info(f"       TurboMode DB: {turbomode_db_path}")

    def generate_backtest_samples(self,
                                  symbol: str,
                                  start_date: str = None,
                                  end_date: str = None,
                                  lookback_days: int = 3650) -> Dict[str, Any]:
        """
        Generate backtest training samples for a symbol

        Args:
            symbol: Stock ticker (canonical format)
            start_date: Start date (YYYY-MM-DD) - optional
            end_date: End date (YYYY-MM-DD) - optional
            lookback_days: Days to look back (default 3650 = ~10 years)

        Returns:
            Dictionary with results
        """
        logger.info(f"[BACKTEST] Generating samples for {symbol}")

        # Calculate date range
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if start_date is None:
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

        logger.info(f"[BACKTEST] Loaded {len(price_data)} days of price data")

        # Generate training samples with canonical labels
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
        Generate training samples with canonical label logic

        Canonical label logic:
        - Look forward 5 days (holding period)
        - Calculate return_pct = (future_price - entry_price) / entry_price
        - if return_pct >= +5%: label = 'buy'
        - if return_pct <= -5%: label = 'sell'
        - else: label = 'hold'

        Args:
            symbol: Stock ticker
            price_data: DataFrame with columns [date, open, high, low, close, volume]

        Returns:
            List of sample dictionaries
        """
        samples = []
        holding_period = 5  # days
        buy_threshold = 0.05  # +5%
        sell_threshold = -0.05  # -5%

        # Ensure price_data is sorted by date
        price_data = price_data.sort_values('date').reset_index(drop=True)

        # Generate samples for each day (except last holding_period days)
        for i in range(len(price_data) - holding_period):
            entry_row = price_data.iloc[i]
            entry_date = entry_row['date']
            entry_price = entry_row['close']

            # Look forward holding_period days
            exit_row = price_data.iloc[i + holding_period]
            exit_date = exit_row['date']
            exit_price = exit_row['close']

            # Calculate return percentage
            return_pct = (exit_price - entry_price) / entry_price
            profit_loss = exit_price - entry_price

            # Apply canonical label logic
            if return_pct >= buy_threshold:
                outcome = 'buy'
            elif return_pct <= sell_threshold:
                outcome = 'sell'
            else:
                outcome = 'hold'

            # Create sample
            # Convert pandas Timestamp to string format
            entry_date_str = entry_date.strftime('%Y-%m-%d') if hasattr(entry_date, 'strftime') else str(entry_date)
            exit_date_str = exit_date.strftime('%Y-%m-%d') if hasattr(exit_date, 'strftime') else str(exit_date)

            sample = {
                'id': str(uuid.uuid4()),
                'symbol': symbol,
                'entry_date': entry_date_str,
                'entry_price': float(entry_price),
                'exit_date': exit_date_str,
                'exit_price': float(exit_price),
                'position_size': 1.0,
                'outcome': outcome,
                'profit_loss': float(profit_loss),
                'profit_loss_pct': float(return_pct),
                'exit_reason': 'backtest',
                'entry_features_json': None,  # Features can be added later
                'trade_type': 'backtest',
                'strategy': 'turbomode',
                'notes': f'{holding_period}d holding period',
                'created_at': datetime.now().isoformat()
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

        saved_count = 0

        for sample in samples:
            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO trades (
                        id, symbol, entry_date, entry_price, exit_date, exit_price,
                        position_size, outcome, profit_loss, profit_loss_pct,
                        exit_reason, entry_features_json, trade_type, strategy,
                        notes, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    sample['id'],
                    sample['symbol'],
                    sample['entry_date'],
                    sample['entry_price'],
                    sample['exit_date'],
                    sample['exit_price'],
                    sample['position_size'],
                    sample['outcome'],
                    sample['profit_loss'],
                    sample['profit_loss_pct'],
                    sample['exit_reason'],
                    sample['entry_features_json'],
                    sample['trade_type'],
                    sample['strategy'],
                    sample['notes'],
                    sample['created_at']
                ))
                saved_count += 1
            except Exception as e:
                logger.error(f"[ERROR] Failed to save sample: {e}")

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
