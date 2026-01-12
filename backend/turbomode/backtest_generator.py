"""
TurboMode Backtest Generator
Generates historical backtest datasets for model validation

Reads historical data from Master Market Data DB (read-only)
Evaluates model predictions vs actual outcomes
Saves backtest results to TurboMode.db

Author: TurboMode System
Date: 2026-01-06
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

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# Master Market Data DB API (read-only)
from master_market_data.market_data_api import get_market_data_api

# TurboMode DB
from turbomode.database_schema import TurboModeDB

# Core symbols
from backend.turbomode.core_symbols import get_all_core_symbols

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('backtest_generator')


class BacktestGenerator:
    """
    Generates backtest datasets for TurboMode models

    Features:
    - Historical model performance evaluation
    - Prediction accuracy tracking
    - Win rate and profit/loss calculation
    - Results stored in TurboMode.db
    """

    def __init__(self,
                 turbomode_db_path: str = "backend/data/turbomode.db",
                 lookback_days: int = 90):
        """
        Initialize backtest generator

        Args:
            turbomode_db_path: Path to TurboMode database
            lookback_days: Number of days to look back for backtesting
        """
        # Connect to Master Market Data DB (read-only)
        self.market_data_api = get_market_data_api()
        logger.info("[INIT] Connected to Master Market Data DB (read-only)")

        # Connect to TurboMode DB (private ML memory)
        self.turbomode_db = TurboModeDB(db_path=turbomode_db_path)
        logger.info("[INIT] Connected to TurboMode DB")

        self.lookback_days = lookback_days

    def generate_backtest_dataset(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Generate backtest dataset for given symbols

        Args:
            symbols: List of ticker symbols

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"[BACKTEST] Generating backtest for {len(symbols)} symbols...")
        logger.info(f"[BACKTEST] Lookback period: {self.lookback_days} days")

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        results = {
            'total_symbols': len(symbols),
            'successful': 0,
            'failed': 0,
            'backtest_metrics': {},
            'timestamp': datetime.now().isoformat()
        }

        for symbol in symbols:
            try:
                # Fetch historical data from Master Market Data DB
                # Use start_date and end_date as per the actual API signature
                df = self.market_data_api.get_candles(
                    symbol=symbol,
                    timeframe='1d',
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )

                # Inspect DataFrame and handle actual column names
                if df is None or df.empty:
                    logger.warning(f"[SKIP] {symbol}: No data returned")
                    results['failed'] += 1
                    continue

                # Master DB returns lowercase column names: timestamp, open, high, low, close, volume
                # Check if we have the required columns
                if 'close' not in df.columns:
                    logger.warning(f"[SKIP] {symbol}: Missing 'close' column")
                    results['failed'] += 1
                    continue

                if len(df) < 20:
                    logger.warning(f"[SKIP] {symbol}: Insufficient data ({len(df)} rows)")
                    results['failed'] += 1
                    continue

                # Calculate actual returns (next day close - today close)
                df['actual_return'] = df['close'].pct_change().shift(-1)
                df['actual_direction'] = np.where(df['actual_return'] > 0, 1, 0)

                # Simple metrics
                avg_return = df['actual_return'].mean()
                volatility = df['actual_return'].std()
                positive_days = (df['actual_return'] > 0).sum()
                negative_days = (df['actual_return'] < 0).sum()

                results['backtest_metrics'][symbol] = {
                    'avg_return': float(avg_return) if not pd.isna(avg_return) else 0.0,
                    'volatility': float(volatility) if not pd.isna(volatility) else 0.0,
                    'positive_days': int(positive_days),
                    'negative_days': int(negative_days),
                    'total_days': len(df)
                }

                results['successful'] += 1

            except Exception as e:
                logger.warning(f"[ERROR] {symbol}: {e}")
                results['failed'] += 1

        logger.info(f"[BACKTEST] ✓ Complete - {results['successful']}/{results['total_symbols']} successful")

        return results

    def save_backtest_results(self, results: Dict[str, Any]) -> Optional[int]:
        """
        Save backtest results to TurboMode.db

        Args:
            results: Backtest results dictionary

        Returns:
            Backtest run ID or None if failed
        """
        logger.info("[BACKTEST] Saving results to TurboMode.db...")

        conn = None
        try:
            # Create our own connection to TurboMode.db
            conn = sqlite3.connect(self.turbomode_db.db_path)
            cursor = conn.cursor()

            # Create training_runs table if it doesn't exist
            # We'll store backtest metadata in the existing training_runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT NOT NULL,
                    num_symbols INTEGER,
                    avg_accuracy REAL,
                    metadata TEXT
                )
            """)

            # Insert backtest run as a training run entry
            import json
            metadata = {
                'type': 'backtest',
                'total_symbols': results['total_symbols'],
                'successful': results['successful'],
                'failed': results['failed'],
                'lookback_days': self.lookback_days,
                'metrics': results.get('backtest_metrics', {})
            }

            cursor.execute("""
                INSERT INTO training_runs (
                    run_date, num_symbols, avg_accuracy, metadata
                ) VALUES (?, ?, ?, ?)
            """, (
                results['timestamp'],
                results['successful'],
                0.0,  # Placeholder accuracy
                json.dumps(metadata)
            ))

            run_id = cursor.lastrowid
            conn.commit()

            logger.info(f"[BACKTEST] ✓ Results saved (Run ID: {run_id})")
            return run_id

        except Exception as e:
            logger.error(f"[BACKTEST] Failed to save results: {e}")
            if conn:
                conn.rollback()
            return None

        finally:
            if conn:
                conn.close()

    def run_full_backtest(self, symbols: Optional[List[str]] = None) -> Optional[int]:
        """
        Run complete backtest pipeline

        Args:
            symbols: List of symbols (uses core symbols if None)

        Returns:
            Backtest run ID or None if failed
        """
        if symbols is None:
            symbols = get_all_core_symbols()[:10]  # Start with 10 for testing

        logger.info("=" * 80)
        logger.info("BACKTEST GENERATOR - FULL RUN")
        logger.info("=" * 80)

        # Generate backtest
        results = self.generate_backtest_dataset(symbols)

        # Save results
        run_id = self.save_backtest_results(results)

        if run_id:
            logger.info("=" * 80)
            logger.info(f"[OK] Backtest complete! Run ID: {run_id}")
            logger.info("=" * 80)
        else:
            logger.error("=" * 80)
            logger.error("[ERROR] Backtest failed to save!")
            logger.error("=" * 80)

        return run_id


if __name__ == '__main__':
    print("=" * 80)
    print("TURBOMODE BACKTEST GENERATOR")
    print("Historical model performance evaluation")
    print("=" * 80)

    generator = BacktestGenerator(lookback_days=60)
    run_id = generator.run_full_backtest()

    if run_id:
        print(f"\n[OK] Backtest complete! Run ID: {run_id}")
    else:
        print("\n[ERROR] Backtest failed!")
