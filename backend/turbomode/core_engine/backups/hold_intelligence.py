
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

"""
HOLD Intelligence Module - Contamination-Proof Options Strategy Engine

This module evaluates HOLD signals ONLY and is completely isolated from BUY/SELL logic.
Designed for iron condor and neutral options strategies.

Key Principles:
- NO mixing with directional (BUY/SELL) performance data
- NO synthetic training data contamination
- NO shared scoring with adaptive stock ranker
- Deterministic, reproducible scoring

Outputs:
- Top 20 HOLD candidates for iron condors (nightly)
- Stability scores (price drift, range compression)
- Volatility compression scores (IV/ATR decay)
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os
import logging

import pandas as pd
import numpy as np

from master_market_data.market_data_api import get_market_data_api

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HoldIntelligence:
    """
    Contamination-proof HOLD signal intelligence engine.

    Evaluates HOLD signals for neutral options strategies (iron condors, strangles).
    Completely isolated from directional (BUY/SELL) performance tracking.
    """

    def __init__(self, db_path: str = 'backend/data/turbomode.db'):
        self.db_path = db_path
        self.market_data_api = get_market_data_api()

        # Scoring weights for stability
        self.STABILITY_WEIGHTS = {
            'price_drift_pct': 0.35,
            'max_intraday_drift_pct': 0.25,
            'volatility_drift': 0.20,
            'range_compression_score': 0.15,
            'sector_stability_factor': 0.05
        }

        # Scoring weights for volatility compression
        self.VOLATILITY_WEIGHTS = {
            'iv_compression_pct': 0.40,
            'atr_compression_pct': 0.30,
            'bandwidth_compression': 0.20,
            'volatility_floor_test': 0.10
        }

        # Selection thresholds
        self.MIN_STABILITY_SCORE = 60
        self.MIN_VOLATILITY_SCORE = 50
        self.TOP_N = 20

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_file = os.path.join(base_dir, "data", "top_20_hold_candidates.json")

    def calculate_price_drift(self, entry_price: float, exit_price: float) -> float:
        """
        Calculate absolute percent drift from entry to exit.

        Lower is better for HOLD signals (indicates stability).
        """
        return abs((exit_price - entry_price) / entry_price) * 100.0

    def calculate_max_intraday_drift(self, symbol: str, entry_date: str, exit_date: str,
                                     entry_price: float) -> Optional[float]:
        """
        Calculate largest intraday deviation during holding period.

        Returns:
            Maximum absolute percent deviation from entry price, or None if data unavailable
        """
        try:
            # Calculate days between entry and exit
            entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
            exit_dt = datetime.strptime(exit_date, '%Y-%m-%d')
            days_held = (exit_dt - entry_dt).days

            # Get daily candles for holding period
            df = self.market_data_api.get_candles(symbol, timeframe='1d', days_back=days_held + 5)

            if df is None or len(df) == 0:
                return None

            # Filter to holding period
            df['date'] = pd.to_datetime(df['date'])
            df = df[(df['date'] >= entry_dt) & (df['date'] <= exit_dt)]

            if len(df) == 0:
                return None

            # Calculate max deviation from entry price using high and low
            high_dev = ((df['high'].max() - entry_price) / entry_price) * 100.0
            low_dev = ((entry_price - df['low'].min()) / entry_price) * 100.0

            max_drift = max(abs(high_dev), abs(low_dev))

            return max_drift

        except Exception as e:
            logger.warning(f"Failed to calculate max intraday drift for {symbol}: {e}")
            return None

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def calculate_volatility_drift(self, symbol: str, entry_date: str, exit_date: str) -> Optional[float]:
        """
        Calculate change in ATR (proxy for volatility) during holding period.

        Negative drift = volatility compression (good for HOLD)
        Positive drift = volatility expansion (bad for HOLD)
        """
        try:
            entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
            exit_dt = datetime.strptime(exit_date, '%Y-%m-%d')
            days_held = (exit_dt - entry_dt).days

            # Get daily candles
            df = self.market_data_api.get_candles(symbol, timeframe='1d', days_back=days_held + 30)

            if df is None or len(df) < 20:
                return None

            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            # Calculate ATR
            df['atr'] = self.calculate_atr(df)

            # Get ATR at entry and exit
            entry_atr_df = df[df['date'] <= entry_dt].tail(1)
            exit_atr_df = df[df['date'] <= exit_dt].tail(1)

            if len(entry_atr_df) == 0 or len(exit_atr_df) == 0:
                return None

            entry_atr = entry_atr_df['atr'].iloc[0]
            exit_atr = exit_atr_df['atr'].iloc[0]

            if pd.isna(entry_atr) or pd.isna(exit_atr) or entry_atr == 0:
                return None

            # Calculate percent change in ATR
            volatility_drift = ((exit_atr - entry_atr) / entry_atr) * 100.0

            return volatility_drift

        except Exception as e:
            logger.warning(f"Failed to calculate volatility drift for {symbol}: {e}")
            return None

    def calculate_range_compression(self, symbol: str, entry_date: str, exit_date: str,
                                    entry_price: float) -> Optional[float]:
        """
        Calculate how tightly price stayed within a band during holding period.

        Uses Bollinger Band width compression as proxy.

        Returns:
            Score 0-100, where 100 = maximum compression (tight range)
        """
        try:
            entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
            exit_dt = datetime.strptime(exit_date, '%Y-%m-%d')
            days_held = (exit_dt - entry_dt).days

            df = self.market_data_api.get_candles(symbol, timeframe='1d', days_back=days_held + 30)

            if df is None or len(df) < 20:
                return None

            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            # Calculate Bollinger Bands
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['std_20'] = df['close'].rolling(window=20).std()
            df['bb_width'] = (2 * df['std_20']) / df['sma_20'] * 100.0  # Percent width

            # Filter to holding period
            holding_df = df[(df['date'] >= entry_dt) & (df['date'] <= exit_dt)]

            if len(holding_df) == 0:
                return None

            # Average BB width during holding period
            avg_width = holding_df['bb_width'].mean()

            if pd.isna(avg_width):
                return None

            # Convert to compression score (inverse of width)
            # Typical BB width ranges from 2% (compressed) to 10% (expanded)
            # Normalize to 0-100 scale
            compression_score = max(0, min(100, 100 - (avg_width * 10)))

            return compression_score

        except Exception as e:
            logger.warning(f"Failed to calculate range compression for {symbol}: {e}")
            return None

    def calculate_stability_metrics(self, row: Dict) -> Dict:
        """
        Calculate all stability metrics for a HOLD signal.

        Args:
            row: Dictionary containing signal data (symbol, entry_date, exit_date, etc.)

        Returns:
            Dictionary with stability metrics and composite score
        """
        symbol = row['symbol']
        entry_date = row['entry_date']
        exit_date = row['exit_date']
        entry_price = row['entry_price']
        exit_price = row['exit_price']
        sector = row.get('sector', 'Unknown')

        # Calculate individual metrics
        price_drift = self.calculate_price_drift(entry_price, exit_price)
        max_drift = self.calculate_max_intraday_drift(symbol, entry_date, exit_date, entry_price)
        vol_drift = self.calculate_volatility_drift(symbol, entry_date, exit_date)
        range_comp = self.calculate_range_compression(symbol, entry_date, exit_date, entry_price)

        # Sector stability factor (placeholder - can be enhanced with sector volatility data)
        sector_factor = 50.0  # Neutral baseline

        # Normalize metrics to 0-100 scale
        # Price drift: 0% = 100, 10%+ = 0
        price_drift_score = max(0, min(100, 100 - (price_drift * 10)))

        # Max intraday drift: 0% = 100, 15%+ = 0
        max_drift_score = 100 if max_drift is None else max(0, min(100, 100 - (max_drift * 6.67)))

        # Volatility drift: -20% (compression) = 100, +20% (expansion) = 0
        vol_drift_score = 100 if vol_drift is None else max(0, min(100, 50 - (vol_drift * 2.5)))

        # Range compression: already 0-100
        range_comp_score = 50 if range_comp is None else range_comp

        # Calculate weighted composite stability score
        stability_score = (
            price_drift_score * self.STABILITY_WEIGHTS['price_drift_pct'] +
            max_drift_score * self.STABILITY_WEIGHTS['max_intraday_drift_pct'] +
            vol_drift_score * self.STABILITY_WEIGHTS['volatility_drift'] +
            range_comp_score * self.STABILITY_WEIGHTS['range_compression_score'] +
            sector_factor * self.STABILITY_WEIGHTS['sector_stability_factor']
        )

        return {
            'price_drift_pct': price_drift,
            'max_intraday_drift_pct': max_drift,
            'volatility_drift_pct': vol_drift,
            'range_compression_score': range_comp,
            'sector_stability_factor': sector_factor,
            'stability_score': round(stability_score, 2)
        }

    def calculate_volatility_compression_metrics(self, row: Dict) -> Dict:
        """
        Calculate volatility compression metrics for a HOLD signal.

        Args:
            row: Dictionary containing signal data

        Returns:
            Dictionary with volatility metrics and composite score
        """
        symbol = row['symbol']
        entry_date = row['entry_date']
        exit_date = row['exit_date']

        try:
            entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
            exit_dt = datetime.strptime(exit_date, '%Y-%m-%d')
            days_held = (exit_dt - entry_dt).days

            df = self.market_data_api.get_candles(symbol, timeframe='1d', days_back=days_held + 30)

            if df is None or len(df) < 20:
                return {
                    'iv_compression_pct': None,
                    'atr_compression_pct': None,
                    'bandwidth_compression': None,
                    'volatility_floor_test': 50.0,
                    'volatility_score': 50.0
                }

            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            # Calculate ATR
            df['atr'] = self.calculate_atr(df)
            df['atr_pct'] = (df['atr'] / df['close']) * 100.0

            # Calculate BB width
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['std_20'] = df['close'].rolling(window=20).std()
            df['bb_width'] = (2 * df['std_20']) / df['sma_20'] * 100.0

            # Get entry and exit values
            entry_df = df[df['date'] <= entry_dt].tail(1)
            exit_df = df[df['date'] <= exit_dt].tail(1)

            if len(entry_df) == 0 or len(exit_df) == 0:
                return {
                    'iv_compression_pct': None,
                    'atr_compression_pct': None,
                    'bandwidth_compression': None,
                    'volatility_floor_test': 50.0,
                    'volatility_score': 50.0
                }

            entry_atr = entry_df['atr_pct'].iloc[0]
            exit_atr = exit_df['atr_pct'].iloc[0]
            entry_bb = entry_df['bb_width'].iloc[0]
            exit_bb = exit_df['bb_width'].iloc[0]

            # Calculate compression percentages
            atr_compression = None
            if not pd.isna(entry_atr) and not pd.isna(exit_atr) and entry_atr > 0:
                atr_compression = ((entry_atr - exit_atr) / entry_atr) * 100.0

            bandwidth_compression = None
            if not pd.isna(entry_bb) and not pd.isna(exit_bb) and entry_bb > 0:
                bandwidth_compression = ((entry_bb - exit_bb) / entry_bb) * 100.0

            # IV compression (placeholder - would need options data)
            iv_compression = None

            # Volatility floor test: check if exit volatility is in bottom quartile of recent range
            recent_atr = df.tail(60)['atr_pct']
            if not pd.isna(exit_atr) and len(recent_atr) > 0:
                percentile = (recent_atr < exit_atr).sum() / len(recent_atr) * 100
                volatility_floor = 100 if percentile <= 25 else 50
            else:
                volatility_floor = 50

            # Normalize to 0-100 scores
            # ATR compression: 30%+ compression = 100, 0% = 50, -30% expansion = 0
            atr_score = 50 if atr_compression is None else max(0, min(100, 50 + (atr_compression * 1.67)))

            # Bandwidth compression: same logic
            bandwidth_score = 50 if bandwidth_compression is None else max(0, min(100, 50 + (bandwidth_compression * 1.67)))

            # IV compression: placeholder
            iv_score = 50

            # Calculate weighted composite volatility score
            volatility_score = (
                iv_score * self.VOLATILITY_WEIGHTS['iv_compression_pct'] +
                atr_score * self.VOLATILITY_WEIGHTS['atr_compression_pct'] +
                bandwidth_score * self.VOLATILITY_WEIGHTS['bandwidth_compression'] +
                volatility_floor * self.VOLATILITY_WEIGHTS['volatility_floor_test']
            )

            return {
                'iv_compression_pct': iv_compression,
                'atr_compression_pct': atr_compression,
                'bandwidth_compression': bandwidth_compression,
                'volatility_floor_test': volatility_floor,
                'volatility_score': round(volatility_score, 2)
            }

        except Exception as e:
            logger.warning(f"Failed to calculate volatility metrics for {symbol}: {e}")
            return {
                'iv_compression_pct': None,
                'atr_compression_pct': None,
                'bandwidth_compression': None,
                'volatility_floor_test': 50.0,
                'volatility_score': 50.0
            }

    def analyze_hold_signals(self) -> pd.DataFrame:
        """
        Analyze all closed HOLD signals from signal_history.

        CONTAMINATION-PROOF: Only processes HOLD signals, never BUY/SELL.

        Returns:
            DataFrame with HOLD signals and their stability/volatility scores
        """
        logger.info("\nHOLD Intelligence Analysis")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("")

        # Load HOLD signals ONLY from signal_history
        conn = sqlite3.connect(self.db_path)
        query = """
        SELECT
            symbol,
            signal_type,
            confidence,
            entry_date,
            entry_price,
            exit_date,
            exit_price,
            exit_reason,
            profit_loss_pct,
            hold_days,
            market_cap,
            sector,
            created_at
        FROM signal_history
        WHERE signal_type = 'HOLD'
        AND exit_date IS NOT NULL
        AND exit_price IS NOT NULL
        ORDER BY exit_date DESC
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        if len(df) == 0:
            logger.info("[INFO] No closed HOLD signals found in signal_history")
            logger.info("[INFO] HOLD intelligence will become available as HOLD signals close")
            logger.info("")
            return None

        logger.info(f"Total closed HOLD signals: {len(df)}")
        logger.info(f"Unique symbols: {df['symbol'].nunique()}")
        logger.info(f"Date range: {df['entry_date'].min()} to {df['exit_date'].max()}")
        logger.info("")

        # Calculate metrics for each HOLD signal
        results = []

        for idx, row in df.iterrows():
            row_dict = row.to_dict()

            # Calculate stability metrics
            stability = self.calculate_stability_metrics(row_dict)

            # Calculate volatility compression metrics
            volatility = self.calculate_volatility_compression_metrics(row_dict)

            # Combine all metrics
            result = {
                'symbol': row['symbol'],
                'entry_date': row['entry_date'],
                'exit_date': row['exit_date'],
                'hold_days': row['hold_days'],
                'entry_price': row['entry_price'],
                'exit_price': row['exit_price'],
                'profit_loss_pct': row['profit_loss_pct'],
                'sector': row.get('sector', 'Unknown'),
                'market_cap': row.get('market_cap', 'Unknown'),
                **stability,
                **volatility
            }

            results.append(result)

        results_df = pd.DataFrame(results)

        return results_df

    def get_top_20_hold_candidates(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select Top 20 HOLD candidates for iron condor strategies.

        Selection criteria:
        - Minimum stability score >= 60
        - Minimum volatility score >= 50
        - Sort by stability_score DESC, then volatility_score DESC
        - Exclude symbols with recent directional signals (future enhancement)
        - Limit to 20
        """
        if results_df is None or len(results_df) == 0:
            logger.info("[INFO] No HOLD signals available for Top 20 selection")
            return None

        # Filter by minimum thresholds
        filtered = results_df[
            (results_df['stability_score'] >= self.MIN_STABILITY_SCORE) &
            (results_df['volatility_score'] >= self.MIN_VOLATILITY_SCORE)
        ].copy()

        if len(filtered) == 0:
            logger.info(f"[INFO] No HOLD signals meet minimum thresholds:")
            logger.info(f"  Stability score >= {self.MIN_STABILITY_SCORE}")
            logger.info(f"  Volatility score >= {self.MIN_VOLATILITY_SCORE}")
            return None

        # Sort by stability and volatility scores
        filtered = filtered.sort_values(
            by=['stability_score', 'volatility_score'],
            ascending=False
        )

        # Take top 20
        top_20 = filtered.head(self.TOP_N)

        logger.info("")
        logger.info("TOP 20 HOLD CANDIDATES FOR IRON CONDORS")
        logger.info("")
        logger.info(f"{'Rank':<5} {'Symbol':<8} {'Stability':<10} {'Volatility':<10} {'Drift%':<8} {'ATR Comp%':<10} {'Days':<6} {'Sector':<15}")
        logger.info("-" * 90)

        for idx, (_, row) in enumerate(top_20.iterrows(), 1):
            atr_comp = row['atr_compression_pct'] if row['atr_compression_pct'] is not None else 0.0
            logger.info(
                f"{idx:<5} {row['symbol']:<8} "
                f"{row['stability_score']:<10.2f} "
                f"{row['volatility_score']:<10.2f} "
                f"{row['price_drift_pct']:<8.2f} "
                f"{atr_comp:<10.2f} "
                f"{row['hold_days']:<6} "
                f"{row['sector']:<15}"
            )

        return top_20

    def save_top_20(self, top_20: pd.DataFrame):
        """Save Top 20 HOLD candidates to JSON file"""
        if top_20 is None or len(top_20) == 0:
            logger.info("[INFO] No Top 20 candidates to save")
            return

        output_data = {
            'timestamp': datetime.now().isoformat(),
            'top_20_hold_candidates': top_20.to_dict('records'),
            'selection_criteria': {
                'min_stability_score': self.MIN_STABILITY_SCORE,
                'min_volatility_score': self.MIN_VOLATILITY_SCORE,
                'limit': self.TOP_N
            }
        }

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info("")
        logger.info(f"[OK] Top 20 HOLD candidates saved to {self.output_file}")

    def run_analysis(self):
        """
        Execute full HOLD intelligence analysis pipeline.

        Returns:
            Dictionary with analysis results
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("HOLD INTELLIGENCE MODULE - CONTAMINATION-PROOF OPTIONS STRATEGY ENGINE")
        logger.info("=" * 80)
        logger.info("")

        # Analyze all closed HOLD signals
        results_df = self.analyze_hold_signals()

        if results_df is None:
            logger.info("")
            logger.info("=" * 80)
            logger.info("HOLD INTELLIGENCE: NO DATA AVAILABLE YET")
            logger.info("=" * 80)
            return None

        # Get Top 20 candidates
        top_20 = self.get_top_20_hold_candidates(results_df)

        # Save results
        self.save_top_20(top_20)

        logger.info("")
        logger.info("=" * 80)
        logger.info("HOLD INTELLIGENCE ANALYSIS COMPLETE")
        logger.info("=" * 80)

        return {
            'all_hold_signals': results_df.to_dict('records'),
            'top_20': top_20.to_dict('records') if top_20 is not None else [],
            'timestamp': datetime.now().isoformat()
        }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='HOLD Intelligence - Options Strategy Engine')
    parser.add_argument('--db', default='backend/data/turbomode.db', help='Path to database')

    args = parser.parse_args()

    db_path = args.db if args.db.startswith('/') or ':' in args.db else str(project_root / args.db)

    hold_intel = HoldIntelligence(db_path=db_path)
    results = hold_intel.run_analysis()

    if results:
        print(f"\n[SUCCESS] HOLD intelligence analysis complete")
        print(f"[INFO] Top 20 candidates: {len(results['top_20'])}")
