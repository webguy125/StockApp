"""
Trend Analyzer
Moving averages and trend strength detection
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trading_system.core.base_analyzer import BaseAnalyzer


class TrendAnalyzer(BaseAnalyzer):
    """
    Analyzes trend using moving averages

    Bullish: Price > SMA and SMAs aligned upward
    Bearish: Price < SMA and SMAs aligned downward
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        super().__init__(name="trend")
        self.fast_period = fast_period
        self.slow_period = slow_period

    def analyze(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze trend using moving averages"""
        try:
            # Fetch data
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if df.empty or len(df) < self.slow_period + 5:
                return {
                    'signal_strength': 0.5,
                    'direction': 'neutral',
                    'confidence': 0.0,
                    'metadata': {'error': 'Insufficient data'}
                }

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Calculate moving averages
            sma_fast = df['Close'].rolling(window=self.fast_period).mean()
            sma_slow = df['Close'].rolling(window=self.slow_period).mean()

            current_price = df['Close'].iloc[-1]
            sma_fast_current = sma_fast.iloc[-1]
            sma_slow_current = sma_slow.iloc[-1]

            # Check alignment
            fast_above_slow = sma_fast_current > sma_slow_current
            price_above_fast = current_price > sma_fast_current
            price_above_slow = current_price > sma_slow_current

            # Calculate distance from MAs (as percentage)
            dist_from_fast = ((current_price - sma_fast_current) / sma_fast_current) * 100
            dist_from_slow = ((current_price - sma_slow_current) / sma_slow_current) * 100

            # Determine signal
            if price_above_fast and price_above_slow and fast_above_slow:
                # Strong uptrend
                signal_strength = 0.7 + min(0.3, dist_from_slow / 10)
                direction = 'bullish'
                confidence = 0.85
            elif not price_above_fast and not price_above_slow and not fast_above_slow:
                # Strong downtrend
                signal_strength = 0.3 - min(0.3, abs(dist_from_slow) / 10)
                direction = 'bearish'
                confidence = 0.85
            elif price_above_fast and fast_above_slow:
                # Moderate uptrend
                signal_strength = 0.6
                direction = 'bullish'
                confidence = 0.7
            elif not price_above_fast and not fast_above_slow:
                # Moderate downtrend
                signal_strength = 0.4
                direction = 'bearish'
                confidence = 0.7
            else:
                # Mixed signals
                signal_strength = 0.5
                direction = 'neutral'
                confidence = 0.4

            return {
                'signal_strength': float(max(0.0, min(1.0, signal_strength))),
                'direction': direction,
                'confidence': float(max(0.0, min(1.0, confidence))),
                'metadata': {
                    'sma_fast': float(sma_fast_current),
                    'sma_slow': float(sma_slow_current),
                    'price': float(current_price),
                    'dist_from_fast_pct': float(dist_from_fast),
                    'dist_from_slow_pct': float(dist_from_slow),
                    'trend': 'uptrend' if fast_above_slow else 'downtrend'
                }
            }

        except Exception as e:
            return {
                'signal_strength': 0.5,
                'direction': 'neutral',
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
