"""
MACD Analyzer
Moving Average Convergence Divergence - Trend following indicator
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


class MACDAnalyzer(BaseAnalyzer):
    """
    Analyzes MACD for trend changes and momentum

    Bullish: MACD crosses above signal line
    Bearish: MACD crosses below signal line
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(name="macd")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal

    def calculate_macd(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate MACD, Signal line, and Histogram"""
        # Calculate EMAs
        ema_fast = prices.ewm(span=self.fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.slow, adjust=False).mean()

        # MACD line
        macd = ema_fast - ema_slow

        # Signal line
        signal = macd.ewm(span=self.signal_period, adjust=False).mean()

        # Histogram
        histogram = macd - signal

        return {
            'macd': macd.iloc[-1],
            'signal': signal.iloc[-1],
            'histogram': histogram.iloc[-1],
            'histogram_prev': histogram.iloc[-2] if len(histogram) > 1 else 0.0
        }

    def analyze(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze MACD for a symbol"""
        try:
            # Fetch data
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if df.empty or len(df) < self.slow + self.signal_period + 5:
                return {
                    'signal_strength': 0.5,
                    'direction': 'neutral',
                    'confidence': 0.0,
                    'metadata': {'error': 'Insufficient data'}
                }

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Calculate MACD
            macd_data = self.calculate_macd(df['Close'])

            # Determine signal
            histogram = macd_data['histogram']
            histogram_prev = macd_data['histogram_prev']

            # Check for crossovers
            bullish_crossover = histogram > 0 and histogram_prev <= 0
            bearish_crossover = histogram < 0 and histogram_prev >= 0

            if bullish_crossover:
                # Strong bullish signal
                signal_strength = 0.8
                direction = 'bullish'
                confidence = 0.85
            elif bearish_crossover:
                # Strong bearish signal
                signal_strength = 0.2
                direction = 'bearish'
                confidence = 0.85
            elif histogram > 0:
                # MACD above signal (bullish territory)
                signal_strength = 0.5 + min(0.4, abs(histogram) * 10)
                direction = 'bullish'
                confidence = min(0.7, abs(histogram) * 20)
            elif histogram < 0:
                # MACD below signal (bearish territory)
                signal_strength = 0.5 - min(0.4, abs(histogram) * 10)
                direction = 'bearish'
                confidence = min(0.7, abs(histogram) * 20)
            else:
                signal_strength = 0.5
                direction = 'neutral'
                confidence = 0.3

            return {
                'signal_strength': float(max(0.0, min(1.0, signal_strength))),
                'direction': direction,
                'confidence': float(max(0.0, min(1.0, confidence))),
                'metadata': {
                    'macd': float(macd_data['macd']),
                    'signal': float(macd_data['signal']),
                    'histogram': float(histogram),
                    'crossover': 'bullish' if bullish_crossover else 'bearish' if bearish_crossover else 'none',
                    'price': float(df['Close'].iloc[-1])
                }
            }

        except Exception as e:
            return {
                'signal_strength': 0.5,
                'direction': 'neutral',
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
