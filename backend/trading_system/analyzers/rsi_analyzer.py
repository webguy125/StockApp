"""
RSI Analyzer
Relative Strength Index - Momentum indicator
"""

import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trading_system.core.base_analyzer import BaseAnalyzer


class RSIAnalyzer(BaseAnalyzer):
    """
    Analyzes RSI (Relative Strength Index) for overbought/oversold conditions

    RSI > 70 = Overbought (bearish signal)
    RSI < 30 = Oversold (bullish signal)
    RSI 40-60 = Neutral
    """

    def __init__(self, period: int = 14):
        super().__init__(name="rsi")
        self.period = period

    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI using the standard formula"""
        deltas = prices.diff()

        gain = deltas.where(deltas > 0, 0.0)
        loss = -deltas.where(deltas < 0, 0.0)

        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not rsi.empty else 50.0

    def analyze(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze RSI for a symbol"""
        try:
            # Fetch data
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if df.empty or len(df) < self.period + 5:
                return {
                    'signal_strength': 0.5,
                    'direction': 'neutral',
                    'confidence': 0.0,
                    'metadata': {'error': 'Insufficient data'}
                }

            # Flatten multi-index if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Calculate RSI
            rsi = self.calculate_rsi(df['Close'])

            # Determine signal
            if rsi < 30:
                # Oversold - bullish signal
                signal_strength = 1.0 - (rsi / 30)  # Lower RSI = stronger signal
                direction = 'bullish'
                confidence = min(0.9, (30 - rsi) / 30)  # More oversold = higher confidence
            elif rsi > 70:
                # Overbought - bearish signal
                signal_strength = (rsi - 70) / 30  # Higher RSI = stronger signal
                direction = 'bearish'
                confidence = min(0.9, (rsi - 70) / 30)
            else:
                # Neutral zone
                signal_strength = 0.5
                direction = 'neutral'
                confidence = 0.5 - abs(rsi - 50) / 40  # Closer to 50 = lower confidence

            return {
                'signal_strength': float(max(0.0, min(1.0, signal_strength))),
                'direction': direction,
                'confidence': float(max(0.0, min(1.0, confidence))),
                'metadata': {
                    'rsi': float(rsi),
                    'period': self.period,
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
