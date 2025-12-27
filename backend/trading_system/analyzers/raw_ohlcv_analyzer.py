"""
Raw OHLCV Analyzer
Minimal preprocessing - let ML discover patterns naturally
Focus: Raw Open, High, Low, Close, Volume data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from trading_system.core.base_analyzer import BaseAnalyzer


class RawOHLCVAnalyzer(BaseAnalyzer):
    """
    Provides raw OHLCV data with minimal preprocessing
    Let the ML model discover patterns naturally from raw price/volume
    """

    def __init__(self):
        super().__init__(name="raw_ohlcv")

    def analyze(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze raw OHLCV data"""
        try:
            # Fetch data
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if df.empty or len(df) < 10:
                return None

            # Flatten multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Get recent data (last 10 days)
            recent = df.tail(10)

            # Extract raw OHLCV features
            open_prices = recent['Open'].values
            high_prices = recent['High'].values
            low_prices = recent['Low'].values
            close_prices = recent['Close'].values
            volumes = recent['Volume'].values

            # Calculate simple statistics (normalized)
            # Price momentum
            price_change = (close_prices[-1] - close_prices[0]) / close_prices[0]

            # Volatility (high-low range)
            avg_range = np.mean((high_prices - low_prices) / close_prices)

            # Volume change
            volume_change = (volumes[-1] - np.mean(volumes[:-1])) / np.mean(volumes[:-1]) if np.mean(volumes[:-1]) > 0 else 0

            # Body size (close-open)
            body_sizes = np.abs(close_prices - open_prices) / open_prices
            avg_body = np.mean(body_sizes)

            # Recent candle direction
            recent_bullish = sum(1 for i in range(len(close_prices)) if close_prices[i] > open_prices[i])
            bullish_ratio = recent_bullish / len(close_prices)

            # Simple signal based on raw data patterns
            # Positive momentum + increasing volume = bullish
            momentum_score = min(1.0, max(0.0, (price_change + 0.1) / 0.2))  # Normalize -0.1 to +0.1 range
            volume_score = min(1.0, max(0.0, (volume_change + 0.5) / 1.0))  # Normalize -0.5 to +0.5 range
            direction_score = bullish_ratio  # 0 to 1

            # Combined signal
            signal_strength = (momentum_score * 0.4 + volume_score * 0.3 + direction_score * 0.3)

            # Determine direction
            if signal_strength > 0.6:
                direction = 'bullish'
            elif signal_strength < 0.4:
                direction = 'bearish'
            else:
                direction = 'neutral'

            # Confidence based on data clarity
            confidence = abs(signal_strength - 0.5) * 2

            return {
                'signal_strength': signal_strength,
                'direction': direction,
                'confidence': confidence,
                'metadata': {
                    # Raw features for ML to learn from
                    'price_change_pct': price_change * 100,
                    'avg_volatility': avg_range,
                    'volume_change_pct': volume_change * 100,
                    'avg_body_size': avg_body,
                    'bullish_candle_ratio': bullish_ratio,
                    # Recent OHLCV (last 3 days for pattern learning)
                    'recent_open': list(open_prices[-3:]),
                    'recent_high': list(high_prices[-3:]),
                    'recent_low': list(low_prices[-3:]),
                    'recent_close': list(close_prices[-3:]),
                    'recent_volume': list(volumes[-3:]),
                    # Normalized features
                    'momentum_score': momentum_score,
                    'volume_score': volume_score,
                    'direction_score': direction_score
                }
            }

        except Exception as e:
            return None

    def _calculate_price_patterns(self, closes: np.ndarray) -> Dict[str, float]:
        """Extract simple price patterns"""
        if len(closes) < 5:
            return {}

        patterns = {}

        # Consecutive up days
        up_days = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        patterns['consecutive_up_ratio'] = up_days / (len(closes) - 1)

        # Price acceleration
        first_half_change = (closes[len(closes)//2] - closes[0]) / closes[0]
        second_half_change = (closes[-1] - closes[len(closes)//2]) / closes[len(closes)//2]
        patterns['acceleration'] = second_half_change - first_half_change

        # Price stability (lower std = more stable)
        returns = np.diff(closes) / closes[:-1]
        patterns['stability'] = 1 - min(1.0, np.std(returns))

        return patterns

    def _calculate_volume_patterns(self, volumes: np.ndarray) -> Dict[str, float]:
        """Extract simple volume patterns"""
        if len(volumes) < 5:
            return {}

        patterns = {}

        # Volume trend (increasing vs decreasing)
        first_half = np.mean(volumes[:len(volumes)//2])
        second_half = np.mean(volumes[len(volumes)//2:])

        if first_half > 0:
            patterns['volume_trend'] = (second_half - first_half) / first_half
        else:
            patterns['volume_trend'] = 0

        # Volume consistency
        patterns['volume_consistency'] = 1 - min(1.0, np.std(volumes) / (np.mean(volumes) + 1))

        return patterns
