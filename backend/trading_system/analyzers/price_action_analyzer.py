"""
Price Action Analyzer
Analyzes pure price patterns without traditional indicators
Focus: Candlestick patterns, support/resistance, chart patterns
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from trading_system.core.base_analyzer import BaseAnalyzer


class PriceActionAnalyzer(BaseAnalyzer):
    """
    Analyzes price action patterns without indicators
    - Candlestick patterns (engulfing, doji, hammer, etc.)
    - Support/resistance levels
    - Price structure (higher highs/lows)
    """

    def __init__(self):
        super().__init__(name="price_action")

    def analyze(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze pure price action"""
        try:
            # Fetch data
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if df.empty or len(df) < 20:
                return None

            # Flatten multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Candlestick patterns
            bullish_patterns = self._detect_bullish_patterns(df)
            bearish_patterns = self._detect_bearish_patterns(df)

            # Support/Resistance
            support_strength = self._calculate_support_strength(df)
            resistance_strength = self._calculate_resistance_strength(df)

            # Price structure
            higher_highs = self._detect_higher_highs(df)
            higher_lows = self._detect_higher_lows(df)

            # Calculate signal strength
            bullish_score = (
                bullish_patterns * 0.3 +
                support_strength * 0.3 +
                (higher_highs + higher_lows) * 0.2
            )

            bearish_score = (
                bearish_patterns * 0.3 +
                resistance_strength * 0.3
            )

            # Normalize to 0-1
            signal_strength = (bullish_score - bearish_score + 1) / 2
            signal_strength = max(0, min(1, signal_strength))

            # Determine direction
            if signal_strength > 0.6:
                direction = 'bullish'
            elif signal_strength < 0.4:
                direction = 'bearish'
            else:
                direction = 'neutral'

            # Confidence based on pattern strength
            confidence = abs(signal_strength - 0.5) * 2  # Convert to 0-1

            return {
                'signal_strength': signal_strength,
                'direction': direction,
                'confidence': confidence,
                'metadata': {
                    'bullish_patterns': bullish_patterns,
                    'bearish_patterns': bearish_patterns,
                    'support_strength': support_strength,
                    'resistance_strength': resistance_strength,
                    'higher_highs': higher_highs,
                    'higher_lows': higher_lows
                }
            }

        except Exception as e:
            return None

    def _detect_bullish_patterns(self, df: pd.DataFrame) -> float:
        """Detect bullish candlestick patterns"""
        if len(df) < 3:
            return 0.0

        score = 0.0
        recent = df.tail(3)

        # Bullish engulfing
        if self._is_bullish_engulfing(recent):
            score += 0.4

        # Hammer
        if self._is_hammer(recent.iloc[-1]):
            score += 0.3

        # Morning star
        if self._is_morning_star(recent):
            score += 0.3

        return min(1.0, score)

    def _detect_bearish_patterns(self, df: pd.DataFrame) -> float:
        """Detect bearish candlestick patterns"""
        if len(df) < 3:
            return 0.0

        score = 0.0
        recent = df.tail(3)

        # Bearish engulfing
        if self._is_bearish_engulfing(recent):
            score += 0.4

        # Shooting star
        if self._is_shooting_star(recent.iloc[-1]):
            score += 0.3

        # Evening star
        if self._is_evening_star(recent):
            score += 0.3

        return min(1.0, score)

    def _is_bullish_engulfing(self, df: pd.DataFrame) -> bool:
        """Check for bullish engulfing pattern"""
        if len(df) < 2:
            return False

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        # Previous candle is bearish
        prev_bearish = prev['Close'] < prev['Open']

        # Current candle is bullish
        curr_bullish = curr['Close'] > curr['Open']

        # Current engulfs previous
        engulfs = (curr['Open'] < prev['Close'] and
                  curr['Close'] > prev['Open'])

        return prev_bearish and curr_bullish and engulfs

    def _is_bearish_engulfing(self, df: pd.DataFrame) -> bool:
        """Check for bearish engulfing pattern"""
        if len(df) < 2:
            return False

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        # Previous candle is bullish
        prev_bullish = prev['Close'] > prev['Open']

        # Current candle is bearish
        curr_bearish = curr['Close'] < curr['Open']

        # Current engulfs previous
        engulfs = (curr['Open'] > prev['Close'] and
                  curr['Close'] < prev['Open'])

        return prev_bullish and curr_bearish and engulfs

    def _is_hammer(self, candle: pd.Series) -> bool:
        """Check if candle is a hammer"""
        body = abs(candle['Close'] - candle['Open'])
        lower_shadow = min(candle['Open'], candle['Close']) - candle['Low']
        upper_shadow = candle['High'] - max(candle['Open'], candle['Close'])

        # Long lower shadow, small body, small upper shadow
        return (lower_shadow > body * 2 and
                upper_shadow < body * 0.3 and
                body > 0)

    def _is_shooting_star(self, candle: pd.Series) -> bool:
        """Check if candle is a shooting star"""
        body = abs(candle['Close'] - candle['Open'])
        lower_shadow = min(candle['Open'], candle['Close']) - candle['Low']
        upper_shadow = candle['High'] - max(candle['Open'], candle['Close'])

        # Long upper shadow, small body, small lower shadow
        return (upper_shadow > body * 2 and
                lower_shadow < body * 0.3 and
                body > 0)

    def _is_morning_star(self, df: pd.DataFrame) -> bool:
        """Check for morning star pattern"""
        if len(df) < 3:
            return False

        first = df.iloc[-3]
        second = df.iloc[-2]
        third = df.iloc[-1]

        # First: Large bearish
        first_bearish = first['Close'] < first['Open']
        first_body = abs(first['Close'] - first['Open'])

        # Second: Small body (doji-like)
        second_body = abs(second['Close'] - second['Open'])

        # Third: Large bullish
        third_bullish = third['Close'] > third['Open']
        third_body = abs(third['Close'] - third['Open'])

        return (first_bearish and
                second_body < first_body * 0.3 and
                third_bullish and
                third_body > first_body * 0.5)

    def _is_evening_star(self, df: pd.DataFrame) -> bool:
        """Check for evening star pattern"""
        if len(df) < 3:
            return False

        first = df.iloc[-3]
        second = df.iloc[-2]
        third = df.iloc[-1]

        # First: Large bullish
        first_bullish = first['Close'] > first['Open']
        first_body = abs(first['Close'] - first['Open'])

        # Second: Small body (doji-like)
        second_body = abs(second['Close'] - second['Open'])

        # Third: Large bearish
        third_bearish = third['Close'] < third['Open']
        third_body = abs(third['Close'] - third['Open'])

        return (first_bullish and
                second_body < first_body * 0.3 and
                third_bearish and
                third_body > first_body * 0.5)

    def _calculate_support_strength(self, df: pd.DataFrame) -> float:
        """Calculate strength of support at current price"""
        if len(df) < 20:
            return 0.0

        current_price = df['Close'].iloc[-1]
        lows = df['Low'].tail(20)

        # Count how many times price bounced near this level
        support_touches = sum(abs(low - current_price) / current_price < 0.02 for low in lows)

        return min(1.0, support_touches / 5)

    def _calculate_resistance_strength(self, df: pd.DataFrame) -> float:
        """Calculate strength of resistance at current price"""
        if len(df) < 20:
            return 0.0

        current_price = df['Close'].iloc[-1]
        highs = df['High'].tail(20)

        # Count how many times price rejected from this level
        resistance_touches = sum(abs(high - current_price) / current_price < 0.02 for high in highs)

        return min(1.0, resistance_touches / 5)

    def _detect_higher_highs(self, df: pd.DataFrame) -> float:
        """Detect if making higher highs (uptrend structure)"""
        if len(df) < 10:
            return 0.0

        highs = df['High'].tail(10)
        peaks = []

        for i in range(1, len(highs) - 1):
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                peaks.append(highs.iloc[i])

        if len(peaks) < 2:
            return 0.0

        # Check if peaks are increasing
        increasing = all(peaks[i] > peaks[i-1] for i in range(1, len(peaks)))

        return 1.0 if increasing else 0.0

    def _detect_higher_lows(self, df: pd.DataFrame) -> float:
        """Detect if making higher lows (uptrend structure)"""
        if len(df) < 10:
            return 0.0

        lows = df['Low'].tail(10)
        troughs = []

        for i in range(1, len(lows) - 1):
            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                troughs.append(lows.iloc[i])

        if len(troughs) < 2:
            return 0.0

        # Check if troughs are increasing
        increasing = all(troughs[i] > troughs[i-1] for i in range(1, len(troughs)))

        return 1.0 if increasing else 0.0
