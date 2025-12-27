"""
Market Structure Analyzer
Analyzes market structure without lagging indicators
Focus: Higher highs/lows, swing points, breakouts, trend structure
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
from trading_system.core.base_analyzer import BaseAnalyzer


class MarketStructureAnalyzer(BaseAnalyzer):
    """
    Analyzes market structure and trend behavior
    - Higher highs and higher lows (uptrend)
    - Lower highs and lower lows (downtrend)
    - Breakouts from consolidation
    - Swing point analysis
    """

    def __init__(self):
        super().__init__(name="market_structure")

    def analyze(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze market structure"""
        try:
            # Fetch data
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if df.empty or len(df) < 20:
                return None

            # Flatten multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Swing point analysis
            swing_highs = self._find_swing_highs(df)
            swing_lows = self._find_swing_lows(df)

            # Trend structure
            higher_highs_score = self._calculate_higher_highs(swing_highs)
            higher_lows_score = self._calculate_higher_lows(swing_lows)
            lower_highs_score = self._calculate_lower_highs(swing_highs)
            lower_lows_score = self._calculate_lower_lows(swing_lows)

            # Breakout detection
            breakout_score = self._detect_breakout(df)

            # Consolidation vs trending
            consolidation_score = self._detect_consolidation(df)
            trending_score = 1 - consolidation_score

            # Calculate signal strength
            # Uptrend = higher highs + higher lows + breakout
            uptrend_score = (
                higher_highs_score * 0.35 +
                higher_lows_score * 0.35 +
                breakout_score * 0.2 +
                trending_score * 0.1
            )

            # Downtrend = lower highs + lower lows
            downtrend_score = (
                lower_highs_score * 0.4 +
                lower_lows_score * 0.4 +
                trending_score * 0.2
            )

            # Normalize to 0-1
            signal_strength = (uptrend_score - downtrend_score + 1) / 2
            signal_strength = max(0, min(1, signal_strength))

            # Determine direction
            if signal_strength > 0.6:
                direction = 'bullish'
            elif signal_strength < 0.4:
                direction = 'bearish'
            else:
                direction = 'neutral'

            # Confidence based on structure clarity
            confidence = abs(signal_strength - 0.5) * 2

            return {
                'signal_strength': signal_strength,
                'direction': direction,
                'confidence': confidence,
                'metadata': {
                    'higher_highs': higher_highs_score,
                    'higher_lows': higher_lows_score,
                    'lower_highs': lower_highs_score,
                    'lower_lows': lower_lows_score,
                    'breakout_strength': breakout_score,
                    'trending': trending_score,
                    'consolidating': consolidation_score,
                    'swing_highs_count': len(swing_highs),
                    'swing_lows_count': len(swing_lows)
                }
            }

        except Exception as e:
            return None

    def _find_swing_highs(self, df: pd.DataFrame, lookback: int = 5) -> List[float]:
        """Find swing high points"""
        highs = df['High'].values
        swing_highs = []

        for i in range(lookback, len(highs) - lookback):
            # Check if this is a local maximum
            is_swing = True
            for j in range(1, lookback + 1):
                if highs[i] <= highs[i-j] or highs[i] <= highs[i+j]:
                    is_swing = False
                    break

            if is_swing:
                swing_highs.append(highs[i])

        return swing_highs

    def _find_swing_lows(self, df: pd.DataFrame, lookback: int = 5) -> List[float]:
        """Find swing low points"""
        lows = df['Low'].values
        swing_lows = []

        for i in range(lookback, len(lows) - lookback):
            # Check if this is a local minimum
            is_swing = True
            for j in range(1, lookback + 1):
                if lows[i] >= lows[i-j] or lows[i] >= lows[i+j]:
                    is_swing = False
                    break

            if is_swing:
                swing_lows.append(lows[i])

        return swing_lows

    def _calculate_higher_highs(self, swing_highs: List[float]) -> float:
        """Check if making higher highs (uptrend)"""
        if len(swing_highs) < 2:
            return 0.0

        # Check if each high is higher than previous
        higher_count = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i] > swing_highs[i-1])

        return higher_count / (len(swing_highs) - 1)

    def _calculate_higher_lows(self, swing_lows: List[float]) -> float:
        """Check if making higher lows (uptrend)"""
        if len(swing_lows) < 2:
            return 0.0

        # Check if each low is higher than previous
        higher_count = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i] > swing_lows[i-1])

        return higher_count / (len(swing_lows) - 1)

    def _calculate_lower_highs(self, swing_highs: List[float]) -> float:
        """Check if making lower highs (downtrend)"""
        if len(swing_highs) < 2:
            return 0.0

        # Check if each high is lower than previous
        lower_count = sum(1 for i in range(1, len(swing_highs)) if swing_highs[i] < swing_highs[i-1])

        return lower_count / (len(swing_highs) - 1)

    def _calculate_lower_lows(self, swing_lows: List[float]) -> float:
        """Check if making lower lows (downtrend)"""
        if len(swing_lows) < 2:
            return 0.0

        # Check if each low is lower than previous
        lower_count = sum(1 for i in range(1, len(swing_lows)) if swing_lows[i] < swing_lows[i-1])

        return lower_count / (len(swing_lows) - 1)

    def _detect_breakout(self, df: pd.DataFrame) -> float:
        """
        Detect if price is breaking out of consolidation
        Breakout = recent high > previous highs
        """
        if len(df) < 20:
            return 0.0

        recent = df.tail(5)
        older = df.iloc[-20:-5]

        recent_high = recent['High'].max()
        older_high = older['High'].max()

        # Breakout if recent high exceeds previous range
        if recent_high > older_high:
            # Strength of breakout
            breakout_pct = (recent_high - older_high) / older_high
            return min(1.0, breakout_pct * 20)  # Scale to 0-1

        return 0.0

    def _detect_consolidation(self, df: pd.DataFrame) -> float:
        """
        Detect if price is consolidating (range-bound)
        Consolidation = low volatility, tight range
        """
        if len(df) < 20:
            return 0.0

        recent = df.tail(20)

        # Calculate range
        price_range = recent['High'].max() - recent['Low'].min()
        avg_price = recent['Close'].mean()

        if avg_price == 0:
            return 0.0

        # Range as percentage
        range_pct = price_range / avg_price

        # Tight range = consolidation
        if range_pct < 0.05:  # <5% range
            return 1.0
        elif range_pct > 0.15:  # >15% range
            return 0.0
        else:
            # Linear interpolation
            return 1 - ((range_pct - 0.05) / (0.15 - 0.05))

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """
        Calculate trend strength based on price structure
        Strong trend = consistent directional movement
        """
        if len(df) < 10:
            return 0.0

        closes = df['Close'].tail(10).values

        # Calculate direction consistency
        up_moves = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        down_moves = len(closes) - 1 - up_moves

        # Trend strength = predominance of one direction
        trend_strength = abs(up_moves - down_moves) / (len(closes) - 1)

        return trend_strength
