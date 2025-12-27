"""
Volume Profile Analyzer
Analyzes volume distribution and order flow without price indicators
Focus: Volume clusters, buying/selling pressure, institutional activity
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from trading_system.core.base_analyzer import BaseAnalyzer


class VolumeProfileAnalyzer(BaseAnalyzer):
    """
    Analyzes volume patterns and order flow
    - Volume distribution across price levels
    - Buying vs selling pressure
    - Institutional activity (large volume spikes)
    - Accumulation vs distribution
    """

    def __init__(self):
        super().__init__(name="volume_profile")

    def analyze(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze volume profile"""
        try:
            # Fetch data
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if df.empty or len(df) < 20:
                return None

            # Flatten multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Volume analysis
            volume_trend = self._calculate_volume_trend(df)
            buying_pressure = self._calculate_buying_pressure(df)
            selling_pressure = self._calculate_selling_pressure(df)
            institutional_activity = self._detect_institutional_activity(df)
            accumulation_score = self._calculate_accumulation(df)

            # Calculate signal strength
            bullish_score = (
                buying_pressure * 0.3 +
                accumulation_score * 0.3 +
                institutional_activity * 0.2 +
                volume_trend * 0.2
            )

            bearish_score = (
                selling_pressure * 0.4 +
                (1 - accumulation_score) * 0.3
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

            # Confidence based on volume clarity
            confidence = abs(signal_strength - 0.5) * 2

            return {
                'signal_strength': signal_strength,
                'direction': direction,
                'confidence': confidence,
                'metadata': {
                    'volume_trend': volume_trend,
                    'buying_pressure': buying_pressure,
                    'selling_pressure': selling_pressure,
                    'institutional_activity': institutional_activity,
                    'accumulation_score': accumulation_score
                }
            }

        except Exception as e:
            return None

    def _calculate_volume_trend(self, df: pd.DataFrame) -> float:
        """Calculate if volume is increasing (bullish) or decreasing"""
        if len(df) < 10:
            return 0.5

        recent_volume = df['Volume'].tail(5).mean()
        older_volume = df['Volume'].iloc[-15:-10].mean()

        if older_volume == 0:
            return 0.5

        ratio = recent_volume / older_volume

        # Normalize: >1.5 = strong increase, <0.7 = decrease
        if ratio > 1.5:
            return 1.0
        elif ratio < 0.7:
            return 0.0
        else:
            # Linear interpolation
            return (ratio - 0.7) / (1.5 - 0.7)

    def _calculate_buying_pressure(self, df: pd.DataFrame) -> float:
        """
        Calculate buying pressure based on price action and volume
        Up days with high volume = buying pressure
        """
        if len(df) < 10:
            return 0.5

        recent = df.tail(10)

        buying_volume = 0
        total_volume = 0

        for idx, row in recent.iterrows():
            volume = row['Volume']
            total_volume += volume

            # If close > open = buying day
            if row['Close'] > row['Open']:
                # Weight by how much it moved up
                strength = (row['Close'] - row['Open']) / row['Open']
                buying_volume += volume * (1 + strength)

        if total_volume == 0:
            return 0.5

        buying_ratio = buying_volume / total_volume

        return min(1.0, buying_ratio)

    def _calculate_selling_pressure(self, df: pd.DataFrame) -> float:
        """
        Calculate selling pressure based on price action and volume
        Down days with high volume = selling pressure
        """
        if len(df) < 10:
            return 0.5

        recent = df.tail(10)

        selling_volume = 0
        total_volume = 0

        for idx, row in recent.iterrows():
            volume = row['Volume']
            total_volume += volume

            # If close < open = selling day
            if row['Close'] < row['Open']:
                # Weight by how much it moved down
                strength = (row['Open'] - row['Close']) / row['Open']
                selling_volume += volume * (1 + strength)

        if total_volume == 0:
            return 0.5

        selling_ratio = selling_volume / total_volume

        return min(1.0, selling_ratio)

    def _detect_institutional_activity(self, df: pd.DataFrame) -> float:
        """
        Detect institutional buying/selling via large volume spikes
        Institutions move large blocks = volume spikes
        """
        if len(df) < 20:
            return 0.0

        recent = df.tail(5)
        avg_volume = df['Volume'].tail(20).mean()

        if avg_volume == 0:
            return 0.0

        # Count volume spikes (>2x average)
        spikes = 0
        bullish_spikes = 0

        for idx, row in recent.iterrows():
            if row['Volume'] > avg_volume * 2:
                spikes += 1
                # If spike on up day = bullish institutional activity
                if row['Close'] > row['Open']:
                    bullish_spikes += 1

        if spikes == 0:
            return 0.0

        # Ratio of bullish spikes
        return bullish_spikes / spikes

    def _calculate_accumulation(self, df: pd.DataFrame) -> float:
        """
        Detect accumulation vs distribution
        Accumulation: Price flat/up + increasing volume = smart money buying
        Distribution: Price flat/down + increasing volume = smart money selling
        """
        if len(df) < 20:
            return 0.5

        recent = df.tail(10)
        older = df.iloc[-20:-10]

        # Price change
        recent_price_change = (recent['Close'].iloc[-1] - recent['Close'].iloc[0]) / recent['Close'].iloc[0]

        # Volume change
        recent_volume = recent['Volume'].mean()
        older_volume = older['Volume'].mean()

        if older_volume == 0:
            return 0.5

        volume_increase = recent_volume / older_volume

        # Accumulation: Price up or flat + volume increasing
        if recent_price_change >= 0 and volume_increase > 1.2:
            return min(1.0, volume_increase / 2)

        # Distribution: Price down + volume increasing
        elif recent_price_change < 0 and volume_increase > 1.2:
            return 0.0

        # Neutral
        else:
            return 0.5

    def _calculate_volume_clusters(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Identify price levels with high volume (value areas)
        These act as support/resistance
        """
        if len(df) < 20:
            return {}

        # Create price bins
        price_min = df['Low'].min()
        price_max = df['High'].max()
        bins = 20

        price_range = price_max - price_min
        bin_size = price_range / bins

        volume_at_price = {}

        for idx, row in df.iterrows():
            # Which bin does this price fall into?
            bin_idx = int((row['Close'] - price_min) / bin_size)
            bin_idx = max(0, min(bins - 1, bin_idx))

            bin_price = price_min + (bin_idx * bin_size)

            if bin_price not in volume_at_price:
                volume_at_price[bin_price] = 0

            volume_at_price[bin_price] += row['Volume']

        return volume_at_price
