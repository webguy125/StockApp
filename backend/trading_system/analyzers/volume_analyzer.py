"""
Volume Analyzer
Analyzes volume patterns, spikes, and VWAP
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trading_system.core.base_analyzer import BaseAnalyzer


class VolumeAnalyzer(BaseAnalyzer):
    """
    Analyzes volume patterns for accumulation/distribution signals

    High volume + price increase = Accumulation (bullish)
    High volume + price decrease = Distribution (bearish)
    Low volume = Neutral
    """

    def __init__(self, lookback: int = 20):
        super().__init__(name="volume")
        self.lookback = lookback

    def calculate_vwap(self, df: pd.DataFrame) -> float:
        """Calculate Volume-Weighted Average Price"""
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).sum() / df['Volume'].sum()
        return vwap

    def analyze(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Analyze volume patterns"""
        try:
            # Fetch data
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)

            if df.empty or len(df) < self.lookback + 5:
                return {
                    'signal_strength': 0.5,
                    'direction': 'neutral',
                    'confidence': 0.0,
                    'metadata': {'error': 'Insufficient data'}
                }

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Calculate volume metrics
            avg_volume = df['Volume'].tail(self.lookback).mean()
            current_volume = df['Volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Price movement
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            price_change_pct = ((current_price - prev_price) / prev_price) * 100

            # VWAP
            recent_df = df.tail(self.lookback)
            vwap = self.calculate_vwap(recent_df)
            price_vs_vwap = ((current_price - vwap) / vwap) * 100

            # Determine signal
            if volume_ratio > 1.5:  # Volume spike
                if price_change_pct > 1.0:
                    # High volume + price up = Accumulation
                    signal_strength = 0.7 + min(0.3, price_change_pct / 10)
                    direction = 'bullish'
                    confidence = min(0.9, volume_ratio / 3)
                elif price_change_pct < -1.0:
                    # High volume + price down = Distribution
                    signal_strength = 0.3 - min(0.3, abs(price_change_pct) / 10)
                    direction = 'bearish'
                    confidence = min(0.9, volume_ratio / 3)
                else:
                    # High volume, no clear price direction
                    signal_strength = 0.5
                    direction = 'neutral'
                    confidence = 0.6
            elif current_price > vwap:
                # Price above VWAP (bullish)
                signal_strength = 0.5 + min(0.3, price_vs_vwap / 10)
                direction = 'bullish'
                confidence = 0.6
            elif current_price < vwap:
                # Price below VWAP (bearish)
                signal_strength = 0.5 - min(0.3, abs(price_vs_vwap) / 10)
                direction = 'bearish'
                confidence = 0.6
            else:
                signal_strength = 0.5
                direction = 'neutral'
                confidence = 0.5

            return {
                'signal_strength': float(max(0.0, min(1.0, signal_strength))),
                'direction': direction,
                'confidence': float(max(0.0, min(1.0, confidence))),
                'metadata': {
                    'volume_ratio': float(volume_ratio),
                    'avg_volume': float(avg_volume),
                    'current_volume': float(current_volume),
                    'vwap': float(vwap),
                    'price': float(current_price),
                    'price_vs_vwap_pct': float(price_vs_vwap)
                }
            }

        except Exception as e:
            return {
                'signal_strength': 0.5,
                'direction': 'neutral',
                'confidence': 0.0,
                'metadata': {'error': str(e)}
            }
