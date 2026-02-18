"""
Trend Alignment Engine - Multi-Timeframe Trend Classification

Determines whether each trade is WITH-TREND or COUNTER-TREND using a deterministic
multi-timeframe trend model.

Model:
- Timeframes: 5m, 15m, 1h
- Indicators: EMA20, EMA50, EMA200, EMA50 slope, HH/HL structure
- Output: 'uptrend', 'downtrend', 'neutral'
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Optional, Literal
from datetime import datetime
from master_market_data.market_data_api import get_market_data_api

TrendType = Literal['uptrend', 'downtrend', 'neutral']


def compute_ema(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
    """Compute Exponential Moving Average"""
    return df[column].ewm(span=period, adjust=False).mean()


def compute_slope(series: pd.Series, lookback: int = 5) -> float:
    """Compute slope of a series over the last N periods"""
    if len(series) < lookback + 1:
        return 0.0

    recent = series.iloc[-lookback:].values
    x = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    return float(slope)


def check_higher_highs(df: pd.DataFrame, lookback: int = 10) -> bool:
    """Check if price is making higher highs (uptrend structure)"""
    if len(df) < lookback:
        return False

    highs = df['high'].iloc[-lookback:].values
    # Check if recent high is higher than previous high
    recent_high = highs[-5:].max()
    previous_high = highs[-lookback:-5].max()
    return recent_high > previous_high


def check_lower_lows(df: pd.DataFrame, lookback: int = 10) -> bool:
    """Check if price is making lower lows (downtrend structure)"""
    if len(df) < lookback:
        return False

    lows = df['low'].iloc[-lookback:].values
    # Check if recent low is lower than previous low
    recent_low = lows[-5:].min()
    previous_low = lows[-lookback:-5].min()
    return recent_low < previous_low


def classify_trend_single_timeframe(df: pd.DataFrame) -> TrendType:
    """
    Classify trend for a single timeframe using EMA model

    Uptrend:
    - EMA20 > EMA50
    - EMA50 > EMA200
    - EMA50 slope > 0
    - Higher-high structure

    Downtrend:
    - EMA20 < EMA50
    - EMA50 < EMA200
    - EMA50 slope < 0
    - Lower-low structure

    Neutral:
    - Mixed signals or flat slope
    """
    if len(df) < 200:
        return 'neutral'

    # Compute EMAs
    df['ema20'] = compute_ema(df, 20)
    df['ema50'] = compute_ema(df, 50)
    df['ema200'] = compute_ema(df, 200)

    # Get most recent values
    ema20 = df['ema20'].iloc[-1]
    ema50 = df['ema50'].iloc[-1]
    ema200 = df['ema200'].iloc[-1]

    # Compute EMA50 slope
    slope = compute_slope(df['ema50'], lookback=5)

    # Check structure
    has_higher_highs = check_higher_highs(df, lookback=20)
    has_lower_lows = check_lower_lows(df, lookback=20)

    # Uptrend rules
    uptrend_signals = 0
    if ema20 > ema50:
        uptrend_signals += 1
    if ema50 > ema200:
        uptrend_signals += 1
    if slope > 0:
        uptrend_signals += 1
    if has_higher_highs:
        uptrend_signals += 1

    # Downtrend rules
    downtrend_signals = 0
    if ema20 < ema50:
        downtrend_signals += 1
    if ema50 < ema200:
        downtrend_signals += 1
    if slope < 0:
        downtrend_signals += 1
    if has_lower_lows:
        downtrend_signals += 1

    # Classification: require 3+ signals for strong trend
    if uptrend_signals >= 3:
        return 'uptrend'
    elif downtrend_signals >= 3:
        return 'downtrend'
    else:
        return 'neutral'


def compute_trend(symbol: str, timestamp: Optional[str] = None) -> TrendType:
    """
    Compute multi-timeframe trend classification

    Args:
        symbol: Stock symbol
        timestamp: Optional timestamp for historical trend (default: now)

    Returns:
        'uptrend', 'downtrend', or 'neutral'
    """
    try:
        market_data_api = get_market_data_api()

        # Get data for multiple timeframes
        df_5m = market_data_api.get_candles(symbol, timeframe='5m')
        df_15m = market_data_api.get_candles(symbol, timeframe='15m')
        df_1h = market_data_api.get_candles(symbol, timeframe='1h')

        # Check if data is available
        if df_5m is None or df_15m is None or df_1h is None:
            return 'neutral'

        if len(df_5m) < 200 or len(df_15m) < 200 or len(df_1h) < 200:
            return 'neutral'

        # Classify trend for each timeframe
        trend_5m = classify_trend_single_timeframe(df_5m.copy())
        trend_15m = classify_trend_single_timeframe(df_15m.copy())
        trend_1h = classify_trend_single_timeframe(df_1h.copy())

        # Multi-timeframe consensus (majority vote)
        trends = [trend_5m, trend_15m, trend_1h]

        uptrend_count = trends.count('uptrend')
        downtrend_count = trends.count('downtrend')

        # Require 2+ timeframes to agree for strong trend
        if uptrend_count >= 2:
            return 'uptrend'
        elif downtrend_count >= 2:
            return 'downtrend'
        else:
            return 'neutral'

    except Exception as e:
        print(f"[TREND_ENGINE] Error computing trend for {symbol}: {e}")
        return 'neutral'


def determine_with_trend(signal_type: str, trend: TrendType) -> Optional[int]:
    """
    Determine if trade is with-trend based on signal type and trend

    Args:
        signal_type: 'BUY', 'SELL', or 'HOLD'
        trend: 'uptrend', 'downtrend', or 'neutral'

    Returns:
        1: with-trend
        0: counter-trend
        None: N/A (HOLD signals or neutral trend)
    """
    if signal_type == 'HOLD':
        return None

    if trend == 'neutral':
        return None

    if signal_type == 'BUY':
        return 1 if trend == 'uptrend' else 0

    if signal_type == 'SELL':
        return 1 if trend == 'downtrend' else 0

    return None


if __name__ == '__main__':
    # Test the trend engine
    test_symbols = ['AAPL', 'TSLA', 'MSFT']

    print("Testing Trend Engine")
    print("=" * 60)

    for symbol in test_symbols:
        trend = compute_trend(symbol)
        print(f"{symbol:6s}: {trend}")

        # Test with-trend logic
        for signal_type in ['BUY', 'SELL', 'HOLD']:
            with_trend = determine_with_trend(signal_type, trend)
            status = 'with-trend' if with_trend == 1 else 'counter-trend' if with_trend == 0 else 'N/A'
            print(f"  {signal_type:4s} -> {status}")

    print("=" * 60)
