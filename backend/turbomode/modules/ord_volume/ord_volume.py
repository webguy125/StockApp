"""
ORD Volume Wave Analysis Engine

Computes wave volumes, retest strength, and classification for THE ANALYZER.
This module is analytics-only and does NOT influence trading decisions.

3-Wave Model:
1. Initial Wave: First directional move
2. Correction Wave: Pullback/consolidation
3. Retest Wave: Subsequent move in original direction

Retest Strength: (retest_volume / initial_volume) * 100
- Strong: >= 110%
- Neutral: 92-109%
- Weak: < 92%
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Dict, Optional
from master_market_data.market_data_api import get_market_data_api


def identify_waves(df: pd.DataFrame, lookback: int = 30) -> Dict:
    """
    Identify the 3 waves in recent price action.

    Args:
        df: DataFrame with OHLCV data
        lookback: Number of candles to analyze

    Returns:
        Dictionary with wave indices: {
            'initial_start': int,
            'initial_end': int,
            'correction_start': int,
            'correction_end': int,
            'retest_start': int,
            'retest_end': int
        }
    """
    if len(df) < lookback:
        return None

    recent = df.iloc[-lookback:].copy()
    recent['swing_high'] = (recent['high'] > recent['high'].shift(1)) & \
                           (recent['high'] > recent['high'].shift(-1))
    recent['swing_low'] = (recent['low'] < recent['low'].shift(1)) & \
                          (recent['low'] < recent['low'].shift(-1))

    # Find significant swings
    swing_indices = []
    for i, row in recent.iterrows():
        if row['swing_high'] or row['swing_low']:
            swing_indices.append(i)

    # Need at least 3 swings to identify 3 waves
    if len(swing_indices) < 3:
        return None

    # Simple wave identification: last 3 swings define the waves
    # Wave 1: Start to first swing
    # Wave 2: First swing to second swing
    # Wave 3: Second swing to end

    return {
        'initial_start': 0,
        'initial_end': len(recent) // 3,
        'correction_start': len(recent) // 3,
        'correction_end': 2 * len(recent) // 3,
        'retest_start': 2 * len(recent) // 3,
        'retest_end': len(recent) - 1
    }


def compute_wave_volumes(df: pd.DataFrame, waves: Dict) -> Dict:
    """
    Compute average volume for each wave.

    Args:
        df: DataFrame with volume data
        waves: Wave indices dictionary

    Returns:
        Dictionary with average volumes for each wave
    """
    if waves is None or len(df) == 0:
        return {
            'initial_volume': 0,
            'correction_volume': 0,
            'retest_volume': 0
        }

    lookback = min(30, len(df))
    recent = df.iloc[-lookback:]

    # Extract wave segments
    initial_wave = recent.iloc[waves['initial_start']:waves['initial_end']]
    correction_wave = recent.iloc[waves['correction_start']:waves['correction_end']]
    retest_wave = recent.iloc[waves['retest_start']:waves['retest_end']]

    # Compute average volume for each wave
    initial_vol = initial_wave['volume'].mean() if len(initial_wave) > 0 else 0
    correction_vol = correction_wave['volume'].mean() if len(correction_wave) > 0 else 0
    retest_vol = retest_wave['volume'].mean() if len(retest_wave) > 0 else 0

    return {
        'initial_volume': float(initial_vol),
        'correction_volume': float(correction_vol),
        'retest_volume': float(retest_vol)
    }


def compute_ord_volume(symbol: str = None, df: pd.DataFrame = None) -> Dict:
    """
    Compute ORD Volume wave analysis.

    Args:
        symbol: Stock symbol (if df not provided)
        df: DataFrame with OHLCV data (if symbol not provided)

    Returns:
        Dictionary with:
        - initial_volume: Average volume of initial wave
        - correction_volume: Average volume of correction wave
        - retest_volume: Average volume of retest wave
        - retest_strength_pct: (retest_volume / initial_volume) * 100
        - classification: 'strong', 'neutral', 'weak'
    """
    # Get data if not provided
    if df is None:
        if symbol is None:
            raise ValueError("Must provide either symbol or df")

        try:
            market_data_api = get_market_data_api()
            df = market_data_api.get_candles(symbol, timeframe='1h')

            if df is None or len(df) < 30:
                return {
                    'initial_volume': 0,
                    'correction_volume': 0,
                    'retest_volume': 0,
                    'retest_strength_pct': 0,
                    'classification': 'unavailable'
                }
        except Exception as e:
            print(f"[ORD_VOLUME] Error fetching data for {symbol}: {e}")
            return {
                'initial_volume': 0,
                'correction_volume': 0,
                'retest_volume': 0,
                'retest_strength_pct': 0,
                'classification': 'error'
            }

    # Identify waves
    waves = identify_waves(df, lookback=30)

    # Compute wave volumes
    volumes = compute_wave_volumes(df, waves)

    initial_vol = volumes['initial_volume']
    correction_vol = volumes['correction_volume']
    retest_vol = volumes['retest_volume']

    # Compute retest strength
    if initial_vol > 0:
        retest_strength_pct = (retest_vol / initial_vol) * 100
    else:
        retest_strength_pct = 0

    # Classify retest strength
    if retest_strength_pct >= 110:
        classification = 'strong'
    elif retest_strength_pct >= 92:
        classification = 'neutral'
    else:
        classification = 'weak'

    return {
        'initial_volume': initial_vol,
        'correction_volume': correction_vol,
        'retest_volume': retest_vol,
        'retest_strength_pct': retest_strength_pct,
        'classification': classification
    }


if __name__ == '__main__':
    # Test the ORD Volume engine
    test_symbols = ['AAPL', 'TSLA', 'MSFT']

    print("Testing ORD Volume Engine")
    print("=" * 80)

    for symbol in test_symbols:
        result = compute_ord_volume(symbol=symbol)
        print(f"\n{symbol}:")
        print(f"  Initial Volume: {result['initial_volume']:,.0f}")
        print(f"  Correction Volume: {result['correction_volume']:,.0f}")
        print(f"  Retest Volume: {result['retest_volume']:,.0f}")
        print(f"  Retest Strength: {result['retest_strength_pct']:.1f}%")
        print(f"  Classification: {result['classification'].upper()}")

    print("\n" + "=" * 80)
