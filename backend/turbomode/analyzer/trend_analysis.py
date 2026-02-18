"""
Trend Analysis Module - Analytics Layer for THE ANALYZER

This module provides trend analytics for THE ANALYZER only.
It does NOT influence scanner, inference, SL/TP, or trade execution.

Role: Analytics-only - provides context, commentary, and metadata.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from typing import Dict, Optional
from backend.turbomode.modules.trend_engine import compute_trend, determine_with_trend


def analyze_trend(symbol: str, signal_type: str) -> Dict:
    """
    Compute trend analytics for THE ANALYZER.

    Args:
        symbol: Stock symbol
        signal_type: 'BUY', 'SELL', or 'HOLD'

    Returns:
        Dictionary with:
        - trend: 'uptrend', 'downtrend', or 'neutral'
        - with_trend: 1 (with-trend), 0 (counter-trend), or None (N/A)
        - trend_strength: Integer 0-100 representing trend confidence
        - trend_commentary: Human-readable trend description
    """

    # Compute trend classification
    trend = compute_trend(symbol)

    # Determine if trade is with-trend
    with_trend_flag = determine_with_trend(signal_type, trend)

    # Compute trend strength (simple heuristic based on trend clarity)
    if trend == 'uptrend':
        trend_strength = 75
    elif trend == 'downtrend':
        trend_strength = 75
    else:
        trend_strength = 25

    # Generate commentary
    if with_trend_flag == 1:
        alignment = "WITH-TREND"
        commentary = f"{signal_type} signal aligns with {trend.upper()} (favorable)"
    elif with_trend_flag == 0:
        alignment = "COUNTER-TREND"
        commentary = f"{signal_type} signal opposes {trend.upper()} (caution advised)"
    else:
        alignment = "N/A"
        if signal_type == 'HOLD':
            commentary = "HOLD signals do not require trend alignment"
        else:
            commentary = f"Trend is {trend.upper()} - alignment not applicable"

    return {
        'trend': trend,
        'with_trend': with_trend_flag,
        'trend_strength': trend_strength,
        'trend_commentary': commentary,
        'alignment': alignment
    }


if __name__ == '__main__':
    # Test the trend analysis module
    test_cases = [
        ('AAPL', 'BUY'),
        ('TSLA', 'SELL'),
        ('MSFT', 'HOLD')
    ]

    print("Testing Trend Analysis Module")
    print("=" * 80)

    for symbol, signal_type in test_cases:
        result = analyze_trend(symbol, signal_type)
        print(f"\n{symbol} {signal_type}:")
        print(f"  Trend: {result['trend']}")
        print(f"  With-Trend: {result['with_trend']}")
        print(f"  Alignment: {result['alignment']}")
        print(f"  Strength: {result['trend_strength']}")
        print(f"  Commentary: {result['trend_commentary']}")

    print("\n" + "=" * 80)
