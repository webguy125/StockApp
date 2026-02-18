"""
THE ANALYZER - Analytics Dashboard for TurboMode

Provides comprehensive analytics, metrics, and insights.
This is a read-only analytics layer that does NOT influence trading decisions.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from typing import Dict, List, Optional
from backend.turbomode.analyzer.trend_analysis import analyze_trend
from backend.turbomode.analyzer.ord_volume_analysis import analyze_ord_volume


def analyze_symbol(symbol: str, signal_type: str) -> Dict:
    """
    Generate comprehensive analytics for a symbol.

    Args:
        symbol: Stock symbol
        signal_type: 'BUY', 'SELL', or 'HOLD'

    Returns:
        Dictionary with analytics fields including trend analytics
    """

    # Get trend analytics (analytics-only, no influence on trading)
    trend_analytics = analyze_trend(symbol, signal_type)

    # Get ORD Volume analytics (analytics-only, no influence on trading)
    ord_analytics = analyze_ord_volume(symbol)

    # Build analytics output
    analytics = {
        'symbol': symbol,
        'signal_type': signal_type,

        # Trend Analytics Fields
        'trend': trend_analytics['trend'],
        'with_trend': trend_analytics['with_trend'],
        'trend_strength': trend_analytics['trend_strength'],
        'trend_commentary': trend_analytics['trend_commentary'],
        'alignment': trend_analytics['alignment'],

        # ORD Volume Analytics Fields
        'ord_initial_volume': ord_analytics['ord_initial_volume'],
        'ord_correction_volume': ord_analytics['ord_correction_volume'],
        'ord_retest_volume': ord_analytics['ord_retest_volume'],
        'ord_retest_strength_pct': ord_analytics['ord_retest_strength_pct'],
        'ord_classification': ord_analytics['ord_classification'],
        'ord_commentary': ord_analytics['ord_commentary'],

        # Additional analytics fields can be added here
        # Example: sentiment, volatility, liquidity, etc.
    }

    return analytics


def analyze_batch(symbols_and_signals: List[tuple]) -> List[Dict]:
    """
    Generate analytics for multiple symbols.

    Args:
        symbols_and_signals: List of (symbol, signal_type) tuples

    Returns:
        List of analytics dictionaries
    """
    results = []

    for symbol, signal_type in symbols_and_signals:
        try:
            analytics = analyze_symbol(symbol, signal_type)
            results.append(analytics)
        except Exception as e:
            print(f"[ANALYZER ERROR] {symbol}: {e}")
            # Return minimal analytics on error
            results.append({
                'symbol': symbol,
                'signal_type': signal_type,
                'trend': 'neutral',
                'with_trend': None,
                'trend_strength': 0,
                'trend_commentary': 'Analytics unavailable',
                'alignment': 'N/A',
                'ord_initial_volume': 0,
                'ord_correction_volume': 0,
                'ord_retest_volume': 0,
                'ord_retest_strength_pct': 0,
                'ord_classification': 'error',
                'ord_commentary': 'ORD Volume analytics unavailable',
                'error': str(e)
            })

    return results


if __name__ == '__main__':
    # Test THE ANALYZER
    test_cases = [
        ('AAPL', 'BUY'),
        ('TSLA', 'SELL'),
        ('MSFT', 'HOLD'),
        ('NVDA', 'BUY')
    ]

    print("THE ANALYZER - Test Run")
    print("=" * 80)

    results = analyze_batch(test_cases)

    for result in results:
        print(f"\n{result['symbol']} {result['signal_type']}:")
        print(f"  Trend: {result['trend']}")
        print(f"  With-Trend: {result['with_trend']}")
        print(f"  Alignment: {result['alignment']}")
        print(f"  Strength: {result['trend_strength']}")
        print(f"  Commentary: {result['trend_commentary']}")
        print(f"\n  ORD Volume Analysis:")
        print(f"    Initial Volume: {result['ord_initial_volume']:,.0f}")
        print(f"    Correction Volume: {result['ord_correction_volume']:,.0f}")
        print(f"    Retest Volume: {result['ord_retest_volume']:,.0f}")
        print(f"    Retest Strength: {result['ord_retest_strength_pct']:.1f}%")
        print(f"    Classification: {result['ord_classification'].upper()}")
        print(f"    Commentary: {result['ord_commentary']}")

    print("\n" + "=" * 80)
