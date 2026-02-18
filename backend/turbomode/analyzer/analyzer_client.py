"""
Analyzer Client - Thin Client for Fetching Analytics

Provides a thin client to fetch analytics data from THE ANALYZER.

IMPORTANT CONSTRAINTS:
- READ-ONLY client
- Does NOT alter analyzer behavior
- Does NOT modify any core engine
- Safe, local imports for analytics data
"""

from typing import Dict, Optional


def get_trend_analytics(symbol: str, signal_type: str = 'BUY') -> Dict:
    """
    Get trend analytics for a symbol.

    Args:
        symbol: Stock symbol
        signal_type: 'BUY', 'SELL', or 'HOLD' (placeholder for HL)

    Returns:
        Dictionary with trend analytics fields

    Constraints:
        - READ-ONLY
        - Does not modify analyzer behavior
    """
    try:
        from backend.turbomode.analyzer import analyze_trend
        return analyze_trend(symbol, signal_type)
    except Exception as e:
        print(f"[ANALYZER CLIENT] Error fetching trend analytics for {symbol}: {e}")
        return {
            'trend': 'neutral',
            'with_trend': None,
            'trend_strength': 50,
            'trend_commentary': 'Trend data unavailable',
            'alignment': 'N/A'
        }


def get_volume_analytics(symbol: str) -> Dict:
    """
    Get ORD Volume analytics for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary with volume analytics fields

    Constraints:
        - READ-ONLY
        - Does not modify analyzer behavior
    """
    try:
        from backend.turbomode.analyzer import analyze_ord_volume
        return analyze_ord_volume(symbol)
    except Exception as e:
        print(f"[ANALYZER CLIENT] Error fetching volume analytics for {symbol}: {e}")
        return {
            'ord_initial_volume': 0,
            'ord_correction_volume': 0,
            'ord_retest_volume': 0,
            'ord_retest_strength_pct': 0,
            'ord_classification': 'unavailable',
            'ord_commentary': 'Volume data unavailable'
        }


def get_full_analytics(symbol: str, signal_type: str = 'BUY') -> Dict:
    """
    Get complete analytics for a symbol from THE ANALYZER.

    Args:
        symbol: Stock symbol
        signal_type: 'BUY', 'SELL', or 'HOLD' (placeholder for HL)

    Returns:
        Dictionary with all analytics fields

    Constraints:
        - READ-ONLY
        - Does not modify analyzer behavior
    """
    try:
        from backend.turbomode.analyzer import analyze_symbol
        return analyze_symbol(symbol, signal_type)
    except Exception as e:
        print(f"[ANALYZER CLIENT] Error fetching full analytics for {symbol}: {e}")
        return {
            'symbol': symbol,
            'signal_type': signal_type,
            'trend': 'neutral',
            'with_trend': None,
            'trend_strength': 50,
            'trend_commentary': 'Analytics unavailable',
            'alignment': 'N/A',
            'ord_initial_volume': 0,
            'ord_correction_volume': 0,
            'ord_retest_volume': 0,
            'ord_retest_strength_pct': 0,
            'ord_classification': 'unavailable',
            'ord_commentary': 'Analytics unavailable'
        }
