"""
ORD Volume Analysis Module - Analytics Layer for THE ANALYZER

This module provides ORD Volume analytics for THE ANALYZER only.
It does NOT influence scanner, inference, SL/TP, or trade execution.

Role: Analytics-only - provides wave volume context, retest strength, and commentary.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from typing import Dict
from backend.turbomode.modules.ord_volume import compute_ord_volume


def analyze_ord_volume(symbol: str) -> Dict:
    """
    Compute ORD Volume analytics for THE ANALYZER.

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary with:
        - ord_initial_volume: Average volume of initial wave
        - ord_correction_volume: Average volume of correction wave
        - ord_retest_volume: Average volume of retest wave
        - ord_retest_strength_pct: Retest volume as percentage of initial
        - ord_classification: 'strong', 'neutral', 'weak', 'unavailable', 'error'
        - ord_commentary: Human-readable interpretation
    """

    # Compute ORD Volume
    ord_result = compute_ord_volume(symbol=symbol)

    # Extract values
    initial_vol = ord_result['initial_volume']
    correction_vol = ord_result['correction_volume']
    retest_vol = ord_result['retest_volume']
    retest_strength_pct = ord_result['retest_strength_pct']
    classification = ord_result['classification']

    # Generate commentary
    if classification == 'strong':
        commentary = f"Retest volume ({retest_vol:,.0f}) is STRONG at {retest_strength_pct:.1f}% of initial wave ({initial_vol:,.0f}). High conviction continuation likely."
    elif classification == 'neutral':
        commentary = f"Retest volume ({retest_vol:,.0f}) is NEUTRAL at {retest_strength_pct:.1f}% of initial wave ({initial_vol:,.0f}). Moderate conviction."
    elif classification == 'weak':
        commentary = f"Retest volume ({retest_vol:,.0f}) is WEAK at {retest_strength_pct:.1f}% of initial wave ({initial_vol:,.0f}). Low conviction - caution advised."
    elif classification == 'unavailable':
        commentary = "ORD Volume analysis unavailable - insufficient data."
    elif classification == 'error':
        commentary = "ORD Volume analysis error - data fetch failed."
    else:
        commentary = "ORD Volume analysis pending."

    return {
        'ord_initial_volume': initial_vol,
        'ord_correction_volume': correction_vol,
        'ord_retest_volume': retest_vol,
        'ord_retest_strength_pct': retest_strength_pct,
        'ord_classification': classification,
        'ord_commentary': commentary
    }


if __name__ == '__main__':
    # Test the ORD Volume analysis module
    test_symbols = ['AAPL', 'TSLA', 'MSFT']

    print("Testing ORD Volume Analysis Module")
    print("=" * 80)

    for symbol in test_symbols:
        result = analyze_ord_volume(symbol)
        print(f"\n{symbol}:")
        print(f"  Initial Volume: {result['ord_initial_volume']:,.0f}")
        print(f"  Correction Volume: {result['ord_correction_volume']:,.0f}")
        print(f"  Retest Volume: {result['ord_retest_volume']:,.0f}")
        print(f"  Retest Strength: {result['ord_retest_strength_pct']:.1f}%")
        print(f"  Classification: {result['ord_classification'].upper()}")
        print(f"  Commentary: {result['ord_commentary']}")

    print("\n" + "=" * 80)
