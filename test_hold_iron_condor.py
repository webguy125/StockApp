"""
Test script for HOLD / Iron Condor implementation
Validates that HOLD signals use symmetric bands instead of directional SL/TP
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from backend.turbomode.core_engine.adaptive_sltp import calculate_adaptive_sltp

def test_hold_iron_condor():
    """Test HOLD signal returns symmetric bands"""
    print("[TEST] HOLD / Iron Condor Implementation")
    print("=" * 60)

    # Test parameters
    entry_price = 627.63  # LMT example from session notes
    atr = 5.0
    sector = 'industrials'
    confidence = 0.92
    horizon = '1d'

    # Model probabilities (HOLD condition: BUY and SELL close)
    prob_buy = 0.45
    prob_sell = 0.43
    prob_hold = 0.92

    print(f"\nTest Case: HOLD Signal")
    print(f"Entry Price: ${entry_price}")
    print(f"ATR: {atr}")
    print(f"Sector: {sector}")
    print(f"Confidence: {confidence}")
    print(f"Prob BUY: {prob_buy}, Prob SELL: {prob_sell}, Prob HOLD: {prob_hold}")

    # Calculate adaptive SL/TP for HOLD
    result = calculate_adaptive_sltp(
        entry_price=entry_price,
        atr=atr,
        sector=sector,
        confidence=confidence,
        horizon=horizon,
        position_type='neutral',
        reward_ratio=2.5,
        prob_buy=prob_buy,
        prob_sell=prob_sell,
        prob_hold=prob_hold
    )

    print(f"\nResult:")
    print(f"Position Type: {result.get('position_type')}")
    print(f"Stop Upper: ${result.get('stop_upper'):.2f}" if result.get('stop_upper') else "Stop Upper: None")
    print(f"Stop Lower: ${result.get('stop_lower'):.2f}" if result.get('stop_lower') else "Stop Lower: None")
    print(f"Target Price: {result.get('target_price')}")
    print(f"Stop Price: {result.get('stop_price')}")
    print(f"Target Pct: {result.get('target_pct')}")
    print(f"Stop Pct: {result.get('stop_pct')}")
    print(f"Reward Ratio: {result.get('reward_ratio')}")
    print(f"Sector Multiplier: {result.get('sector_multiplier')}")
    print(f"Confidence Modifier: {result.get('confidence_modifier')}")

    # Validation
    print(f"\n[VALIDATION]")
    checks = {
        'position_type == neutral': result.get('position_type') == 'neutral',
        'stop_upper > entry_price': result.get('stop_upper') and result.get('stop_upper') > entry_price,
        'stop_lower < entry_price': result.get('stop_lower') and result.get('stop_lower') < entry_price,
        'target_price is None': result.get('target_price') is None,
        'stop_price is None': result.get('stop_price') is None,
        'target_pct is None': result.get('target_pct') is None,
        'stop_pct is None': result.get('stop_pct') is None,
        'reward_ratio is None': result.get('reward_ratio') is None,
        'sector_multiplier is None': result.get('sector_multiplier') is None,
        'confidence_modifier is None': result.get('confidence_modifier') is None,
    }

    all_passed = True
    for check, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {check}")
        if not passed:
            all_passed = False

    # Calculate band width for reference
    if result.get('stop_upper') and result.get('stop_lower'):
        band_width = result.get('stop_upper') - result.get('stop_lower')
        band_pct = (band_width / entry_price) * 100
        print(f"\n[INFO] Band Width: ${band_width:.2f} ({band_pct:.2f}%)")

    return all_passed


def test_buy_regression():
    """Test BUY signal still uses directional SL/TP"""
    print("\n" + "=" * 60)
    print("[TEST] BUY Regression (ensure unchanged)")
    print("=" * 60)

    entry_price = 257.10
    atr = 2.45
    sector = 'technology'
    confidence = 0.75
    horizon = '1d'

    print(f"\nTest Case: BUY Signal")
    print(f"Entry Price: ${entry_price}")
    print(f"ATR: {atr}")
    print(f"Sector: {sector}")
    print(f"Confidence: {confidence}")

    result = calculate_adaptive_sltp(
        entry_price=entry_price,
        atr=atr,
        sector=sector,
        confidence=confidence,
        horizon=horizon,
        position_type='long',
        reward_ratio=2.5
    )

    print(f"\nResult:")
    print(f"Stop Price: ${result.get('stop_price'):.2f}")
    print(f"Target Price: ${result.get('target_price'):.2f}")
    print(f"Stop Pct: {result.get('stop_pct')*100:.2f}%")
    print(f"Target Pct: {result.get('target_pct')*100:.2f}%")
    print(f"Sector Multiplier: {result.get('sector_multiplier')}")
    print(f"Confidence Modifier: {result.get('confidence_modifier'):.2f}")

    # Validation
    print(f"\n[VALIDATION]")
    checks = {
        'stop_price < entry_price': result.get('stop_price') < entry_price,
        'target_price > entry_price': result.get('target_price') > entry_price,
        'stop_pct is not None': result.get('stop_pct') is not None,
        'target_pct is not None': result.get('target_pct') is not None,
        'sector_multiplier is not None': result.get('sector_multiplier') is not None,
        'confidence_modifier is not None': result.get('confidence_modifier') is not None,
    }

    all_passed = True
    for check, passed in checks.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {check}")
        if not passed:
            all_passed = False

    return all_passed


if __name__ == '__main__':
    hold_passed = test_hold_iron_condor()
    buy_passed = test_buy_regression()

    print("\n" + "=" * 60)
    print("[SUMMARY]")
    print("=" * 60)
    print(f"HOLD Test: {'[PASS]' if hold_passed else '[FAIL]'}")
    print(f"BUY Regression Test: {'[PASS]' if buy_passed else '[FAIL]'}")
    print(f"Overall: {'[ALL TESTS PASSED]' if (hold_passed and buy_passed) else '[SOME TESTS FAILED]'}")
