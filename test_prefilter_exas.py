"""
Test the pre-filter with EXAS to verify it gets filtered out
EXAS should be flagged as flatlined (buyout target at ~$101)
"""

import sys
import os

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from turbomode.overnight_scanner import OvernightScanner

def test_exas_filter():
    """Test that EXAS gets filtered out"""
    print("=" * 70)
    print("TESTING PRE-FILTER WITH EXAS (Buyout Target)")
    print("=" * 70)
    print()

    scanner = OvernightScanner()

    # Test EXAS (should be filtered as flatlined)
    print("[TEST 1] EXAS - Expected: FILTERED (flatlined buyout target)")
    print("-" * 70)
    result = scanner.is_stock_tradeable('EXAS')
    print(f"Result: {'[PASS]' if not result['tradeable'] else '[FAIL]'}")
    print(f"Tradeable: {result['tradeable']}")
    print(f"Reason: {result['reason']}")
    print()

    # Test a known active stock (AAPL) for comparison
    print("[TEST 2] AAPL - Expected: TRADEABLE (active stock)")
    print("-" * 70)
    result = scanner.is_stock_tradeable('AAPL')
    print(f"Result: {'[PASS]' if result['tradeable'] else '[FAIL]'}")
    print(f"Tradeable: {result['tradeable']}")
    print(f"Reason: {result['reason']}")
    print()

    # Test a few more stocks for coverage
    test_stocks = [
        ('NVDA', True, 'High volatility tech stock'),
        ('TSLA', True, 'High volatility EV stock'),
        ('META', True, 'Active tech stock')
    ]

    for symbol, expected_tradeable, description in test_stocks:
        print(f"[TEST] {symbol} - Expected: {'TRADEABLE' if expected_tradeable else 'FILTERED'} ({description})")
        print("-" * 70)
        result = scanner.is_stock_tradeable(symbol)

        is_pass = (result['tradeable'] == expected_tradeable)
        print(f"Result: {'[PASS]' if is_pass else '[FAIL]'}")
        print(f"Tradeable: {result['tradeable']}")
        print(f"Reason: {result['reason']}")
        print()

    print("=" * 70)
    print("PRE-FILTER TEST COMPLETE")
    print("=" * 70)

if __name__ == '__main__':
    test_exas_filter()
