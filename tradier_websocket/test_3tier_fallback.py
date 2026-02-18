"""
Test 3-Tier Fallback Logic for Scheduler Data Sources
======================================================

This script tests the complete 3-tier fallback chain:
1. Tradier REST API (primary)
2. IB Gateway (secondary)
3. Yahoo Finance (last resort)

Usage:
    python test_3tier_fallback.py

Expected Results:
- Market open: Tradier should provide data
- IBKR offline: Should fall back to Yahoo
- All sources working: Tradier should be used
"""

import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), "..", "backend")
sys.path.insert(0, backend_path)

# Import hybrid data fetcher
from backend.turbomode.core_engine.hybrid_data_fetcher import HybridDataFetcher

def test_tradier_tier():
    """Test Tier 1: Tradier only"""
    print("\n" + "=" * 80)
    print("TEST 1: TIER 1 ONLY (Tradier)")
    print("=" * 80)

    # Create fetcher with only Tradier enabled
    fetcher = HybridDataFetcher(use_ibkr=False, use_tradier=True)

    # Test symbols
    symbols = ['AAPL', 'MSFT', 'TSLA']

    for symbol in symbols:
        print(f"\nFetching {symbol} (5 days)...")
        df = fetcher.fetch_candles(symbol, period='5d', interval='1d')

        if df is not None and not df.empty:
            print(f"  [SUCCESS] Got {len(df)} rows")
            print(f"  Latest: {df.tail(1).to_dict('records')}")
        else:
            print(f"  [FAILED] No data")

    fetcher.disconnect()


def test_ibkr_tier():
    """Test Tier 2: IBKR only"""
    print("\n" + "=" * 80)
    print("TEST 2: TIER 2 ONLY (IBKR)")
    print("=" * 80)

    # Create fetcher with only IBKR enabled
    fetcher = HybridDataFetcher(use_ibkr=True, use_tradier=False)

    # Test symbols
    symbols = ['AAPL', 'MSFT']

    for symbol in symbols:
        print(f"\nFetching {symbol} (5 days)...")
        df = fetcher.fetch_candles(symbol, period='5d', interval='1d')

        if df is not None and not df.empty:
            print(f"  [SUCCESS] Got {len(df)} rows")
            print(f"  Latest: {df.tail(1).to_dict('records')}")
        else:
            print(f"  [FAILED] No data")

    fetcher.disconnect()


def test_yahoo_tier():
    """Test Tier 3: Yahoo only"""
    print("\n" + "=" * 80)
    print("TEST 3: TIER 3 ONLY (Yahoo)")
    print("=" * 80)

    # Create fetcher with only Yahoo enabled
    fetcher = HybridDataFetcher(use_ibkr=False, use_tradier=False)

    # Test symbols
    symbols = ['AAPL', 'MSFT']

    for symbol in symbols:
        print(f"\nFetching {symbol} (5 days)...")
        df = fetcher.fetch_candles(symbol, period='5d', interval='1d')

        if df is not None and not df.empty:
            print(f"  [SUCCESS] Got {len(df)} rows")
            print(f"  Latest: {df.tail(1).to_dict('records')}")
        else:
            print(f"  [FAILED] No data")

    fetcher.disconnect()


def test_full_3tier_fallback():
    """Test complete 3-tier fallback"""
    print("\n" + "=" * 80)
    print("TEST 4: FULL 3-TIER FALLBACK (Tradier -> IBKR -> Yahoo)")
    print("=" * 80)

    # Create fetcher with all tiers enabled
    fetcher = HybridDataFetcher(use_ibkr=True, use_tradier=True)

    # Test symbols
    symbols = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'META']

    for symbol in symbols:
        print(f"\nFetching {symbol} (1 year)...")
        df = fetcher.fetch_candles(symbol, period='1y', interval='1d')

        if df is not None and not df.empty:
            print(f"  [SUCCESS] Got {len(df)} rows")
            print(f"  First date: {df.index[0]}")
            print(f"  Last date: {df.index[-1]}")
            print(f"  Latest close: ${df.iloc[-1]['Close']:.2f}")
        else:
            print(f"  [FAILED] No data")

    fetcher.disconnect()


def test_invalid_symbol():
    """Test fallback with invalid symbol (should fail gracefully)"""
    print("\n" + "=" * 80)
    print("TEST 5: INVALID SYMBOL (Should fail gracefully)")
    print("=" * 80)

    fetcher = HybridDataFetcher(use_ibkr=True, use_tradier=True)

    invalid_symbols = ['INVALID123', 'XXXXXX']

    for symbol in invalid_symbols:
        print(f"\nFetching {symbol}...")
        df = fetcher.fetch_candles(symbol, period='5d', interval='1d')

        if df is not None and not df.empty:
            print(f"  [UNEXPECTED] Got data for invalid symbol!")
        else:
            print(f"  [EXPECTED] Failed as expected (all tiers failed)")

    fetcher.disconnect()


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("3-TIER FALLBACK SYSTEM TEST SUITE")
    print("=" * 80)
    print(f"Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Test each tier individually
        test_tradier_tier()
        test_ibkr_tier()
        test_yahoo_tier()

        # Test complete fallback chain
        test_full_3tier_fallback()

        # Test error handling
        test_invalid_symbol()

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)
    print(f"Test finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    run_all_tests()
