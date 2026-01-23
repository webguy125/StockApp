"""
Unit Tests for Symbol Normalizer

Tests canonical symbol normalization across all providers to ensure
consistent symbol format throughout the platform.

Run:
    python master_market_data/test_symbol_normalizer.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.turbomode.core_engine.symbol_normalizer import (
    is_canonical,
    to_canonical,
    from_canonical,
    auto_correct,
    validate_and_normalize,
    get_provider_formats
)


def test_canonical_validation():
    """Test canonical format validation"""
    print("\n" + "=" * 80)
    print("TEST: Canonical Format Validation")
    print("=" * 80)

    tests = [
        # (symbol, expected_result, description)
        ("AAPL", True, "Simple stock ticker"),
        ("GOOGL", True, "Simple stock ticker"),
        ("BRK-B", True, "Class share with hyphen"),
        ("BF-A", True, "Class share with hyphen"),
        ("BTC-USD", True, "Crypto pair with hyphen"),
        ("ETH-USD", True, "Crypto pair with hyphen"),
        ("BRK.B", False, "Class share with dot (non-canonical)"),
        ("BRK B", False, "Class share with space (non-canonical)"),
        ("BTC/USD", False, "Crypto pair with slash (non-canonical)"),
        ("brk-b", False, "Lowercase (non-canonical)"),
        ("aapl", False, "Lowercase (non-canonical)"),
        ("", False, "Empty string"),
        ("BRK-", False, "Incomplete hyphen format"),
        ("-B", False, "Incomplete hyphen format"),
    ]

    passed = 0
    failed = 0

    for symbol, expected, description in tests:
        result = is_canonical(symbol)
        status = "PASS" if result == expected else "FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"  [{status}] is_canonical('{symbol}') = {result} | {description}")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_schwab_mapping():
    """Test Schwab to Canonical mapping"""
    print("\n" + "=" * 80)
    print("TEST: Schwab to Canonical Mapping")
    print("=" * 80)

    tests = [
        # (schwab_format, expected_canonical, description)
        ("BRK.B", "BRK-B", "Class B share"),
        ("BF.A", "BF-A", "Class A share"),
        ("AAPL", "AAPL", "Regular stock"),
        ("GOOGL", "GOOGL", "Regular stock"),
    ]

    passed = 0
    failed = 0

    for schwab_symbol, expected, description in tests:
        try:
            result = to_canonical(schwab_symbol, provider="schwab")
            status = "PASS" if result == expected else "FAIL"

            if result == expected:
                passed += 1
            else:
                failed += 1

            print(f"  [{status}] '{schwab_symbol}' -> '{result}' (expected '{expected}') | {description}")

        except Exception as e:
            failed += 1
            print(f"  [FAIL] '{schwab_symbol}' raised {type(e).__name__}: {e}")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_ibkr_mapping():
    """Test IBKR to Canonical mapping"""
    print("\n" + "=" * 80)
    print("TEST: IBKR to Canonical Mapping")
    print("=" * 80)

    tests = [
        # (ibkr_format, expected_canonical, description)
        ("BRK B", "BRK-B", "Class B share with space"),
        ("BF A", "BF-A", "Class A share with space"),
        ("BTC/USD", "BTC-USD", "Crypto pair with slash"),
        ("ETH/USD", "ETH-USD", "Crypto pair with slash"),
        ("SOL/USD", "SOL-USD", "Crypto pair with slash"),
        ("AAPL", "AAPL", "Regular stock"),
        ("GOOGL", "GOOGL", "Regular stock"),
    ]

    passed = 0
    failed = 0

    for ibkr_symbol, expected, description in tests:
        try:
            result = to_canonical(ibkr_symbol, provider="ibkr")
            status = "PASS" if result == expected else "FAIL"

            if result == expected:
                passed += 1
            else:
                failed += 1

            print(f"  [{status}] '{ibkr_symbol}' -> '{result}' (expected '{expected}') | {description}")

        except Exception as e:
            failed += 1
            print(f"  [FAIL] '{ibkr_symbol}' raised {type(e).__name__}: {e}")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_tiingo_mapping():
    """Test Tiingo to Canonical mapping"""
    print("\n" + "=" * 80)
    print("TEST: Tiingo to Canonical Mapping")
    print("=" * 80)

    tests = [
        # (tiingo_format, expected_canonical, description)
        ("btcusd", "BTC-USD", "Crypto lowercase"),
        ("ethusd", "ETH-USD", "Crypto lowercase"),
        ("solusd", "SOL-USD", "Crypto lowercase"),
        ("AAPL", "AAPL", "Regular stock uppercase"),
        ("aapl", "AAPL", "Regular stock lowercase"),
    ]

    passed = 0
    failed = 0

    for tiingo_symbol, expected, description in tests:
        try:
            result = to_canonical(tiingo_symbol, provider="tiingo")
            status = "PASS" if result == expected else "FAIL"

            if result == expected:
                passed += 1
            else:
                failed += 1

            print(f"  [{status}] '{tiingo_symbol}' -> '{result}' (expected '{expected}') | {description}")

        except Exception as e:
            failed += 1
            print(f"  [FAIL] '{tiingo_symbol}' raised {type(e).__name__}: {e}")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_auto_correction():
    """Test auto-correction functionality"""
    print("\n" + "=" * 80)
    print("TEST: Auto-Correction")
    print("=" * 80)

    tests = [
        # (input, expected_canonical, should_correct, description)
        ("brk-b", "BRK-B", True, "Lowercase to uppercase"),
        ("BRK.B", "BRK-B", True, "Dot to hyphen"),
        ("BRK B", "BRK-B", True, "Space to hyphen"),
        ("BTC/USD", "BTC-USD", True, "Slash to hyphen"),
        ("AAPL", "AAPL", False, "Already canonical"),
        ("BRK-B", "BRK-B", False, "Already canonical"),
    ]

    passed = 0
    failed = 0

    for input_symbol, expected_canonical, should_correct, description in tests:
        canonical, was_corrected, message = auto_correct(input_symbol)

        # Check canonical result
        canonical_ok = (canonical == expected_canonical)
        # Check correction flag
        correction_ok = (was_corrected == should_correct)

        if canonical_ok and correction_ok:
            passed += 1
            status = "PASS"
        else:
            failed += 1
            status = "FAIL"

        print(f"  [{status}] '{input_symbol}' -> '{canonical}' (corrected={was_corrected}) | {description}")
        print(f"          Message: {message}")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_reverse_mapping():
    """Test Canonical to Provider reverse mapping"""
    print("\n" + "=" * 80)
    print("TEST: Canonical to Provider Reverse Mapping")
    print("=" * 80)

    tests = [
        # (canonical, provider, expected_format, description)
        ("BRK-B", "ibkr", "BRK B", "Class share to IBKR format"),
        ("BRK-B", "schwab", "BRK.B", "Class share to Schwab format"),
        ("BRK-B", "yahoo", "BRK-B", "Class share to Yahoo format"),
        ("BTC-USD", "ibkr", "BTC/USD", "Crypto to IBKR format"),
        ("BTC-USD", "yahoo", "BTC-USD", "Crypto to Yahoo format"),
        ("BTC-USD", "tiingo", "btcusd", "Crypto to Tiingo format"),
        ("AAPL", "ibkr", "AAPL", "Regular stock to IBKR"),
        ("AAPL", "schwab", "AAPL", "Regular stock to Schwab"),
    ]

    passed = 0
    failed = 0

    for canonical, provider, expected, description in tests:
        try:
            result = from_canonical(canonical, provider=provider)
            status = "PASS" if result == expected else "FAIL"

            if result == expected:
                passed += 1
            else:
                failed += 1

            print(f"  [{status}] '{canonical}' -> {provider}:'{result}' (expected '{expected}') | {description}")

        except Exception as e:
            failed += 1
            print(f"  [FAIL] '{canonical}' -> {provider} raised {type(e).__name__}: {e}")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_validate_and_normalize():
    """Test validate_and_normalize function (DB write gatekeeper)"""
    print("\n" + "=" * 80)
    print("TEST: Validate and Normalize (DB Write Gatekeeper)")
    print("=" * 80)

    # Forgiving mode tests
    print("\n  Forgiving Mode (ingestion edges):")
    forgiving_tests = [
        ("BRK.B", "schwab", "BRK-B", True, "Auto-correct Schwab class share"),
        ("BTC/USD", "ibkr", "BTC-USD", True, "Auto-correct IBKR crypto"),
        ("AAPL", "yahoo", "AAPL", True, "Already canonical"),
    ]

    passed = 0
    failed = 0

    for symbol, provider, expected, should_succeed, description in forgiving_tests:
        try:
            result = validate_and_normalize(symbol, provider=provider, strict=False)

            if result == expected:
                passed += 1
                status = "PASS"
            else:
                failed += 1
                status = "FAIL"

            print(f"    [{status}] '{symbol}' ({provider}) -> '{result}' | {description}")

        except Exception as e:
            if should_succeed:
                failed += 1
                print(f"    [FAIL] '{symbol}' raised {type(e).__name__}: {e}")
            else:
                passed += 1
                print(f"    [PASS] '{symbol}' correctly rejected: {e}")

    # Strict mode tests
    print("\n  Strict Mode (core systems):")
    strict_tests = [
        ("BRK-B", "yahoo", "BRK-B", True, "Canonical symbol accepted"),
        ("AAPL", "yahoo", "AAPL", True, "Canonical symbol accepted"),
        ("BRK.B", "yahoo", None, False, "Non-canonical rejected"),
        ("BTC/USD", "yahoo", None, False, "Non-canonical rejected"),
    ]

    for symbol, provider, expected, should_succeed, description in strict_tests:
        try:
            result = validate_and_normalize(symbol, provider=provider, strict=True)

            if should_succeed and result == expected:
                passed += 1
                status = "PASS"
            elif not should_succeed:
                failed += 1
                status = "FAIL"
            else:
                failed += 1
                status = "FAIL"

            print(f"    [{status}] '{symbol}' -> '{result}' | {description}")

        except ValueError as e:
            if not should_succeed:
                passed += 1
                status = "PASS"
                print(f"    [{status}] '{symbol}' correctly rejected | {description}")
            else:
                failed += 1
                print(f"    [FAIL] '{symbol}' wrongly rejected: {e}")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_unknown_symbol_rejection():
    """Test that unknown symbols are rejected"""
    print("\n" + "=" * 80)
    print("TEST: Unknown Symbol Rejection")
    print("=" * 80)

    tests = [
        ("INVALID@SYMBOL", "yahoo", "Symbol with special char"),
        ("123ABC", "yahoo", "Symbol starting with number"),
        ("A", "yahoo", "Single letter (too short for most stocks)"),
        ("XXXXXXXXXXXX", "yahoo", "Overly long symbol"),
    ]

    passed = 0
    failed = 0

    for symbol, provider, description in tests:
        try:
            result = validate_and_normalize(symbol, provider=provider, strict=False)
            failed += 1
            print(f"  [FAIL] '{symbol}' wrongly accepted as '{result}' | {description}")

        except ValueError as e:
            passed += 1
            print(f"  [PASS] '{symbol}' correctly rejected | {description}")

    print(f"\nResults: {passed} passed, {failed} failed")
    return failed == 0


def test_get_provider_formats():
    """Test getting all provider formats for a canonical symbol"""
    print("\n" + "=" * 80)
    print("TEST: Get Provider Formats (all providers)")
    print("=" * 80)

    test_symbols = ["BRK-B", "BTC-USD", "AAPL"]

    all_passed = True

    for symbol in test_symbols:
        print(f"\n  Canonical: {symbol}")
        try:
            formats = get_provider_formats(symbol)

            for provider, format_str in formats.items():
                print(f"    {provider:12s} -> {format_str}")

            # Verify we got all providers
            expected_providers = ["yahoo", "ibkr", "schwab", "polygon", "tiingo", "kraken"]
            for provider in expected_providers:
                if provider not in formats:
                    print(f"    [FAIL] Missing provider: {provider}")
                    all_passed = False

        except Exception as e:
            print(f"    [FAIL] Error: {e}")
            all_passed = False

    status = "PASSED" if all_passed else "FAILED"
    print(f"\nResult: {status}")
    return all_passed


def run_all_tests():
    """Run all test suites"""
    print("\n" + "=" * 80)
    print("SYMBOL NORMALIZER - COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    results = []

    results.append(("Canonical Validation", test_canonical_validation()))
    results.append(("Schwab Mapping", test_schwab_mapping()))
    results.append(("IBKR Mapping", test_ibkr_mapping()))
    results.append(("Tiingo Mapping", test_tiingo_mapping()))
    results.append(("Auto-Correction", test_auto_correction()))
    results.append(("Reverse Mapping", test_reverse_mapping()))
    results.append(("Validate and Normalize", test_validate_and_normalize()))
    results.append(("Unknown Symbol Rejection", test_unknown_symbol_rejection()))
    results.append(("Get Provider Formats", test_get_provider_formats()))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed_count = sum(1 for _, passed in results if passed)
    failed_count = len(results) - passed_count

    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {test_name}")

    print(f"\nTotal: {passed_count}/{len(results)} test suites passed")

    if failed_count == 0:
        print("\n[OK] ALL TESTS PASSED")
        return 0
    else:
        print(f"\n[FAIL] {failed_count} test suite(s) failed")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
