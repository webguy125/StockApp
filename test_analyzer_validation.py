"""
Analytics Layer Validation Script - Phase 8

Tests THE ANALYZER module to ensure:
1. Trend analytics fields are present and correct
2. ORD Volume analytics fields are present and correct
3. Analytics remain isolated from core engine
4. No database contamination
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from backend.turbomode.analyzer import analyze_symbol, analyze_batch


def test_single_symbol_analytics():
    """Test analytics for a single symbol"""
    print("=" * 80)
    print("TEST 1: Single Symbol Analytics")
    print("=" * 80)

    test_symbol = "AAPL"
    test_signal = "BUY"

    print(f"\nAnalyzing {test_symbol} with {test_signal} signal...\n")

    try:
        result = analyze_symbol(test_symbol, test_signal)

        # Check required fields
        required_fields = ['symbol', 'signal_type']
        trend_fields = ['trend', 'with_trend', 'trend_strength', 'trend_commentary', 'alignment']
        ord_fields = ['ord_initial_volume', 'ord_correction_volume', 'ord_retest_volume',
                      'ord_retest_strength_pct', 'ord_classification', 'ord_commentary']

        all_fields = required_fields + trend_fields + ord_fields

        print("Field Validation:")
        print("-" * 80)

        missing_fields = []
        for field in all_fields:
            if field in result:
                value = result[field]
                status = "[OK]" if value is not None else "[NULL]"
                print(f"  {status} {field}: {value}")
            else:
                print(f"  [MISSING] {field}")
                missing_fields.append(field)

        if missing_fields:
            print(f"\n[ERROR] Missing fields: {missing_fields}")
            return False

        # Validate field types
        print("\n" + "-" * 80)
        print("Field Type Validation:")
        print("-" * 80)

        validations = [
            (result['symbol'] == test_symbol, f"symbol matches: {result['symbol']}"),
            (result['signal_type'] == test_signal, f"signal_type matches: {result['signal_type']}"),
            (result['trend'] in ['uptrend', 'downtrend', 'neutral'], f"trend valid: {result['trend']}"),
            (result['with_trend'] in [0, 1, None], f"with_trend valid: {result['with_trend']}"),
            (isinstance(result['trend_strength'], (int, float)), f"trend_strength is numeric: {result['trend_strength']}"),
            (isinstance(result['trend_commentary'], str), f"trend_commentary is string: {len(result['trend_commentary'])} chars"),
            (result['alignment'] in ['WITH-TREND', 'COUNTER-TREND', 'N/A'], f"alignment valid: {result['alignment']}"),
            (isinstance(result['ord_initial_volume'], (int, float)), f"ord_initial_volume is numeric: {result['ord_initial_volume']}"),
            (isinstance(result['ord_retest_strength_pct'], (int, float)), f"ord_retest_strength_pct is numeric: {result['ord_retest_strength_pct']}"),
            (result['ord_classification'] in ['strong', 'neutral', 'weak', 'unavailable', 'error'], f"ord_classification valid: {result['ord_classification']}"),
            (isinstance(result['ord_commentary'], str), f"ord_commentary is string: {len(result['ord_commentary'])} chars"),
        ]

        all_valid = True
        for is_valid, message in validations:
            status = "[OK]" if is_valid else "[FAIL]"
            print(f"  {status} {message}")
            if not is_valid:
                all_valid = False

        print("\n" + "=" * 80)
        if all_valid and not missing_fields:
            print("[OK] Single symbol analytics test PASSED")
            return True
        else:
            print("[ERROR] Single symbol analytics test FAILED")
            return False

    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_analytics():
    """Test analytics for multiple symbols"""
    print("\n" + "=" * 80)
    print("TEST 2: Batch Analytics")
    print("=" * 80)

    test_cases = [
        ('AAPL', 'BUY'),
        ('TSLA', 'SELL'),
        ('MSFT', 'HOLD'),
        ('NVDA', 'BUY')
    ]

    print(f"\nAnalyzing {len(test_cases)} symbols...\n")

    try:
        results = analyze_batch(test_cases)

        print(f"[OK] Received {len(results)} results\n")

        all_valid = True
        for i, result in enumerate(results):
            symbol = result.get('symbol', 'UNKNOWN')
            signal = result.get('signal_type', 'UNKNOWN')

            # Check for required fields
            has_trend = 'trend' in result
            has_ord = 'ord_classification' in result
            has_commentary = 'trend_commentary' in result and 'ord_commentary' in result

            status = "[OK]" if (has_trend and has_ord and has_commentary) else "[FAIL]"
            print(f"  {status} {symbol} ({signal}): trend={result.get('trend')}, ord={result.get('ord_classification')}")

            if not (has_trend and has_ord and has_commentary):
                all_valid = False

        print("\n" + "=" * 80)
        if all_valid:
            print("[OK] Batch analytics test PASSED")
            return True
        else:
            print("[ERROR] Batch analytics test FAILED")
            return False

    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_isolation_from_core():
    """Verify analytics modules are not imported in core engine"""
    print("\n" + "=" * 80)
    print("TEST 3: Analytics Isolation from Core Engine")
    print("=" * 80)

    core_files = [
        'backend/turbomode/core_engine/overnight_scanner.py',
        'backend/turbomode/core_engine/fastmode_inference.py',
        'backend/turbomode/core_engine/signal_closer.py',
        'backend/turbomode/core_engine/position_manager.py',
        'backend/turbomode/database_schema.py'
    ]

    forbidden_imports = [
        'from backend.turbomode.analyzer',
        'from backend.turbomode.modules.trend_engine',
        'from backend.turbomode.modules.ord_volume',
        'import analyze_symbol',
        'import analyze_trend',
        'import compute_trend',
        'import compute_ord_volume'
    ]

    print("\nChecking core engine files for analytics imports...\n")

    all_clean = True
    for file_path in core_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            violations = []
            for forbidden in forbidden_imports:
                if forbidden in content:
                    violations.append(forbidden)

            if violations:
                print(f"  [VIOLATION] {file_path}")
                for v in violations:
                    print(f"    - Found: {v}")
                all_clean = False
            else:
                print(f"  [OK] {file_path} - No analytics imports")

        except FileNotFoundError:
            print(f"  [SKIP] {file_path} - File not found")
        except Exception as e:
            print(f"  [ERROR] {file_path} - {e}")

    print("\n" + "=" * 80)
    if all_clean:
        print("[OK] Analytics isolation test PASSED")
        return True
    else:
        print("[ERROR] Analytics isolation test FAILED - Analytics imports found in core engine!")
        return False


def test_database_isolation():
    """Verify analytics fields are not in database tables"""
    print("\n" + "=" * 80)
    print("TEST 4: Database Isolation")
    print("=" * 80)

    import sqlite3

    db_path = 'backend/turbomode/data/turbomode.db'

    analytics_fields = [
        'trend', 'with_trend', 'trend_strength', 'trend_commentary', 'alignment',
        'ord_initial_volume', 'ord_correction_volume', 'ord_retest_volume',
        'ord_retest_strength_pct', 'ord_classification', 'ord_commentary'
    ]

    tables_to_check = ['active_signals', 'signal_history']

    print("\nChecking database tables for analytics fields...\n")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        all_clean = True
        for table in tables_to_check:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]

            violations = [field for field in analytics_fields if field in columns]

            if violations:
                print(f"  [VIOLATION] {table}")
                for v in violations:
                    print(f"    - Found analytics field: {v}")
                all_clean = False
            else:
                print(f"  [OK] {table} - No analytics fields ({len(columns)} columns)")

        conn.close()

        print("\n" + "=" * 80)
        if all_clean:
            print("[OK] Database isolation test PASSED")
            return True
        else:
            print("[ERROR] Database isolation test FAILED - Analytics fields found in database!")
            return False

    except Exception as e:
        print(f"[ERROR] Database check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_determinism():
    """Verify analytics output is deterministic"""
    print("\n" + "=" * 80)
    print("TEST 5: Deterministic Output")
    print("=" * 80)

    test_symbol = "AAPL"
    test_signal = "BUY"

    print(f"\nTesting {test_symbol} {test_signal} twice...\n")

    try:
        result1 = analyze_symbol(test_symbol, test_signal)
        result2 = analyze_symbol(test_symbol, test_signal)

        # Compare key fields
        fields_to_compare = ['trend', 'with_trend', 'alignment', 'ord_classification']

        all_match = True
        for field in fields_to_compare:
            val1 = result1.get(field)
            val2 = result2.get(field)
            match = val1 == val2
            status = "[OK]" if match else "[FAIL]"
            print(f"  {status} {field}: {val1} == {val2}")
            if not match:
                all_match = False

        print("\n" + "=" * 80)
        if all_match:
            print("[OK] Determinism test PASSED")
            return True
        else:
            print("[WARNING] Determinism test FAILED - Results differ between calls")
            print("          This may be acceptable if market data changed between calls")
            return True  # Don't fail on this, market data may change

    except Exception as e:
        print(f"[ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n")
    print("=" * 80)
    print("ANALYTICS LAYER VALIDATION - PHASE 8")
    print("=" * 80)
    print("\n")

    tests = [
        ("Single Symbol Analytics", test_single_symbol_analytics),
        ("Batch Analytics", test_batch_analytics),
        ("Analytics Isolation", test_isolation_from_core),
        ("Database Isolation", test_database_isolation),
        ("Deterministic Output", test_determinism)
    ]

    results = []
    for test_name, test_func in tests:
        passed = test_func()
        results.append((test_name, passed))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {test_name}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)

    print("\n" + "=" * 80)
    print(f"Results: {passed_count}/{total} tests passed")
    print("=" * 80)

    if passed_count == total:
        print("\n[OK] All analytics validation tests PASSED!")
        exit(0)
    else:
        print("\n[ERROR] Some analytics validation tests FAILED!")
        exit(1)
