"""
DIAGNOSTIC MODE - Investigate why generate_predictions_for_web.py produced 0 predictions
DO NOT MODIFY ANY CODE - REPORT RESULTS ONLY
"""

import sys
import traceback
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.turbomode.core_engine.overnight_scanner import ProductionScanner

def diagnostic_check():
    print("="*80)
    print("DIAGNOSTIC CHECK: generate_predictions_for_web.py - 0 Predictions Issue")
    print("="*80)
    print()

    # Initialize scanner
    print("[STEP 0] Initializing ProductionScanner...")
    try:
        scanner = ProductionScanner()
        print("  [OK] Scanner initialized successfully")
    except Exception as e:
        print(f"  [FAIL] Scanner initialization failed: {e}")
        traceback.print_exc()
        return

    print()

    # CHECK 1: Market Data Availability
    print("[CHECK 1] Market Data Availability")
    print("-" * 80)
    test_symbol = "AAPL"
    print(f"  Testing symbol: {test_symbol}")
    print(f"  Calling: scanner.get_ohlcv_dataframe('{test_symbol}', days_back=730)")

    try:
        df = scanner.get_ohlcv_dataframe(test_symbol, days_back=730)

        if df is None:
            print(f"  [RESULT] DataFrame is None - NO MARKET DATA AVAILABLE")
            print(f"  [IMPACT] This would cause all predictions to fail")
            print()
            print("[ROOT CAUSE IDENTIFIED] No market data available in database")
            print("  Likely cause: Markets closed (it's 7:18 AM Wednesday)")
            print("  Expected behavior: Script should work when markets are open or after data ingestion")
            return
        else:
            print(f"  [RESULT] DataFrame returned successfully")
            print(f"  [ROWS] {len(df)} rows")
            print(f"  [COLUMNS] {list(df.columns)}")
            print(f"  [DATE RANGE] {df.index[0]} to {df.index[-1]}")
            print()
    except Exception as e:
        print(f"  [FAIL] Exception occurred: {e}")
        traceback.print_exc()
        return

    # CHECK 2: Feature Extraction Check
    print("[CHECK 2] Feature Extraction Check")
    print("-" * 80)
    print(f"  Calling: scanner.extract_features(df, '{test_symbol}')")

    try:
        features = scanner.extract_features(df, test_symbol)

        if features is None:
            print(f"  [RESULT] features is None - FEATURE EXTRACTION FAILED")
            print(f"  [IMPACT] This would cause all predictions to fail")
            print()
            print("[ROOT CAUSE IDENTIFIED] Feature extraction returning None")
            return
        else:
            print(f"  [RESULT] Feature vector returned successfully")
            print(f"  [LENGTH] {len(features)} features")
            print(f"  [TYPE] {type(features)}")
            print(f"  [SAMPLE] First 5 features: {features[:5]}")
            print()
    except Exception as e:
        print(f"  [FAIL] Exception occurred: {e}")
        traceback.print_exc()
        return

    # CHECK 3: Data Threshold Check
    print("[CHECK 3] Data Threshold Check")
    print("-" * 80)
    min_required = 400
    actual_rows = len(df)
    print(f"  Minimum required rows: {min_required}")
    print(f"  Actual rows available: {actual_rows}")

    if actual_rows < min_required:
        print(f"  [RESULT] INSUFFICIENT DATA - {actual_rows} < {min_required}")
        print(f"  [IMPACT] This would cause all predictions to fail")
        print()
        print("[ROOT CAUSE IDENTIFIED] Insufficient historical data")
        return
    else:
        print(f"  [RESULT] Sufficient data available ({actual_rows} >= {min_required})")
        print()

    # CHECK 4: Model Loading Check
    print("[CHECK 4] Model Loading Check")
    print("-" * 80)
    print(f"  Calling: scanner.get_prediction('{test_symbol}', features)")

    try:
        prediction = scanner.get_prediction(test_symbol, features)

        if prediction is None:
            print(f"  [RESULT] prediction is None - MODEL PREDICTION FAILED")
            print(f"  [IMPACT] This would cause all predictions to fail")
            print()
            print("[ROOT CAUSE IDENTIFIED] Model prediction returning None")
            return
        else:
            print(f"  [RESULT] Prediction returned successfully")
            print(f"  [PREDICTION] {prediction}")
            print()
    except Exception as e:
        print(f"  [FAIL] Exception occurred: {e}")
        traceback.print_exc()
        return

    # CHECK 5: Full Pipeline Test
    print("[CHECK 5] Full Pipeline Test")
    print("-" * 80)
    print(f"  Testing complete prediction flow for {test_symbol}")

    try:
        # This mimics what generate_predictions_for_web.py does
        df = scanner.get_ohlcv_dataframe(test_symbol, days_back=730)
        if df is None or len(df) < 400:
            print(f"  [FAIL] Data check failed")
            return

        features = scanner.extract_features(df, test_symbol)
        if features is None:
            print(f"  [FAIL] Feature extraction failed")
            return

        prediction = scanner.get_prediction(test_symbol, features)
        if prediction is None:
            print(f"  [FAIL] Prediction failed")
            return

        print(f"  [SUCCESS] Full pipeline works for {test_symbol}")
        print(f"  [OUTPUT] {prediction}")
        print()

    except Exception as e:
        print(f"  [FAIL] Exception in full pipeline: {e}")
        traceback.print_exc()
        return

    print("="*80)
    print("[CONCLUSION] All checks passed - pipeline should work")
    print("  If generate_predictions_for_web.py still fails, the issue may be:")
    print("  - Symbol-specific failures (some symbols have no data)")
    print("  - Error handling suppressing exceptions")
    print("  - Different code path in the script vs scanner")
    print("="*80)

if __name__ == "__main__":
    diagnostic_check()
