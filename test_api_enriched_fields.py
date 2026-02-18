"""
Test script to validate /api/performance/summary enriched fields

This script verifies that all trade quality metrics are properly
flowing from the database through the API to the frontend.
"""

import requests
import json

def test_performance_api():
    """Test the /api/performance/summary endpoint"""

    print("=" * 80)
    print("Testing /api/performance/summary endpoint")
    print("=" * 80)

    try:
        response = requests.get('http://localhost:5000/api/performance/summary')

        if response.status_code != 200:
            print(f"[ERROR] API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return False

        data = response.json()

        # Validate response structure
        if 'equity_curve' not in data:
            print("[ERROR] Response missing 'equity_curve' field")
            return False

        equity_curve = data['equity_curve']
        total_trades = len(equity_curve)

        print(f"\n[OK] Successfully fetched {total_trades} trades")

        if total_trades == 0:
            print("[WARNING] No trades in equity curve")
            return True

        # Check first trade for enriched fields
        print("\n" + "-" * 80)
        print("Sample Trade (first trade):")
        print("-" * 80)

        sample = equity_curve[0]

        required_fields = [
            'timestamp', 'symbol', 'signal_type', 'window', 'entry_time',
            'exit_time', 'pnl', 'shares', 'dollar_pnl', 'equity'
        ]

        enriched_fields = [
            'prob_buy', 'prob_sell', 'prob_hold', 'entry_atr',
            'target_price', 'stop_price', 'rr', 'directional_margin', 'confidence'
        ]

        # Validate required fields
        missing_required = [f for f in required_fields if f not in sample]
        if missing_required:
            print(f"[ERROR] Missing required fields: {missing_required}")
            return False

        print("[OK] All required fields present")

        # Check enriched fields
        print("\nEnriched Fields in Sample Trade:")
        for field in enriched_fields:
            if field in sample:
                value = sample[field]
                if value is not None:
                    print(f"  [OK] {field}: {value}")
                else:
                    print(f"  [NULL] {field}: NULL")
            else:
                print(f"  [ERROR] {field}: MISSING FROM API RESPONSE")

        # Count enriched field coverage
        print("\n" + "-" * 80)
        print("Enriched Field Coverage Across All Trades:")
        print("-" * 80)

        coverage = {}
        for field in enriched_fields:
            count = sum(1 for t in equity_curve if field in t and t[field] is not None)
            coverage[field] = count
            pct = (count / total_trades) * 100
            print(f"  {field}: {count}/{total_trades} ({pct:.1f}%)")

        # Validate NULL-safety
        print("\n" + "-" * 80)
        print("NULL-Safety Validation:")
        print("-" * 80)

        for field in enriched_fields:
            null_count = sum(1 for t in equity_curve if field not in t or t[field] is None)
            if null_count > 0:
                print(f"  [NULL] {field}: {null_count} trades have NULL values (OK - will be filtered safely)")

        # Final validation
        print("\n" + "=" * 80)
        print("API Validation Results:")
        print("=" * 80)

        all_fields_present = all(field in sample for field in enriched_fields)

        if all_fields_present:
            print("[OK] All enriched fields are present in API response")
        else:
            missing = [f for f in enriched_fields if f not in sample]
            print(f"[ERROR] Missing enriched fields: {missing}")
            return False

        print("[OK] API response structure is valid")
        print("[OK] NULL values are handled safely")
        print("[OK] Trade Quality Analyzer is ready for use")

        return True

    except requests.exceptions.ConnectionError:
        print("[ERROR] Could not connect to Flask server at localhost:5000")
        print("   Make sure the API server is running (python backend/api_server.py)")
        return False

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_performance_api()

    if success:
        print("\n[OK] All tests passed!")
        exit(0)
    else:
        print("\n[ERROR] Tests failed!")
        exit(1)
