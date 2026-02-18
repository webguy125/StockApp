"""
Test EODHD Subscription Level and API Access
"""
import requests
from backend.eodhd_client import load_api_key

api_key = load_api_key()

print("="*80)
print("EODHD API SUBSCRIPTION TEST")
print("="*80)
print(f"\nAPI Key: {api_key[:10]}...{api_key[-5:]}")
print(f"Key Length: {len(api_key)}")

# Test 1: Basic endpoint (should work for all subscriptions)
print("\n[TEST 1] Testing basic stock data endpoint...")
try:
    response = requests.get(
        "https://eodhd.com/api/eod/AAPL.US",
        params={"api_token": api_key, "fmt": "json"},
        timeout=10
    )
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  [OK] Basic access works: {len(data)} records")
    else:
        print(f"  [FAIL] Error: {response.text[:200]}")
except Exception as e:
    print(f"  [FAIL] Exception: {e}")

# Test 2: Options endpoint (requires premium)
print("\n[TEST 2] Testing options data endpoint...")
try:
    response = requests.get(
        "https://eodhd.com/api/options/AAPL.US",
        params={"api_token": api_key, "fmt": "json"},
        timeout=10
    )
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  [OK] Options access works!")
        print(f"  Data keys: {list(data.keys())}")
        if 'data' in data:
            print(f"  Options count: {len(data['data'])}")
    elif response.status_code == 404:
        print(f"  [FAIL] 404 Not Found - Options data not available")
        print(f"  This might mean:")
        print(f"    - Your subscription doesn't include options data")
        print(f"    - Options endpoint requires 'All-World' or 'All-in-One' plan")
    elif response.status_code == 403:
        print(f"  [FAIL] 403 Forbidden - No access to options data")
        print(f"  Please upgrade your EODHD subscription to include options")
    else:
        print(f"  [FAIL] Error: {response.text[:200]}")
except Exception as e:
    print(f"  [FAIL] Exception: {e}")

# Test 3: User info endpoint (check subscription details)
print("\n[TEST 3] Checking subscription details...")
try:
    response = requests.get(
        "https://eodhd.com/api/user",
        params={"api_token": api_key, "fmt": "json"},
        timeout=10
    )
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"  [OK] Subscription info:")
        for key, value in data.items():
            print(f"    {key}: {value}")
    else:
        print(f"  Response: {response.text[:200]}")
except Exception as e:
    print(f"  Exception: {e}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("\nIf options endpoint returned 404 or 403:")
print("  -> Your EODHD plan might not include historical options data")
print("  -> Required: 'All-World' or 'All-in-One' subscription")
print("  -> Visit: https://eodhd.com/pricing")
print("\nFor now, you can:")
print("  -> Use Tradier for live options data (already working)")
print("  -> Use Yahoo for historical stock prices (backup)")
print("="*80)
