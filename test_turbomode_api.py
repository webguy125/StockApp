"""
Test TurboMode API Endpoints
"""

import requests
import json

API_BASE = 'http://127.0.0.1:5000'

print("\n" + "=" * 70)
print("TURBOMODE API ENDPOINT TESTS")
print("=" * 70)

# Test 1: Get signals
print("\n[TEST 1] GET /turbomode/signals (all)")
try:
    response = requests.get(f'{API_BASE}/turbomode/signals?limit=20')
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Success: {data.get('success')}")
    print(f"Count: {data.get('count')}")
    if data.get('signals'):
        print(f"First signal: {data['signals'][0]['symbol']}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 2: Get BUY signals
print("\n[TEST 2] GET /turbomode/signals (BUY only)")
try:
    response = requests.get(f'{API_BASE}/turbomode/signals?signal_type=BUY&limit=20')
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Success: {data.get('success')}")
    print(f"Count: {data.get('count')}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 3: Get large_cap signals
print("\n[TEST 3] GET /turbomode/signals (large_cap only)")
try:
    response = requests.get(f'{API_BASE}/turbomode/signals?market_cap=large_cap&limit=20')
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Success: {data.get('success')}")
    print(f"Count: {data.get('count')}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 4: Get sectors
print("\n[TEST 4] GET /turbomode/sectors")
try:
    response = requests.get(f'{API_BASE}/turbomode/sectors')
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Success: {data.get('success')}")
    print(f"Bullish sectors: {len(data.get('bullish', []))}")
    print(f"Bearish sectors: {len(data.get('bearish', []))}")
    print(f"Neutral sectors: {len(data.get('neutral', []))}")
except Exception as e:
    print(f"ERROR: {e}")

# Test 5: Get stats
print("\n[TEST 5] GET /turbomode/stats")
try:
    response = requests.get(f'{API_BASE}/turbomode/stats')
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Success: {data.get('success')}")
    if data.get('stats'):
        stats = data['stats']
        print(f"Active signals: {stats.get('active_signals')}")
        print(f"Win rate: {stats.get('win_rate')}%")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 70)
print("[COMPLETE] All API tests finished!")
print("=" * 70)
