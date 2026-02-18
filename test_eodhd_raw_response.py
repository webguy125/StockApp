"""
Test EODHD raw response to see data structure
"""
import sys
sys.path.insert(0, 'C:\\StockApp\\backend')

from backend.eodhd_client import get_eodhd_client
from datetime import datetime, timedelta
import json

client = get_eodhd_client()

# Test with AAPL on a historical date (6 months ago)
test_date = (datetime.now() - timedelta(days=150)).strftime('%Y-%m-%d')

print(f"Testing EODHD API with AAPL on {test_date}")
print("="*80)

data = client.get_options_data('AAPL', test_date)

if data:
    print(f"\nSymbol: {data.get('symbol')}")
    print(f"Date: {data.get('date')}")
    print(f"Expirations: {data.get('expirations', [])}")
    print(f"Total options: {len(data.get('data', []))}")

    if data.get('data'):
        print(f"\nFirst 3 options:")
        for i, opt in enumerate(data['data'][:3]):
            print(f"\nOption {i+1}:")
            print(json.dumps(opt, indent=2))
else:
    print("No data returned")
