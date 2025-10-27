"""
Test Coinbase REST API date handling
"""
import sys
sys.path.append('backend')
from coinbase_rest import CoinbaseREST

# Get data
client = CoinbaseREST()
candles = client.get_daily_candles("BTC-USD", days=7)

print("\n=== COINBASE DATA CHECK ===")
print(f"Total candles: {len(candles)}")
print("\nAll dates returned:")
for i, candle in enumerate(candles):
    print(f"  {i}: {candle['Date']} - O:{candle['Open']:.2f} C:{candle['Close']:.2f}")

# Check for duplicates
dates = [c['Date'] for c in candles]
unique_dates = set(dates)
if len(dates) != len(unique_dates):
    print("\n⚠️ DUPLICATE DATES FOUND!")
    from collections import Counter
    counts = Counter(dates)
    for date, count in counts.items():
        if count > 1:
            print(f"  {date} appears {count} times")
else:
    print("\n✅ No duplicate dates")

# Check sorting
sorted_dates = sorted(dates)
if dates == sorted_dates:
    print("✅ Dates are properly sorted")
else:
    print("⚠️ Dates are not sorted correctly")
    print(f"  Original: {dates}")
    print(f"  Should be: {sorted_dates}")