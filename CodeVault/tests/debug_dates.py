"""
Debug date issue - check if Coinbase is returning 10/23
"""
import sys
sys.path.append('backend')
from coinbase_rest import CoinbaseREST
from datetime import datetime

# Get data
client = CoinbaseREST()
candles = client.get_daily_candles("BTC-USD", days=5)

print("\n=== CHECKING FOR 10/23 ===")
print(f"Today according to Python: {datetime.now().strftime('%Y-%m-%d')}")
print(f"\nLast 5 candles from Coinbase:")
for candle in candles[-5:]:
    print(f"  {candle['Date']} - Close: ${candle['Close']:.2f}")

# Check if 10/23 is in there
today = datetime.now().strftime('%Y-%m-%d')
has_today = any(c['Date'] == today for c in candles)
print(f"\nDoes Coinbase data include today ({today})? {has_today}")

# Check if 10/22 is in there
yesterday = '2025-10-22'
has_yesterday = any(c['Date'] == yesterday for c in candles)
print(f"Does Coinbase data include 10/22? {has_yesterday}")

# What's the last date?
last_date = candles[-1]['Date']
print(f"\nLast date in Coinbase data: {last_date}")
print(f"Should we create a new live candle? {last_date != today}")