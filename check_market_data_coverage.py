"""
Quick diagnostic: Check Master Market Data date coverage
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from master_market_data.market_data_api import get_market_data_api
import pandas as pd

api = get_market_data_api()

# Test with AAPL
symbol = 'AAPL'
print(f"Checking data coverage for {symbol}...")
print()

# Try to get all available data
try:
    data = api.get_candles(symbol=symbol, timeframe='1d')

    if data is not None and len(data) > 0:
        data = data.reset_index()
        if 'timestamp' in data.columns:
            dates = pd.to_datetime(data['timestamp'])
        else:
            dates = pd.to_datetime(data['date'])

        min_date = dates.min()
        max_date = dates.max()
        count = len(data)

        print(f"[OK] Data available for {symbol}")
        print(f"  Earliest date: {min_date}")
        print(f"  Latest date: {max_date}")
        print(f"  Total days: {count:,}")
        print(f"  Date range: {(max_date - min_date).days} days")
    else:
        print(f"[ERROR] No data found for {symbol}")

except Exception as e:
    print(f"[ERROR] Error fetching data: {e}")

print()
print("Recommendation:")
print("  If earliest date is after 2016-01-01:")
print("    - Need to reduce backtest lookback window")
print("  If earliest date is before 2016-01-01:")
print("    - Data should be available, may be a query issue")
