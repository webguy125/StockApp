"""
Test Master Market Data API date query behavior
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from master_market_data.market_data_api import get_market_data_api
import pandas as pd

api = get_market_data_api()

# Test exact dates from the backtest (2016-01-11 was shown in warnings)
test_date = '2016-01-11'
symbol = 'AAPL'

print(f"Testing query for {symbol} on {test_date}")
print("=" * 80)

# Query with specific end date
data = api.get_candles(
    symbol=symbol,
    start_date='2016-01-01',
    end_date=test_date,
    timeframe='1d'
)

if data is not None and len(data) > 0:
    data = data.reset_index()

    if 'timestamp' in data.columns:
        print(f"Column name: 'timestamp'")
        data['date'] = data['timestamp']
    elif 'date' in data.columns:
        print(f"Column name: 'date'")

    data['date'] = pd.to_datetime(data['date'])

    print(f"\n[OK] Got {len(data)} candles")
    print(f"First 5 dates:")
    for i, row in data.head(5).iterrows():
        print(f"  {row['date']}")

    print(f"\nLast 5 dates:")
    for i, row in data.tail(5).iterrows():
        print(f"  {row['date']}")

    # Check if exact target date exists
    target_dt = pd.to_datetime(test_date)
    print(f"\nTarget date: {target_dt}")
    print(f"Is target in data? {target_dt in data['date'].values}")

    # Show date range
    print(f"\nDate range in returned data:")
    print(f"  Min: {data['date'].min()}")
    print(f"  Max: {data['date'].max()}")

else:
    print(f"[ERROR] No data returned for {symbol}")
