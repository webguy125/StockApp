import yfinance as yf
import pandas as pd

# Test what yfinance returns for 1m interval with period=1d
symbol = "BTC-USD"
interval = "1m"
period = "1d"

print(f"Fetching {symbol} with interval={interval}, period={period}")
data = yf.download(symbol, interval=interval, period=period)

print(f"\nTotal candles returned: {len(data)}")
print(f"\nFirst 5 candles:")
print(data.head())
print(f"\nLast 5 candles:")
print(data.tail())

print(f"\nPrice range:")
# Handle multi-level column indexing
if isinstance(data.columns, pd.MultiIndex):
    min_low = data[('Low', 'BTC-USD')].min()
    max_high = data[('High', 'BTC-USD')].max()
else:
    min_low = data['Low'].min()
    max_high = data['High'].max()

print(f"  Min Low: ${min_low:.2f}")
print(f"  Max High: ${max_high:.2f}")
print(f"  Range: ${max_high - min_low:.2f}")

print(f"\nDate range:")
print(f"  Start: {data.index[0]}")
print(f"  End: {data.index[-1]}")
