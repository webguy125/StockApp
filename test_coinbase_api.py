import requests
from datetime import datetime, timedelta

# Test Coinbase API directly
symbol = "BTC-USD"
granularity = 60  # 1 minute
end_dt = datetime.utcnow()

# Limit to 300 candles max
max_seconds = granularity * 300
max_delta = timedelta(seconds=max_seconds)
start_dt = end_dt - max_delta

url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
params = {
    'granularity': granularity,
    'end': end_dt.isoformat(),
    'start': start_dt.isoformat()
}

print(f"URL: {url}")
print(f"Params: {params}")
print()

try:
    response = requests.get(url, params=params, timeout=10)
    print(f"Status Code: {response.status_code}")
    print(f"Headers: {response.headers}")
    print()

    if response.status_code == 200:
        candles = response.json()
        print(f"Total candles: {len(candles)}")
        if candles:
            print(f"\nFirst candle: {candles[0]}")
            print(f"Last candle: {candles[-1]}")
    else:
        print(f"Error: {response.text}")

except Exception as e:
    print(f"Exception: {e}")
