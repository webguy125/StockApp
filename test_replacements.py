"""Quick test to verify MOS and CASY are healthy replacements"""
import yfinance as yf
from datetime import datetime, timedelta

symbols_to_test = ['MOS', 'CASY']

for symbol in symbols_to_test:
    print(f"\n Testing {symbol}...")
    ticker = yf.Ticker(symbol)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    hist = ticker.history(start=start_date, end=end_date, interval='1d')

    if hist.empty:
        print(f"  [X] FAILED: No data")
    else:
        avg_volume = hist['Volume'].mean()
        latest_price = hist['Close'].iloc[-1]
        print(f"  [OK] Price: ${latest_price:.2f}, Volume: {avg_volume:,.0f}, Days: {len(hist)}")
