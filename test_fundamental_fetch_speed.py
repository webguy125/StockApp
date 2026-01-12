"""
Test how much fundamentals will slow down backtesting
"""

import yfinance as yf
import time
import pandas as pd

def test_current_speed(symbol):
    """Current backtest data fetch (what we do now)"""
    start = time.time()

    ticker = yf.Ticker(symbol)
    # Current: Only download price history
    df = ticker.history(period="2y", interval="1d")

    elapsed = time.time() - start
    return elapsed, len(df)


def test_with_fundamentals(symbol):
    """With fundamentals added (proposed)"""
    start = time.time()

    ticker = yf.Ticker(symbol)

    # Download price history (same as before)
    df = ticker.history(period="2y", interval="1d")

    # NEW: Get fundamentals (one-time fetch)
    info = ticker.info

    # Extract the 12 fundamental features
    fundamentals = {
        'beta': info.get('beta', 1.0),
        'shortPercentOfFloat': info.get('shortPercentOfFloat', 0.0),
        'shortRatio': info.get('shortRatio', 0.0),
        'targetMeanPrice': info.get('targetMeanPrice', 0.0),
        'profitMargins': info.get('profitMargins', 0.0),
        'debtToEquity': info.get('debtToEquity', 0.0),
        'priceToBook': info.get('priceToBook', 0.0),
        'priceToSalesTrailing12Months': info.get('priceToSalesTrailing12Months', 0.0),
        'returnOnEquity': info.get('returnOnEquity', 0.0),
        'currentRatio': info.get('currentRatio', 1.0),
        'revenueGrowth': info.get('revenueGrowth', 0.0),
        'forwardPE': info.get('forwardPE', 0.0),
    }

    elapsed = time.time() - start
    return elapsed, len(df), fundamentals


print("=" * 70)
print("BACKTEST PERFORMANCE TEST: Current vs With Fundamentals")
print("=" * 70)

test_symbols = ['AAPL', 'NVDA', 'TSLA', 'JPM', 'XOM', 'TMDX']

print("\n[CURRENT METHOD] Price data only:")
print("-" * 70)

current_times = []
for symbol in test_symbols:
    elapsed, rows = test_current_speed(symbol)
    current_times.append(elapsed)
    print(f"{symbol:6s} - {elapsed:5.2f}s ({rows} rows)")
    time.sleep(0.3)  # Rate limit

avg_current = sum(current_times) / len(current_times)
print(f"\nAverage: {avg_current:.2f}s per stock")

print("\n[NEW METHOD] Price data + fundamentals:")
print("-" * 70)

new_times = []
for symbol in test_symbols:
    elapsed, rows, fundamentals = test_with_fundamentals(symbol)
    new_times.append(elapsed)
    print(f"{symbol:6s} - {elapsed:5.2f}s ({rows} rows, beta={fundamentals['beta']:.2f})")
    time.sleep(0.3)  # Rate limit

avg_new = sum(new_times) / len(new_times)
print(f"\nAverage: {avg_new:.2f}s per stock")

# Calculate slowdown
slowdown_pct = ((avg_new - avg_current) / avg_current) * 100
slowdown_sec = avg_new - avg_current

print("\n" + "=" * 70)
print("IMPACT ANALYSIS")
print("=" * 70)

print(f"\nPer-stock slowdown:")
print(f"  Current:  {avg_current:.2f}s")
print(f"  With fundamentals: {avg_new:.2f}s")
print(f"  Additional time: +{slowdown_sec:.2f}s ({slowdown_pct:.1f}% slower)")

# Backtest scenarios
print(f"\nBacktest time estimates:")
print(f"  80 stocks (current):  {(avg_current * 80) / 60:.1f} minutes")
print(f"  80 stocks (with fundamentals): {(avg_new * 80) / 60:.1f} minutes")
print(f"  Difference: +{((avg_new - avg_current) * 80) / 60:.1f} minutes")

print(f"\n  500 stocks (current):  {(avg_current * 500) / 60:.1f} minutes")
print(f"  500 stocks (with fundamentals): {(avg_new * 500) / 60:.1f} minutes")
print(f"  Difference: +{((avg_new - avg_current) * 500) / 60:.1f} minutes")

print("\n" + "=" * 70)
print("CACHING STRATEGY")
print("=" * 70)

print("\nFundamentals change SLOWLY (quarterly earnings).")
print("We can CACHE them for 24 hours to eliminate the slowdown!")
print()
print("Strategy:")
print("  1. Fetch fundamentals once per day (first run)")
print("  2. Save to cache file (fundamentals_cache.json)")
print("  3. Reuse cached values for all backtests that day")
print("  4. Update cache daily (fundamentals don't change intraday)")
print()
print("Result:")
print(f"  First run: {avg_new:.2f}s per stock (slow)")
print(f"  Subsequent runs: {avg_current:.2f}s per stock (same as now!)")
print(f"  Daily overhead: {(avg_new * 80) / 60:.1f} min to update cache for 80 stocks")
