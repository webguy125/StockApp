"""
Check EXAS price action to understand the buyout situation
"""

import yfinance as yf
import pandas as pd

# Download EXAS data
ticker = yf.Ticker('EXAS')
df = ticker.history(period="6mo", interval="1d")

print("=" * 70)
print("EXAS PRICE ANALYSIS (Last 6 Months)")
print("=" * 70)
print()

# Last 30 days stats
df_30d = df.tail(30)
print("[LAST 30 DAYS]")
print(f"  High: ${df_30d['High'].max():.2f}")
print(f"  Low: ${df_30d['Low'].min():.2f}")
print(f"  Current: ${df_30d['Close'].iloc[-1]:.2f}")
print(f"  Range: {((df_30d['High'].max() - df_30d['Low'].min()) / df_30d['Low'].min() * 100):.1f}%")
print(f"  Avg Volume: {df_30d['Volume'].mean():,.0f}")
print()

# Last 60 days stats
df_60d = df.tail(60)
print("[LAST 60 DAYS]")
print(f"  High: ${df_60d['High'].max():.2f}")
print(f"  Low: ${df_60d['Low'].min():.2f}")
print(f"  Range: {((df_60d['High'].max() - df_60d['Low'].min()) / df_60d['Low'].min() * 100):.1f}%")
print(f"  Avg Volume: {df_60d['Volume'].mean():,.0f}")
print()

# Full 6-month stats
print("[LAST 6 MONTHS]")
print(f"  High: ${df['High'].max():.2f}")
print(f"  Low: ${df['Low'].min():.2f}")
print(f"  Range: {((df['High'].max() - df['Low'].min()) / df['Low'].min() * 100):.1f}%")
print(f"  Avg Volume: {df['Volume'].mean():,.0f}")
print()

# Recent closes
print("[LAST 10 DAYS CLOSE PRICES]")
for i, (date, row) in enumerate(df.tail(10).iterrows(), 1):
    print(f"  {date.strftime('%Y-%m-%d')}: ${row['Close']:.2f} (vol: {row['Volume']:,.0f})")
print()

print("=" * 70)
print("ANALYSIS:")
print("=" * 70)
print()
print("The stock shows:")
print(f"- 30-day volatility: {((df_30d['High'].max() - df_30d['Low'].min()) / df_30d['Low'].min() * 100):.1f}%")
print(f"- 60-day volatility: {((df_60d['High'].max() - df_60d['Low'].min()) / df_60d['Low'].min() * 100):.1f}%")
print(f"- Average volume: {df_30d['Volume'].mean():,.0f} shares/day")
print()
print("This is NOT flatlined in the traditional sense (dead stock).")
print("It's range-bound due to buyout at $105, but still has normal trading activity.")
print()
print("Our filter should detect:")
print("1. ABSOLUTE price ceiling (buyout limit)")
print("2. Distance to profit target (can it reach +10%?)")
print()
print(f"Current price: ${df_30d['Close'].iloc[-1]:.2f}")
print(f"Buyout cap: $105.00")
print(f"Max possible gain: {((105 - df_30d['Close'].iloc[-1]) / df_30d['Close'].iloc[-1] * 100):.1f}%")
print(f"Profit target: +10%")
print(f"Can reach target? {'NO - Below 10%' if ((105 - df_30d['Close'].iloc[-1]) / df_30d['Close'].iloc[-1] * 100) < 10 else 'YES'}")
