"""
Analyze Top 10 Most Predictable Stocks
Identifies which stocks have the highest win rates with our ML models
"""

import sqlite3
import pandas as pd
from collections import defaultdict

# Connect to database
db_path = "backend/data/advanced_ml_system.db"
conn = sqlite3.connect(db_path)

print("=" * 80)
print("ANALYZING TOP 10 MOST PREDICTABLE STOCKS")
print("=" * 80)
print()

# Query all backtest trades with symbols
query = """
SELECT
    symbol,
    outcome,
    profit_loss_pct,
    entry_date,
    exit_date
FROM trades
WHERE trade_type = 'backtest'
AND outcome IN ('buy', 'sell')
AND symbol IS NOT NULL
ORDER BY symbol, entry_date
"""

df = pd.read_sql_query(query, conn)
conn.close()

if len(df) == 0:
    print("[ERROR] No backtest data found in database")
    print("Make sure you've run generate_backtest_data.py first")
    exit(1)

print(f"Total signals: {len(df):,}")
print(f"Unique symbols: {df['symbol'].nunique()}")
print()

# Analyze per-symbol statistics
symbol_stats = []

for symbol in df['symbol'].unique():
    symbol_df = df[df['symbol'] == symbol]

    total_signals = len(symbol_df)
    buy_signals = len(symbol_df[symbol_df['outcome'] == 'buy'])
    sell_signals = len(symbol_df[symbol_df['outcome'] == 'sell'])

    # Calculate actual win rate (if profit_loss_pct is available)
    # For now, we'll just count signals - actual win rate needs model predictions

    # Signals per year (7 years of data)
    signals_per_year = total_signals / 7

    symbol_stats.append({
        'symbol': symbol,
        'total_signals': total_signals,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'signals_per_year': signals_per_year,
        'buy_ratio': buy_signals / total_signals if total_signals > 0 else 0
    })

# Convert to DataFrame and sort
stats_df = pd.DataFrame(symbol_stats)
stats_df = stats_df.sort_values('total_signals', ascending=False)

print("=" * 80)
print("TOP 10 STOCKS BY SIGNAL FREQUENCY (Most Tradeable)")
print("=" * 80)
print()

top_10 = stats_df.head(10)
for idx, row in top_10.iterrows():
    print(f"{row['symbol']:6s} | Signals: {row['total_signals']:4.0f} | "
          f"Per Year: {row['signals_per_year']:5.1f} | "
          f"BUY: {row['buy_signals']:3.0f} | SELL: {row['sell_signals']:3.0f} | "
          f"BUY%: {row['buy_ratio']*100:5.1f}%")

print()
print("=" * 80)
print("BOTTOM 10 STOCKS BY SIGNAL FREQUENCY (Least Tradeable)")
print("=" * 80)
print()

bottom_10 = stats_df.tail(10)
for idx, row in bottom_10.iterrows():
    print(f"{row['symbol']:6s} | Signals: {row['total_signals']:4.0f} | "
          f"Per Year: {row['signals_per_year']:5.1f} | "
          f"BUY: {row['buy_signals']:3.0f} | SELL: {row['sell_signals']:3.0f} | "
          f"BUY%: {row['buy_ratio']*100:5.1f}%")

print()
print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print()
print(f"Average signals per symbol: {stats_df['total_signals'].mean():.1f}")
print(f"Median signals per symbol: {stats_df['total_signals'].median():.1f}")
print(f"Min signals: {stats_df['total_signals'].min():.0f}")
print(f"Max signals: {stats_df['total_signals'].max():.0f}")
print()
print(f"Average signals per year per symbol: {stats_df['signals_per_year'].mean():.1f}")
print()

# Save top 10 list to file
top_10_symbols = top_10['symbol'].tolist()

print("=" * 80)
print("TOP 10 RECOMMENDED SYMBOLS FOR OPTIONS TRADING")
print("=" * 80)
print()
print("Symbol List (copy to config):")
print(top_10_symbols)
print()

# Save to file
with open("backend/data/top_10_symbols.txt", "w") as f:
    f.write("# Top 10 Most Tradeable Stocks (by signal frequency)\n")
    f.write("# Generated from 7 years of backtest data with 10%/10% thresholds\n")
    f.write("\n")
    for symbol in top_10_symbols:
        f.write(f"{symbol}\n")

print("[OK] Saved to backend/data/top_10_symbols.txt")
print()

print("=" * 80)
print("NOTE: This analysis is based on SIGNAL FREQUENCY (how often ±10% moves occur)")
print()
print("For WIN RATE analysis (which stocks the model predicts best), we need to:")
print("  1. Complete model training")
print("  2. Run backtest with model predictions")
print("  3. Calculate per-symbol accuracy")
print()
print("The most frequent signals + highest model accuracy = BEST stocks to trade!")
print("=" * 80)
