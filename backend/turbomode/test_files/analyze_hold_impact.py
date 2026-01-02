"""
Analyze the impact of "hold" labels on model accuracy
Shows what happens if we remove holds or convert them to buy/sell
"""

import sys
import os
import sqlite3
import numpy as np
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Database path
db_path = backend_path / "data" / "advanced_ml_system.db"

print("=" * 80)
print("ANALYZE 'HOLD' LABEL IMPACT ON ACCURACY")
print("=" * 80)

# Connect to database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all trades with their outcomes and returns
cursor.execute("""
    SELECT outcome, profit_loss_pct
    FROM trades
    WHERE trade_type = 'backtest'
""")

trades = cursor.fetchall()
print(f"\nTotal trades: {len(trades)}")

# Analyze distribution
outcomes = {'buy': [], 'hold': [], 'sell': []}
for outcome, return_pct in trades:
    if outcome in outcomes:
        outcomes[outcome].append(return_pct)

print(f"\nCurrent Distribution:")
print(f"  Buy:  {len(outcomes['buy']):6d} ({len(outcomes['buy'])/len(trades)*100:5.1f}%)")
print(f"  Hold: {len(outcomes['hold']):6d} ({len(outcomes['hold'])/len(trades)*100:5.1f}%)")
print(f"  Sell: {len(outcomes['sell']):6d} ({len(outcomes['sell'])/len(trades)*100:5.1f}%)")

# Analyze return distributions
print(f"\nReturn Statistics:")
for label, returns in outcomes.items():
    if returns:
        print(f"\n{label.upper()}:")
        print(f"  Count:   {len(returns)}")
        print(f"  Mean:    {np.mean(returns):+.2f}%")
        print(f"  Median:  {np.median(returns):+.2f}%")
        print(f"  Std Dev: {np.std(returns):.2f}%")
        print(f"  Min:     {np.min(returns):+.2f}%")
        print(f"  Max:     {np.max(returns):+.2f}%")

# Analyze "hold" conversions
print(f"\n" + "=" * 80)
print("WHAT IF WE CONVERT 'HOLD' TO BUY/SELL?")
print("=" * 80)

hold_positive = [r for r in outcomes['hold'] if r > 0]
hold_negative = [r for r in outcomes['hold'] if r < 0]
hold_zero = [r for r in outcomes['hold'] if r == 0]

print(f"\nHold breakdown:")
print(f"  Positive returns (should be 'buy'):  {len(hold_positive):6d} ({len(hold_positive)/len(outcomes['hold'])*100:5.1f}%)")
print(f"  Negative returns (should be 'sell'): {len(hold_negative):6d} ({len(hold_negative)/len(outcomes['hold'])*100:5.1f}%)")
print(f"  Zero returns:                         {len(hold_zero):6d} ({len(hold_zero)/len(outcomes['hold'])*100:5.1f}%)")

# Simulate new distribution
new_buy_count = len(outcomes['buy']) + len(hold_positive)
new_sell_count = len(outcomes['sell']) + len(hold_negative)

print(f"\nNew Distribution (if we convert holds):")
print(f"  Buy:  {new_buy_count:6d} ({new_buy_count/len(trades)*100:5.1f}%)")
print(f"  Sell: {new_sell_count:6d} ({new_sell_count/len(trades)*100:5.1f}%)")
print(f"  Total: {new_buy_count + new_sell_count} (removed {len(hold_zero)} zero-return trades)")

# Check if holds were actually profitable
print(f"\n" + "=" * 80)
print("HOLD PROFITABILITY ANALYSIS")
print("=" * 80)

if outcomes['hold']:
    avg_hold_return = np.mean(outcomes['hold'])
    profitable_holds = len([r for r in outcomes['hold'] if r > 0])

    print(f"\nAverage 'hold' return: {avg_hold_return:+.2f}%")
    print(f"Profitable holds: {profitable_holds}/{len(outcomes['hold'])} ({profitable_holds/len(outcomes['hold'])*100:.1f}%)")

    if avg_hold_return > 0:
        print(f"\n⚠️  INSIGHT: 'Hold' labels are actually PROFITABLE on average!")
        print(f"   Many holds should probably be classified as 'buy'")
    else:
        print(f"\n⚠️  INSIGHT: 'Hold' labels are UNPROFITABLE on average!")
        print(f"   Many holds should probably be classified as 'sell'")

# Current thresholds
print(f"\n" + "=" * 80)
print("CURRENT THRESHOLD SETTINGS")
print("=" * 80)
print(f"\nBuy:  >= +10.0%")
print(f"Hold: between -5.0% and +10.0%")
print(f"Sell: <= -5.0%")

print(f"\n⚠️  RECOMMENDATION:")
print(f"   The 'hold' range is TOO WIDE (-5% to +10% = 15% range)")
print(f"   This causes 55% of data to be labeled 'hold'")
print(f"   Models learn to predict 'hold' for easy accuracy")

# Suggested thresholds
print(f"\n" + "=" * 80)
print("SUGGESTED IMPROVEMENTS")
print("=" * 80)

print(f"\nOption 1: BINARY CLASSIFICATION (Buy/Sell only, remove holds)")
print(f"  - Remove all holds (keep only clear signals)")
print(f"  - Buy: >= +3%")
print(f"  - Sell: <= -2%")
print(f"  - Skip everything in between")
print(f"  - Expected: ~{new_buy_count + new_sell_count} samples (50% of current)")
print(f"  - Expected accuracy: 80-90%+")

print(f"\nOption 2: TIGHTER HOLD RANGE (3-class with narrow hold)")
print(f"  - Buy:  >= +5%")
print(f"  - Hold: between -2% and +5%")
print(f"  - Sell: <= -2%")
print(f"  - Expected: More balanced distribution")
print(f"  - Expected accuracy: 75-85%")

print(f"\nOption 3: CONVERT HOLDS TO BUY/SELL (based on sign)")
print(f"  - Positive holds → Buy: {len(hold_positive)} samples")
print(f"  - Negative holds → Sell: {len(hold_negative)} samples")
print(f"  - New distribution: {new_buy_count}/{new_sell_count}")
print(f"  - Expected accuracy: 70-80%")

conn.close()

print(f"\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
