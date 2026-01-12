"""
Phase 3: Inspect Regenerated Training Data
Comprehensive validation of backtest samples with canonical label logic
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from turbomode.core_symbols import get_all_core_symbols

# Database path
db_path = os.path.join(backend_path, "data", "turbomode.db")

print("=" * 80)
print("PHASE 3: TRAINING DATA INSPECTION")
print("=" * 80)
print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Database: {db_path}")
print()

# Load all backtest trades
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("""
    SELECT
        symbol,
        entry_date,
        entry_price,
        exit_date,
        exit_price,
        outcome,
        profit_loss,
        profit_loss_pct,
        trade_type,
        strategy,
        notes
    FROM trades
    WHERE trade_type = 'backtest'
    ORDER BY symbol, entry_date
""", conn)
conn.close()

print("=" * 80)
print("1. BASIC STATISTICS")
print("=" * 80)
print(f"Total samples: {len(df):,}")
print(f"Unique symbols: {df['symbol'].nunique()}")
print(f"Date range: {df['entry_date'].min()} to {df['entry_date'].max()}")
print()

# Expected curated symbols
expected_symbols = set(get_all_core_symbols())
actual_symbols = set(df['symbol'].unique())

print("=" * 80)
print("2. SYMBOL VALIDATION")
print("=" * 80)
print(f"Expected symbols: {len(expected_symbols)}")
print(f"Actual symbols: {len(actual_symbols)}")

if expected_symbols == actual_symbols:
    print("[OK] All symbols match curated list")
else:
    missing = expected_symbols - actual_symbols
    extra = actual_symbols - expected_symbols
    if missing:
        print(f"[WARNING] Missing symbols: {missing}")
    if extra:
        print(f"[WARNING] Extra symbols (contamination): {extra}")

print()

# Samples per symbol
print("=" * 80)
print("3. SAMPLES PER SYMBOL")
print("=" * 80)
samples_per_symbol = df.groupby('symbol').size()
print(f"Mean: {samples_per_symbol.mean():.0f}")
print(f"Median: {samples_per_symbol.median():.0f}")
print(f"Min: {samples_per_symbol.min()}")
print(f"Max: {samples_per_symbol.max()}")
print()

# Show symbols with unusual counts
unusual = samples_per_symbol[
    (samples_per_symbol < samples_per_symbol.median() - 100) |
    (samples_per_symbol > samples_per_symbol.median() + 100)
]
if len(unusual) > 0:
    print("Symbols with unusual sample counts:")
    for symbol, count in unusual.items():
        print(f"  {symbol}: {count}")
else:
    print("[OK] All symbols have consistent sample counts")
print()

# Label distribution
print("=" * 80)
print("4. LABEL DISTRIBUTION (CANONICAL LOGIC)")
print("=" * 80)
label_counts = df['outcome'].value_counts()
total = len(df)
print("Overall distribution:")
for label in ['buy', 'sell', 'hold']:
    count = label_counts.get(label, 0)
    pct = count / total * 100
    print(f"  {label.upper():5s}: {count:7,} ({pct:5.2f}%)")
print()

# Validate canonical thresholds
print("=" * 80)
print("5. CANONICAL LABEL LOGIC VALIDATION")
print("=" * 80)
print("Checking if labels match canonical thresholds (+5%/-5%)...")

# Check BUY labels
buy_samples = df[df['outcome'] == 'buy']
buy_violations = buy_samples[buy_samples['profit_loss_pct'] < 0.05]
print(f"BUY labels: {len(buy_samples):,}")
print(f"  Expected: profit_loss_pct >= 5%")
print(f"  Violations: {len(buy_violations)}")
if len(buy_violations) > 0:
    print(f"  [WARNING] {len(buy_violations)} BUY samples below +5% threshold")
    print(f"  Range: {buy_violations['profit_loss_pct'].min()*100:.2f}% to {buy_violations['profit_loss_pct'].max()*100:.2f}%")
else:
    print("  [OK] All BUY labels meet +5% threshold")

# Check SELL labels
sell_samples = df[df['outcome'] == 'sell']
sell_violations = sell_samples[sell_samples['profit_loss_pct'] > -0.05]
print(f"\nSELL labels: {len(sell_samples):,}")
print(f"  Expected: profit_loss_pct <= -5%")
print(f"  Violations: {len(sell_violations)}")
if len(sell_violations) > 0:
    print(f"  [WARNING] {len(sell_violations)} SELL samples above -5% threshold")
    print(f"  Range: {sell_violations['profit_loss_pct'].min()*100:.2f}% to {sell_violations['profit_loss_pct'].max()*100:.2f}%")
else:
    print("  [OK] All SELL labels meet -5% threshold")

# Check HOLD labels
hold_samples = df[df['outcome'] == 'hold']
hold_violations = hold_samples[
    (hold_samples['profit_loss_pct'] >= 0.05) |
    (hold_samples['profit_loss_pct'] <= -0.05)
]
print(f"\nHOLD labels: {len(hold_samples):,}")
print(f"  Expected: -5% < profit_loss_pct < +5%")
print(f"  Violations: {len(hold_violations)}")
if len(hold_violations) > 0:
    print(f"  [WARNING] {len(hold_violations)} HOLD samples outside -5%/+5% range")
    print(f"  Range: {hold_violations['profit_loss_pct'].min()*100:.2f}% to {hold_violations['profit_loss_pct'].max()*100:.2f}%")
else:
    print("  [OK] All HOLD labels within -5%/+5% range")

print()

# Distribution by symbol
print("=" * 80)
print("6. LABEL DISTRIBUTION BY SYMBOL (TOP 10 VOLATILE)")
print("=" * 80)
symbol_dist = df.groupby(['symbol', 'outcome']).size().unstack(fill_value=0)
symbol_dist['total'] = symbol_dist.sum(axis=1)
symbol_dist['buy_pct'] = symbol_dist['buy'] / symbol_dist['total'] * 100
symbol_dist['sell_pct'] = symbol_dist['sell'] / symbol_dist['total'] * 100
symbol_dist['actionable_pct'] = symbol_dist['buy_pct'] + symbol_dist['sell_pct']
symbol_dist = symbol_dist.sort_values('actionable_pct', ascending=False)

print(f"{'Symbol':<8} {'BUY%':>7} {'SELL%':>7} {'HOLD%':>7} {'Actionable':>10}")
print("-" * 50)
for symbol, row in symbol_dist.head(10).iterrows():
    print(f"{symbol:<8} {row['buy_pct']:>6.1f}% {row['sell_pct']:>6.1f}% "
          f"{row['hold']/row['total']*100:>6.1f}% {row['actionable_pct']:>9.1f}%")
print()

# Date coverage
print("=" * 80)
print("7. DATE COVERAGE ANALYSIS")
print("=" * 80)
df['entry_date'] = pd.to_datetime(df['entry_date'])
df['year'] = df['entry_date'].dt.year

yearly_counts = df.groupby('year').size()
print("Samples per year:")
for year, count in yearly_counts.items():
    pct = count / total * 100
    print(f"  {year}: {count:6,} ({pct:5.2f}%)")
print()

# Check for gaps
print("=" * 80)
print("8. DATA QUALITY CHECKS")
print("=" * 80)

# Null values
null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    print("Null values found:")
    for col, count in null_counts[null_counts > 0].items():
        print(f"  {col}: {count}")
else:
    print("[OK] No null values")

# Duplicate check
duplicates = df.duplicated(subset=['symbol', 'entry_date']).sum()
if duplicates > 0:
    print(f"[WARNING] {duplicates} duplicate entries (symbol + entry_date)")
else:
    print("[OK] No duplicate entries")

# Price sanity check
negative_prices = df[(df['entry_price'] <= 0) | (df['exit_price'] <= 0)]
if len(negative_prices) > 0:
    print(f"[ERROR] {len(negative_prices)} samples with invalid prices")
else:
    print("[OK] All prices are positive")

# Profit/loss consistency
df['calculated_pnl_pct'] = (df['exit_price'] - df['entry_price']) / df['entry_price']
pnl_mismatch = df[abs(df['profit_loss_pct'] - df['calculated_pnl_pct']) > 0.0001]
if len(pnl_mismatch) > 0:
    print(f"[ERROR] {len(pnl_mismatch)} samples with P&L calculation mismatch")
else:
    print("[OK] All P&L calculations are correct")

print()

# Final summary
print("=" * 80)
print("INSPECTION SUMMARY")
print("=" * 80)

total_issues = 0
if expected_symbols != actual_symbols:
    total_issues += 1
if len(buy_violations) > 0:
    total_issues += 1
if len(sell_violations) > 0:
    total_issues += 1
if len(hold_violations) > 0:
    total_issues += 1
if null_counts.sum() > 0:
    total_issues += 1
if duplicates > 0:
    total_issues += 1
if len(negative_prices) > 0:
    total_issues += 1
if len(pnl_mismatch) > 0:
    total_issues += 1

if total_issues == 0:
    print("[SUCCESS] All validation checks passed!")
    print("Training data is ready for model training.")
    exit_code = 0
else:
    print(f"[WARNING] Found {total_issues} issue(s)")
    print("Review warnings above before proceeding to training.")
    exit_code = 1

print()
print("=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)

sys.exit(exit_code)
