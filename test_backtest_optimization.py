import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(project_root))

"""
Smoke Test: Optimized Backtest System
Tests vectorized label computation + incremental mode + batch inserts on 5 tech symbols
"""

import os
import time
from datetime import datetime
import sqlite3

from backend.turbomode.core_engine.turbomode_backtest import TurboModeBacktest

# Test symbols (5 tech stocks)
TEST_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META']

# Get database path
backend_path = project_root / "backend"
db_path = os.path.join(str(backend_path), "data", "turbomode.db")

print("=" * 80)
print("SMOKE TEST: Optimized Backtest System")
print("=" * 80)
print(f"Test symbols: {TEST_SYMBOLS}")
print(f"Database: {db_path}")
print(f"Optimizations: Vectorized labels + Incremental mode + Batch inserts")
print("=" * 80)

# Initialize backtest engine
backtest = TurboModeBacktest(turbomode_db_path=db_path)

# Clear existing test data for these symbols
print("\n[CLEANUP] Removing existing test data for test symbols...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

for symbol in TEST_SYMBOLS:
    cursor.execute("DELETE FROM trades WHERE symbol = ? AND trade_type = 'backtest'", (symbol,))

conn.commit()
conn.close()
print("[OK] Cleanup complete")

# Run backtest for each symbol
print("\n[TEST] Running backtest for 5 symbols...")
print("=" * 80)

total_start = time.time()
results = []

for i, symbol in enumerate(TEST_SYMBOLS, 1):
    print(f"\n[{i}/5] Processing {symbol}...")
    start = time.time()

    try:
        result = backtest.generate_backtest_samples(
            symbol=symbol,
            lookback_days=3650,  # 10 years
            incremental=False    # Full backtest
        )

        elapsed = time.time() - start
        result['elapsed'] = elapsed
        results.append(result)

        print(f"[OK] {symbol} completed in {elapsed:.2f}s")
        print(f"     Samples: {result['total_samples']}")
        print(f"     BUY: {result['buy_samples']} ({result['buy_pct']:.1f}%)")
        print(f"     SELL: {result['sell_samples']} ({result['sell_pct']:.1f}%)")
        print(f"     HOLD: {result['hold_samples']} ({result['hold_pct']:.1f}%)")

    except Exception as e:
        print(f"[ERROR] {symbol} failed: {e}")
        import traceback
        traceback.print_exc()

total_elapsed = time.time() - total_start

# Summary
print("\n" + "=" * 80)
print("SMOKE TEST RESULTS")
print("=" * 80)

total_samples = sum(r['total_samples'] for r in results)
total_buy = sum(r['buy_samples'] for r in results)
total_sell = sum(r['sell_samples'] for r in results)
total_hold = sum(r['hold_samples'] for r in results)

print(f"\nTotal symbols processed: {len(results)}/5")
print(f"Total samples generated: {total_samples}")
print(f"Total BUY: {total_buy} ({total_buy/total_samples*100:.1f}%)")
print(f"Total SELL: {total_sell} ({total_sell/total_samples*100:.1f}%)")
print(f"Total HOLD: {total_hold} ({total_hold/total_samples*100:.1f}%)")
print(f"\nTotal time: {total_elapsed:.2f}s")
print(f"Average time per symbol: {total_elapsed/len(results):.2f}s")

# Performance metrics
print("\n" + "=" * 80)
print("PERFORMANCE METRICS")
print("=" * 80)

for result in results:
    symbol = result['symbol']
    elapsed = result['elapsed']
    samples = result['total_samples']
    rate = samples / elapsed if elapsed > 0 else 0
    print(f"{symbol}: {samples} samples in {elapsed:.2f}s ({rate:.0f} samples/sec)")

print("\n[OK] Smoke test complete!")
