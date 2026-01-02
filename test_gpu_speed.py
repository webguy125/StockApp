"""
Quick test to verify GPU backtest speed
"""

import sys
sys.path.insert(0, 'backend')

from advanced_ml.backtesting.historical_backtest import HistoricalBacktest
import time

print("=" * 80)
print("GPU BACKTEST SPEED TEST")
print("=" * 80)

db_path = "backend/backend/data/test_gpu.db"

# Test 1 symbol with GPU
print("\n[TEST] Running backtest on 1 symbol (AAPL) with GPU...")
print("-" * 80)

start_gpu = time.time()
backtest_gpu = HistoricalBacktest(db_path, use_gpu=True)
results_gpu = backtest_gpu.run_backtest(
    symbols=['AAPL'],
    years=2,
    save_to_db=True
)
gpu_time = time.time() - start_gpu

print(f"\n[RESULT] GPU Time: {gpu_time:.1f} seconds ({gpu_time/60:.2f} minutes)")
print(f"[RESULT] Samples generated: {results_gpu.get('total_samples', 0)}")

# Calculate label distribution
stats = results_gpu.get('stats', {})
total_trades = stats.get('total_trades', 0)
if total_trades > 0:
    buy_pct = stats.get('buy_labels', 0) / total_trades * 100
    hold_pct = stats.get('hold_labels', 0) / total_trades * 100
    sell_pct = stats.get('sell_labels', 0) / total_trades * 100
    print(f"[RESULT] Label Distribution:")
    print(f"         Buy: {stats.get('buy_labels', 0)} ({buy_pct:.1f}%)")
    print(f"         Hold: {stats.get('hold_labels', 0)} ({hold_pct:.1f}%)")
    print(f"         Sell: {stats.get('sell_labels', 0)} ({sell_pct:.1f}%)")

print("\n" + "=" * 80)
print(f"GPU BACKTEST: {gpu_time:.1f}s for 1 symbol")
print(f"ESTIMATED for 510 symbols: {(gpu_time * 510) / 3600:.1f} hours")
print("=" * 80)
