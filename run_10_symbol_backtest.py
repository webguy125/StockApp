"""
Simple 10-symbol GPU backtest - based on working test_end_to_end.py
This is a simplified version that directly calls the backtest without wrappers
"""
import sys
sys.path.insert(0, 'backend')

from advanced_ml.backtesting.historical_backtest import HistoricalBacktest
from turbomode.sp500_symbols import get_all_symbols

# Setup
db_path = "backend/data/test_10_symbols.db"  # Fresh database
backtest = HistoricalBacktest(db_path, use_gpu=True)

# Get 10 symbols
all_symbols = get_all_symbols()
symbols_to_test = all_symbols[:10]

print(f"\n{'='*80}")
print(f"10-SYMBOL GPU BACKTEST")
print(f"{'='*80}")
print(f"Database: {db_path}")
print(f"Symbols: {symbols_to_test}")
print(f"GPU: Enabled")
print(f"{'='*80}\n")

# Run backtest
results = backtest.run_backtest(symbols_to_test, years=2, save_to_db=True)

# Print results
print(f"\n{'='*80}")
print(f"RESULTS")
print(f"{'='*80}")
print(f"Total samples: {results.get('total_samples', 0)}")
print(f"Symbols processed: {results.get('symbols_processed', 0)}")
print(f"Time: {results.get('time_seconds', 0):.1f}s")
print(f"Avg time per symbol: {results.get('time_seconds', 0) / max(1, results.get('symbols_processed', 1)):.1f}s")
print(f"{'='*80}\n")

# Print label distribution
if results.get('total_samples', 0) > 0:
    print("Label distribution:")
    label_counts = results.get('label_counts', {})
    for label, count in label_counts.items():
        pct = 100 * count / results['total_samples']
        print(f"  {label}: {count} ({pct:.1f}%)")

print("\n✅ GPU backtest complete!")
