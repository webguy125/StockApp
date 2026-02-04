import sys
sys.path.insert(0, 'C:/StockApp')
from backend.turbomode.core_engine.turbomode_backtest import TurboModeBacktest
import sqlite3

print('=' * 80)
print('REGENERATING BACKTEST DATA WITH 14-DAY LOGIC')
print('=' * 80)
print()

# Get symbol universe from master_market_data
import os
master_db_path = os.path.join(os.getcwd(), 'master_market_data', 'market_data.db')
conn = sqlite3.connect(master_db_path)
cursor = conn.cursor()
cursor.execute("SELECT DISTINCT symbol FROM candles ORDER BY symbol")
symbols = [row[0] for row in cursor.fetchall()]
conn.close()

print(f'Found {len(symbols)} symbols in universe')
print()

# Initialize backtest engine
backtest = TurboModeBacktest()

# Regenerate for each symbol
total_samples = 0
completed = 0
failed = 0

for i, symbol in enumerate(symbols, 1):
    try:
        print(f'[{i}/{len(symbols)}] Processing {symbol}...', end=' ')

        result = backtest.generate_backtest_samples(
            symbol=symbol,
            lookback_days=3650  # 10 years
        )

        total_samples += result['total_samples']
        completed += 1

        print(f"{result['total_samples']} samples (BUY:{result['buy_samples']}, SELL:{result['sell_samples']}, HOLD:{result['hold_samples']})")

    except Exception as e:
        print(f"ERROR: {e}")
        failed += 1
        break

print()
print('=' * 80)
print('REGENERATION COMPLETE')
print('=' * 80)
print(f'Completed: {completed}/{len(symbols)}')
print(f'Failed: {failed}')
print(f'Total samples generated: {total_samples:,}')
