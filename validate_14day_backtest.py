import sys
sys.path.insert(0, 'C:/StockApp')
from master_market_data.market_data_api import get_market_data_api
import pandas as pd
from datetime import datetime

print('=' * 80)
print('14-DAY BACKTEST VALIDATION TEST')
print('=' * 80)
print(f'Symbol: AAPL')
print(f'Date range: 2024-01-01 to 2024-12-31')
print()

# Fetch data
api = get_market_data_api()
df = api.get_candles('AAPL', timeframe='1d', start_date='2024-01-01', end_date='2024-12-31')

# Convert timestamp index to column
df = df.reset_index()
price_data = df[['timestamp', 'close']].copy()
price_data.columns = ['date', 'close']

print(f'Loaded {len(price_data)} days of price data')
print()

# Manual 14-day backtest (matching turbomode_backtest.py logic)
holding_period = 14
buy_threshold = 0.05
sell_threshold = -0.05

samples = []
for i in range(len(price_data) - holding_period):
    entry_row = price_data.iloc[i]
    exit_row = price_data.iloc[i + holding_period]
    return_pct = (exit_row['close'] - entry_row['close']) / entry_row['close']

    outcome = 'buy' if return_pct >= buy_threshold else ('sell' if return_pct <= sell_threshold else 'hold')

    samples.append({
        'entry_date': str(entry_row['date']).split()[0],
        'exit_date': str(exit_row['date']).split()[0],
        'entry_price': entry_row['close'],
        'exit_price': exit_row['close'],
        'return_pct': return_pct * 100,
        'outcome': outcome
    })

# Display first 3 trades
print('FIRST 3 TRADES:')
print('-' * 80)
for i, sample in enumerate(samples[:3], 1):
    entry_dt = pd.to_datetime(sample['entry_date'])
    exit_dt = pd.to_datetime(sample['exit_date'])
    holding_days = (exit_dt - entry_dt).days

    print(f'Trade #{i}:')
    print(f"  Entry: {sample['entry_date']} @ ${sample['entry_price']:.2f}")
    print(f"  Exit:  {sample['exit_date']} @ ${sample['exit_price']:.2f}")
    print(f'  Holding period: {holding_days} days (target: 14)')
    print(f"  Return: {sample['return_pct']:+.2f}%")
    print(f"  Outcome: {sample['outcome'].upper()}")
    print()

# Calculate statistics
buy_count = sum(1 for s in samples if s['outcome'] == 'buy')
sell_count = sum(1 for s in samples if s['outcome'] == 'sell')
hold_count = sum(1 for s in samples if s['outcome'] == 'hold')
total = len(samples)

returns = [s['return_pct'] for s in samples]

print('STATISTICS:')
print('-' * 80)
print(f'Total samples: {total}')
print(f'BUY:  {buy_count:4d} ({buy_count/total*100:5.1f}%)')
print(f'SELL: {sell_count:4d} ({sell_count/total*100:5.1f}%)')
print(f'HOLD: {hold_count:4d} ({hold_count/total*100:5.1f}%)')
print()
print(f'Return distribution:')
print(f'  Min:  {min(returns):+.2f}%')
print(f'  Max:  {max(returns):+.2f}%')
print(f'  Avg:  {sum(returns)/len(returns):+.2f}%')
print()

# Verify holding period consistency
entry_dates = [pd.to_datetime(s['entry_date']) for s in samples[:5]]
exit_dates = [pd.to_datetime(s['exit_date']) for s in samples[:5]]
holding_periods = [(exit_dates[i] - entry_dates[i]).days for i in range(5)]

print('HOLDING PERIOD VERIFICATION (first 5 trades):')
print('-' * 80)
for i, days in enumerate(holding_periods, 1):
    status = 'OK' if days == 14 else 'MISMATCH'
    print(f'  Trade {i}: {days} days [{status}]')
print()

if all(h == 14 for h in holding_periods):
    print('[OK] All holding periods = 14 days')
else:
    print('[ERROR] Holding period mismatch detected!')
print()
print('=' * 80)
print('VALIDATION COMPLETE')
print('=' * 80)
