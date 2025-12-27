"""
Quick test to verify archive generation fix
Tests with single event (2020 COVID crash) and 2 symbols
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from advanced_ml.backtesting.historical_backtest import HistoricalBacktest

print(f"\n{'='*70}")
print(f"ARCHIVE FIX TEST - 2020 COVID Crash Event")
print(f"{'='*70}\n")

# Test configuration
test_symbol = 'AAPL'
event_start = datetime(2020, 2, 20)
event_end = datetime(2020, 3, 23)

# Calculate fetch dates (event range + lookback)
fetch_start = event_start - timedelta(days=365)
fetch_end = event_end + timedelta(days=30)

print(f"Event: 2020 COVID Crash")
print(f"Event dates: {event_start.strftime('%Y-%m-%d')} to {event_end.strftime('%Y-%m-%d')}")
print(f"Fetch dates: {fetch_start.strftime('%Y-%m-%d')} to {fetch_end.strftime('%Y-%m-%d')}")
print()

# Initialize backtest
backtest = HistoricalBacktest()

# Fetch data with custom dates (THE FIX!)
print(f"[1] Fetching historical data for {test_symbol}...")
df = backtest.fetch_historical_data(
    symbol=test_symbol,
    years=1,  # Ignored when start/end provided
    start_date=fetch_start,
    end_date=fetch_end
)

if df is None:
    print(f"  [FAIL] Could not fetch data")
    exit(1)

print(f"  [OK] Fetched {len(df)} days of data")
print(f"  Data range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

# Check if data covers event period (convert to same timezone)
df_start = df.index[0].replace(tzinfo=None)
df_end = df.index[-1].replace(tzinfo=None)
covers_event = (df_start <= event_start) and (df_end >= event_end)
print(f"  Covers event period: {covers_event}")
print()

# Generate samples
print(f"[2] Generating labeled samples...")
samples = backtest.generate_labeled_data(test_symbol, df)

print(f"  Total samples generated: {len(samples)}")

if len(samples) == 0:
    print(f"  [FAIL] No samples generated!")
    exit(1)

# Filter to event window
event_samples = []
for sample in samples:
    sample_date = datetime.strptime(sample['date'], '%Y-%m-%d')
    if event_start <= sample_date <= event_end:
        event_samples.append(sample)

print(f"  Samples in event window: {len(event_samples)}")
print()

if len(event_samples) > 0:
    print(f"[OK] ARCHIVE FIX WORKING!")
    print(f"\nSample breakdown in event window:")

    # Count labels
    labels = {'buy': 0, 'hold': 0, 'sell': 0}
    for sample in event_samples:
        labels[sample['label_name']] += 1

    for label, count in labels.items():
        pct = (count / len(event_samples) * 100) if len(event_samples) > 0 else 0
        print(f"  {label:6s}: {count:4d} ({pct:5.1f}%)")

    print(f"\nFirst event sample:")
    print(f"  Date: {event_samples[0]['date']}")
    print(f"  Label: {event_samples[0]['label_name']}")
    print(f"  Return: {event_samples[0]['return_pct']:.2f}%")
    print(f"  Features: {event_samples[0]['features'].get('feature_count', 0)}")
else:
    print(f"[FAIL] No samples in event window - fix not working")
    exit(1)

print()
print(f"{'='*70}")
print(f"TEST PASSED - Archive generation fix verified!")
print(f"{'='*70}\n")
