"""
Test feature extraction for archive generation
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from advanced_ml.backtesting.historical_backtest import HistoricalBacktest
from advanced_ml.features.feature_engineer import FeatureEngineer

# Test with 2008 crisis period
test_symbol = 'AAPL'
print(f"\n{'='*70}")
print(f"TEST: Feature Extraction for {test_symbol} (2008 Crisis)")
print(f"{'='*70}\n")

# Initialize
backtest = HistoricalBacktest()
feature_engineer = FeatureEngineer()

# Fetch historical data (1 year of data for 2008 period)
print(f"[1] Fetching historical data for {test_symbol}...")
df = backtest.fetch_historical_data(test_symbol, years=1)

if df is None:
    print(f"[FAIL] Could not fetch data")
    exit(1)

print(f"  [OK] Fetched {len(df)} days of data")
print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
print()

# Try to extract features for the first 100 days
print(f"[2] Testing feature extraction on first 100 days...")

test_data = df.iloc[:100]
features = feature_engineer.extract_features(test_data, symbol=test_symbol)

print(f"\n  Feature extraction result:")
print(f"    feature_count: {features.get('feature_count', 0)}")
print(f"    Keys in features dict: {len(features)}")

if features.get('feature_count', 0) > 0:
    print(f"  [OK] Feature extraction WORKING")
    print(f"  Sample features:")
    for i, (key, value) in enumerate(list(features.items())[:5]):
        print(f"    {key}: {value}")
else:
    print(f"  [FAIL] Feature extraction FAILED - returning 0 features")
    print(f"  Features dict content:")
    for key, value in features.items():
        print(f"    {key}: {value}")

print()

# Try generating samples
print(f"[3] Testing generate_labeled_data...")
samples = backtest.generate_labeled_data(test_symbol, df)

print(f"  Generated {len(samples)} samples")

if len(samples) > 0:
    print(f"  [OK] Sample generation WORKING")
    print(f"  First sample:")
    print(f"    Date: {samples[0]['date']}")
    print(f"    Label: {samples[0]['label_name']}")
    print(f"    Features: {samples[0]['features'].get('feature_count', 0)}")
else:
    print(f"  [FAIL] Sample generation returned 0 samples")
    print(f"  This is why archive has 0 samples!")

print()
print(f"{'='*70}")
print(f"TEST COMPLETE")
print(f"{'='*70}\n")
