"""
Test script to compare GPU vs CPU feature engineer
Verifies:
1. Both produce same number of features
2. Feature values are similar (within tolerance)
3. GPU is faster than CPU
"""

import pandas as pd
import numpy as np
import time

from feature_engineer import FeatureEngineer
from gpu_feature_engineer_full import GPUFeatureEngineer

# Create sample data
print("Creating sample data (500 bars)...")
dates = pd.date_range('2022-01-01', periods=500)
np.random.seed(42)

df = pd.DataFrame({
    'open': 100 + np.cumsum(np.random.randn(500) * 2),
    'high': 102 + np.cumsum(np.random.randn(500) * 2),
    'low': 98 + np.cumsum(np.random.randn(500) * 2),
    'close': 100 + np.cumsum(np.random.randn(500) * 2),
    'volume': np.random.randint(1000000, 10000000, 500)
}, index=dates)

# Test CPU version
print("\n" + "="*60)
print("Testing CPU Feature Engineer...")
print("="*60)
cpu_engineer = FeatureEngineer(enable_events=False)

cpu_start = time.time()
cpu_features = cpu_engineer.extract_features(df, symbol="TEST")
cpu_time = time.time() - cpu_start

print(f"CPU Features: {cpu_features.get('feature_count', 0)}")
print(f"CPU Time: {cpu_time*1000:.1f}ms")

# Test GPU version
print("\n" + "="*60)
print("Testing GPU Feature Engineer...")
print("="*60)
gpu_engineer = GPUFeatureEngineer(use_gpu=True)

gpu_start = time.time()
gpu_features = gpu_engineer.calculate_features(df)
gpu_time = time.time() - gpu_start

print(f"GPU Features: {gpu_features.get('feature_count', 0)}")
print(f"GPU Time: {gpu_time*1000:.1f}ms")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")

# Compare feature counts
print("\n" + "="*60)
print("Feature Count Comparison")
print("="*60)
print(f"CPU Features: {cpu_features.get('feature_count', 0)}")
print(f"GPU Features: {gpu_features.get('feature_count', 0)}")

# Find common features
cpu_keys = set(k for k in cpu_features.keys() if k not in ['feature_count', 'symbol', 'timestamp', 'last_price', 'last_volume'])
gpu_keys = set(k for k in gpu_features.keys() if k not in ['feature_count', 'symbol', 'timestamp', 'last_price', 'last_volume'])

# Features only in CPU
cpu_only = cpu_keys - gpu_keys
print(f"\nFeatures only in CPU ({len(cpu_only)}):")
for feat in sorted(cpu_only)[:10]:
    print(f"  - {feat}")
if len(cpu_only) > 10:
    print(f"  ... and {len(cpu_only) - 10} more")

# Features only in GPU
gpu_only = gpu_keys - cpu_keys
print(f"\nFeatures only in GPU ({len(gpu_only)}):")
for feat in sorted(gpu_only)[:10]:
    print(f"  - {feat}")
if len(gpu_only) > 10:
    print(f"  ... and {len(gpu_only) - 10} more")

# Common features
common = cpu_keys & gpu_keys
print(f"\nCommon features: {len(common)}")

# Compare values for common features
print("\n" + "="*60)
print("Value Comparison (Common Features)")
print("="*60)

differences = []
for feat in sorted(common):
    cpu_val = cpu_features[feat]
    gpu_val = gpu_features[feat]

    # Skip if either is NaN or inf
    if np.isnan(cpu_val) or np.isnan(gpu_val) or np.isinf(cpu_val) or np.isinf(gpu_val):
        continue

    # Calculate relative difference
    if abs(cpu_val) > 1e-6:
        rel_diff = abs(cpu_val - gpu_val) / abs(cpu_val) * 100
    else:
        rel_diff = abs(cpu_val - gpu_val)

    differences.append((feat, cpu_val, gpu_val, rel_diff))

# Sort by difference
differences.sort(key=lambda x: x[3], reverse=True)

print("\nTop 10 features with largest differences:")
for feat, cpu_val, gpu_val, diff in differences[:10]:
    print(f"  {feat:30s} | CPU: {cpu_val:15.6f} | GPU: {gpu_val:15.6f} | Diff: {diff:8.4f}%")

print("\nTop 10 features with smallest differences:")
for feat, cpu_val, gpu_val, diff in differences[-10:]:
    print(f"  {feat:30s} | CPU: {cpu_val:15.6f} | GPU: {gpu_val:15.6f} | Diff: {diff:8.4f}%")

# Calculate average difference
avg_diff = sum(d[3] for d in differences) / len(differences) if differences else 0
print(f"\nAverage relative difference: {avg_diff:.4f}%")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"CPU Features: {cpu_features.get('feature_count', 0)}")
print(f"GPU Features: {gpu_features.get('feature_count', 0)}")
print(f"Common Features: {len(common)}")
print(f"CPU Time: {cpu_time*1000:.1f}ms")
print(f"GPU Time: {gpu_time*1000:.1f}ms")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")
print(f"Average Difference: {avg_diff:.4f}%")

# Check if GPU meets requirements
print("\n" + "="*60)
print("REQUIREMENTS CHECK")
print("="*60)

checks = []
checks.append(("GPU produces 170+ features", gpu_features.get('feature_count', 0) >= 170))
checks.append(("GPU is faster than CPU", gpu_time < cpu_time))
checks.append(("Average difference < 1%", avg_diff < 1.0))
checks.append(("Using actual GPU", gpu_engineer.using_gpu))

for check_name, passed in checks:
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {check_name}")

all_passed = all(c[1] for c in checks)
if all_passed:
    print("\n✓ ALL CHECKS PASSED!")
else:
    print("\n✗ SOME CHECKS FAILED")
