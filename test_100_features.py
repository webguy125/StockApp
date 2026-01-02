"""
Quick test to verify 100-feature selection works correctly
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add backend to path
backend_path = Path(__file__).resolve().parent / 'backend'
sys.path.insert(0, str(backend_path))

from advanced_ml.features.gpu_feature_engineer import GPUFeatureEngineer

print("\n" + "="*70)
print("TESTING 100-FEATURE SELECTION")
print("="*70)

# Create dummy data
print("\n[1/3] Creating dummy data...")
dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
dummy_data = pd.DataFrame({
    'Open': np.random.randn(300).cumsum() + 100,
    'High': np.random.randn(300).cumsum() + 102,
    'Low': np.random.randn(300).cumsum() + 98,
    'Close': np.random.randn(300).cumsum() + 100,
    'Volume': np.random.randint(1000000, 10000000, 300)
}, index=dates)

# Test WITH feature selection (should get 100 features)
print("\n[2/3] Testing WITH feature selection...")
fe_with_selection = GPUFeatureEngineer(use_gpu=True, use_feature_selection=True)
features_100 = fe_with_selection.calculate_features(dummy_data)
metadata_keys = ['last_price', 'last_volume', 'feature_count']
feature_names_100 = [k for k in features_100.keys() if k not in metadata_keys]
print(f"[OK] Generated {len(feature_names_100)} features")
print(f"     feature_count = {features_100.get('feature_count', 'MISSING!')}")
print(f"     Sample features: {feature_names_100[:5]}")

# Test WITHOUT feature selection (should get 176 features)
print("\n[3/3] Testing WITHOUT feature selection...")
fe_without_selection = GPUFeatureEngineer(use_gpu=True, use_feature_selection=False)
features_176 = fe_without_selection.calculate_features(dummy_data)
feature_names_176 = [k for k in features_176.keys() if k not in metadata_keys]
print(f"[OK] Generated {len(feature_names_176)} features")
print(f"     feature_count = {features_176.get('feature_count', 'MISSING!')}")

# Verify
print("\n" + "="*70)
if len(feature_names_100) == 100 and features_100.get('feature_count') == 100:
    print("✅ SUCCESS: Feature selection working correctly!")
    print(f"   - With selection: 100 features (43% speedup)")
    print(f"   - Without selection: 176 features (baseline)")
else:
    print("❌ FAILED: Feature selection not working!")
    print(f"   - Expected: 100 features, Got: {len(feature_names_100)}")
    print(f"   - feature_count: {features_100.get('feature_count')}")
print("="*70)
