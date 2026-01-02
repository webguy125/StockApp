"""
Create Feature Name Mapping
Maps feature indices (feature_0, feature_1, ...) to actual descriptive names (sma_20, rsi_14, ...)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).resolve().parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from advanced_ml.features.gpu_feature_engineer import GPUFeatureEngineer

print("\n" + "="*70)
print("CREATING FEATURE NAME MAPPING")
print("="*70)

# Initialize GPU feature engineer (with feature selection disabled)
print("\n[1/3] Initializing GPU Feature Engineer...")
feature_engineer = GPUFeatureEngineer(use_gpu=True, use_feature_selection=False)

# Create dummy data to generate one sample
print("\n[2/3] Generating sample features to capture feature names...")
dates = pd.date_range(start='2020-01-01', periods=300, freq='D')
dummy_data = pd.DataFrame({
    'Open': np.random.randn(300).cumsum() + 100,
    'High': np.random.randn(300).cumsum() + 102,
    'Low': np.random.randn(300).cumsum() + 98,
    'Close': np.random.randn(300).cumsum() + 100,
    'Volume': np.random.randint(1000000, 10000000, 300)
}, index=dates)

# Calculate features for one window
features = feature_engineer.calculate_features(dummy_data)

# Remove metadata keys
metadata_keys = ['last_price', 'last_volume', 'feature_count']
feature_names = [k for k in features.keys() if k not in metadata_keys]
feature_names = sorted(feature_names)  # Sort alphabetically for consistency

print(f"[OK] Captured {len(feature_names)} feature names")

# Create mapping: feature_0 -> actual_name
print("\n[3/3] Creating mapping...")
feature_mapping = {}
for idx, name in enumerate(feature_names):
    feature_mapping[f'feature_{idx}'] = name

# Save mapping
output_file = Path(__file__).parent / "feature_name_mapping.json"
with open(output_file, 'w') as f:
    json.dump({
        'total_features': len(feature_names),
        'mapping': feature_mapping,
        'feature_names_in_order': feature_names,
        'created_at': pd.Timestamp.now().isoformat()
    }, f, indent=2)

print(f"[OK] Saved mapping to {output_file}")
print(f"\nMapping sample:")
for i in range(min(10, len(feature_names))):
    print(f"  feature_{i} -> {feature_names[i]}")

print("\n" + "="*70)
print("FEATURE MAPPING COMPLETE!")
print("="*70)
print(f"\nTotal features: {len(feature_names)}")
print(f"Output file: {output_file}")
