import sys
sys.path.insert(0, 'backend')

from advanced_ml.features.gpu_feature_engineer import GPUFeatureEngineer
from advanced_ml.features.feature_engineer import FeatureEngineer
import pandas as pd
import numpy as np

# Create sample data
df = pd.DataFrame({
    'open': np.random.rand(100)*100,
    'high': np.random.rand(100)*100,
    'low': np.random.rand(100)*100,
    'close': np.random.rand(100)*100,
    'volume': np.random.rand(100)*1000000
})

# Get GPU features
gpu_eng = GPUFeatureEngineer(use_gpu=False)  # Use CPU for comparison
gpu_features = gpu_eng.calculate_features(df)
gpu_keys = set(gpu_features.keys())

# Get CPU features
cpu_eng = FeatureEngineer()
cpu_features = cpu_eng.extract_features(df, symbol='TEST')
cpu_keys = set(cpu_features.keys())

print(f"GPU features: {len(gpu_keys)}")
print(f"CPU features: {len(cpu_keys)}")
print(f"\nMissing from GPU ({len(cpu_keys - gpu_keys)} features):")
for key in sorted(cpu_keys - gpu_keys):
    print(f"  - {key}")

print(f"\nExtra in GPU ({len(gpu_keys - cpu_keys)} features):")
for key in sorted(gpu_keys - cpu_keys):
    print(f"  - {key}")
