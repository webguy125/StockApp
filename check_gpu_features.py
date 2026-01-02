import sys
sys.path.insert(0, 'backend')

from advanced_ml.features.gpu_feature_engineer import GPUFeatureEngineer
import pandas as pd
import numpy as np

eng = GPUFeatureEngineer(use_gpu=True)
df = pd.DataFrame({
    'open': np.random.rand(100)*100,
    'high': np.random.rand(100)*100,
    'low': np.random.rand(100)*100,
    'close': np.random.rand(100)*100,
    'volume': np.random.rand(100)*1000000
})

features = eng.calculate_features(df)

cats = {}
for k in features.keys():
    cat = k.split('_')[0] if '_' in k else 'other'
    cats[cat] = cats.get(cat, 0) + 1

print('Features by category:')
for cat, count in sorted(cats.items()):
    print(f'  {cat}: {count}')

print(f'\nTotal: {len(features)}')
print(f'Missing: {179 - len(features)} features to reach target of 179')
