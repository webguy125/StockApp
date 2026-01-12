import sys
sys.path.insert(0, 'C:/StockApp')

from backend.turbomode.overnight_scanner import OvernightScanner
import numpy as np

scanner = OvernightScanner()

# Test on 3 different symbols
symbols = ['AAPL', 'TSLA', 'WMT']

print('Testing base model predictions on 3 different stocks:')
print('='*100)

for symbol in symbols:
    try:
        # Extract features
        features = scanner.extract_features(symbol)
        if features is None:
            print(f'{symbol}: Could not extract features')
            continue

        # Convert to array
        from backend.turbomode.feature_list import FEATURE_LIST
        feature_array = np.array([features.get(f, 0.0) for f in FEATURE_LIST], dtype=np.float32)

        print(f'\n{symbol}:')
        print('  XGBoost:       ', scanner.xgb_model.predict(feature_array))
        print('  XGBoost ET:    ', scanner.xgb_et_model.predict(feature_array))
        print('  LightGBM:      ', scanner.lgbm_model.predict(feature_array))
        print('  CatBoost:      ', scanner.catboost_model.predict(feature_array))

    except Exception as e:
        print(f'{symbol}: ERROR - {e}')
        import traceback
        traceback.print_exc()
