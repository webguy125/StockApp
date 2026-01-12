import sys
sys.path.insert(0, 'C:/StockApp')

from backend.turbomode.overnight_scanner import OvernightScanner
import numpy as np

scanner = OvernightScanner()

# Test on 2 different symbols
symbols = ['AAPL', 'TSLA']

for symbol in symbols:
    try:
        # Extract features
        features = scanner.extract_features(symbol)
        if features is None:
            continue

        # Convert to array
        from backend.turbomode.feature_list import FEATURE_LIST
        feature_array = np.array([features.get(f, 0.0) for f in FEATURE_LIST], dtype=np.float32)

        # Get base model predictions
        base_predictions = {}
        base_predictions['xgboost'] = scanner.xgb_model.predict(feature_array)
        base_predictions['xgboost_et'] = scanner.xgb_et_model.predict(feature_array)
        base_predictions['lightgbm'] = scanner.lgbm_model.predict(feature_array)
        base_predictions['catboost'] = scanner.catboost_model.predict(feature_array)
        base_predictions['xgboost_hist'] = scanner.xgb_hist_model.predict(feature_array)
        base_predictions['xgboost_dart'] = scanner.xgb_dart_model.predict(feature_array)
        base_predictions['xgboost_gblinear'] = scanner.xgb_gblinear_model.predict(feature_array)
        base_predictions['xgboost_approx'] = scanner.xgb_approx_model.predict(feature_array)

        print(f'\n{symbol}:')
        print('Base predictions dict keys:', list(base_predictions.keys()))
        print('\nBase predictions:')
        for model_name, probs in base_predictions.items():
            print(f'  {model_name:20s}: {probs}')

        # Get meta-learner prediction
        print('\nCalling meta-learner.predict()...')
        meta_pred = scanner.meta_learner.predict(base_predictions)
        print('Meta-learner output:', meta_pred)

    except Exception as e:
        print(f'{symbol}: ERROR - {e}')
        import traceback
        traceback.print_exc()
