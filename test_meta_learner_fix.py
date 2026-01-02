"""
Quick test to re-evaluate meta-learner with fixed evaluation code
"""
import sys
import os
import numpy as np

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from advanced_ml.models.meta_learner import MetaLearner
from advanced_ml.models.xgboost_model import XGBoostModel
from advanced_ml.models.xgboost_rf_model import XGBoostRFModel
from advanced_ml.models.lightgbm_model import LightGBMModel
from advanced_ml.models.xgboost_et_model import XGBoostETModel
from advanced_ml.models.catboost_model import CatBoostModel
from advanced_ml.models.pytorch_nn_model import PyTorchNNModel
from advanced_ml.models.xgboost_linear_model import XGBoostLinearModel
from advanced_ml.models.lstm_model import LSTMModel
import sqlite3

# Load test data from database
db_path = "backend/data/advanced_ml_system.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Load all backtest data
cursor.execute("""
    SELECT * FROM trades WHERE trade_type = 'backtest'
    ORDER BY timestamp
""")

rows = cursor.fetchall()
conn.close()

# Get feature columns (skip non-feature columns)
feature_start_col = 8  # Features start after symbol, timestamp, etc.
n_features = 176

# Split into train/test (80/20)
split_idx = int(len(rows) * 0.8)
test_rows = rows[split_idx:]

print(f"Test samples: {len(test_rows)}")

# Extract features and labels
X_test = np.array([[float(row[i]) if row[i] is not None else 0.0
                    for i in range(feature_start_col, feature_start_col + n_features)]
                   for row in test_rows])
y_test = np.array([0 if row[4] == 'buy' else 1 for row in test_rows])  # Binary: 0=Buy, 1=Sell

# Load models
print("Loading models...")
model_path_base = "backend/data/turbomode_models"

xgb_rf_model = XGBoostRFModel(model_path=f"{model_path_base}/xgboost_rf")
xgb_model = XGBoostModel(model_path=f"{model_path_base}/xgboost")
lgbm_model = LightGBMModel(model_path=f"{model_path_base}/lightgbm")
et_model = XGBoostETModel(model_path=f"{model_path_base}/xgboost_et")
catboost_model = CatBoostModel(model_path=f"{model_path_base}/catboost")
nn_model = PyTorchNNModel(model_path=f"{model_path_base}/pytorch_nn")
linear_model = XGBoostLinearModel(model_path=f"{model_path_base}/xgboost_linear")
svm_model = CatBoostModel(model_path=f"{model_path_base}/catboost_svm")
lstm_model = LSTMModel(model_path=f"{model_path_base}/lstm")

meta_learner = MetaLearner(model_path=f"{model_path_base}/meta_learner")

print("Evaluating meta-learner on test set...")
meta_correct = 0
meta_total = len(X_test)

for i in range(meta_total):
    # Convert to feature dict
    feat_dict = {f'feature_{j}': float(X_test[i, j]) for j in range(X_test.shape[1])}

    # Get base model predictions
    base_preds = {
        'xgboost_rf': xgb_rf_model.predict(feat_dict),
        'xgboost': xgb_model.predict(feat_dict),
        'lightgbm': lgbm_model.predict(feat_dict),
        'xgboost_et': et_model.predict(feat_dict),
        'catboost': catboost_model.predict(feat_dict),
        'pytorch_nn': nn_model.predict(feat_dict),
        'xgboost_linear': linear_model.predict(feat_dict),
        'catboost_svm': svm_model.predict(feat_dict),
        'lstm': lstm_model.predict(feat_dict)
    }

    # Get meta-learner prediction
    ensemble_pred = meta_learner.predict(base_preds)

    # Map prediction to class (BINARY: 0=Buy, 1=Sell) - FIXED!
    pred_map = {'buy': 0, 'sell': 1}
    pred_class = pred_map[ensemble_pred['prediction']]

    if pred_class == y_test[i]:
        meta_correct += 1

    # Progress indicator
    if (i + 1) % 1000 == 0:
        print(f"Progress: {i+1}/{meta_total} ({(i+1)/meta_total*100:.1f}%)")

meta_test_accuracy = meta_correct / meta_total
print(f"\n{'='*70}")
print(f"META-LEARNER TEST ACCURACY (FIXED): {meta_test_accuracy:.4f} ({meta_test_accuracy*100:.2f}%)")
print(f"{'='*70}")
print(f"\nPrevious (BROKEN): 41.62%")
print(f"Expected improvement: ~30% jump to 70%+")
