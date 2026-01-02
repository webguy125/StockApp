"""
Retrain Meta-Learner Only
Loads existing 8 base models and retrains only the meta-learner
"""

import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from advanced_ml.models.xgboost_model import XGBoostModel
from advanced_ml.models.xgboost_et_model import XGBoostETModel
from advanced_ml.models.lightgbm_model import LightGBMModel
from advanced_ml.models.catboost_model import CatBoostModel
from advanced_ml.models.xgboost_hist_model import XGBoostHistModel
from advanced_ml.models.xgboost_dart_model import XGBoostDartModel
from advanced_ml.models.xgboost_gblinear_model import XGBoostGBLinearModel
from advanced_ml.models.xgboost_approx_model import XGBoostApproxModel
from advanced_ml.models.meta_learner import MetaLearner
from advanced_ml.backtesting.historical_backtest import HistoricalBacktest
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split

print("\n" + "=" * 70)
print("RETRAIN META-LEARNER ONLY - 8-MODEL ENSEMBLE")
print("=" * 70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Get script directory and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Define TurboMode model directory
TURBOMODE_MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "turbomode_models")

# Initialize database connection
db_path = os.path.join(PROJECT_ROOT, "backend", "data", "advanced_ml_system.db")
backtest = HistoricalBacktest(db_path)

# Load training data
print(f"\nPHASE 1: LOAD TRAINING DATA")
print("=" * 70)
X, y = backtest.prepare_training_data()

if len(X) == 0:
    print("[ERROR] No training data found in database!")
    sys.exit(1)

print(f"[DATA] Total samples: {len(X)}")
print(f"[DATA] Features: {X.shape[1]}")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[DATA] Training samples: {len(X_train)}")
print(f"[DATA] Test samples: {len(X_test)}")

# Load existing 8 base models (already trained)
print(f"\nPHASE 2: LOAD EXISTING BASE MODELS")
print("=" * 70)

print("[INFO] Loading pre-trained base models from disk...")

xgb_model = XGBoostModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "xgboost"), use_gpu=True)
et_model = XGBoostETModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "xgboost_et"))
lgbm_model = LightGBMModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "lightgbm"), use_gpu=True)
gb_model = CatBoostModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "catboost"))
hist_model = XGBoostHistModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "xgboost_hist"))
dart_model = XGBoostDartModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "xgboost_dart"))
gblinear_model = XGBoostGBLinearModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "xgboost_gblinear"))
approx_model = XGBoostApproxModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "xgboost_approx"))

# Load models from disk
xgb_model.load()
et_model.load()
lgbm_model.load()
gb_model.load()
hist_model.load()
dart_model.load()
gblinear_model.load()
approx_model.load()

print("[OK] All 8 base models loaded from disk")

# Initialize meta-learner
meta_learner = MetaLearner(model_path=os.path.join(TURBOMODE_MODEL_PATH, "meta_learner"), use_gpu=True)

# Register base models with meta-learner
print(f"\nPHASE 3: TRAIN META-LEARNER")
print("=" * 70)

print("[INFO] Registering all 8 base models with meta-learner...")
meta_learner.register_base_model('xgboost', xgb_model)
meta_learner.register_base_model('xgboost_et', et_model)
meta_learner.register_base_model('lightgbm', lgbm_model)
meta_learner.register_base_model('catboost', gb_model)
meta_learner.register_base_model('xgboost_hist', hist_model)
meta_learner.register_base_model('xgboost_dart', dart_model)
meta_learner.register_base_model('xgboost_gblinear', gblinear_model)
meta_learner.register_base_model('xgboost_approx', approx_model)

print("[INFO] Generating base model predictions for meta-learner training...")
# Convert X_train to feature dictionaries
feature_dicts = []
for i in range(len(X_train)):
    feat_dict = {f'feature_{j}': float(X_train[i, j]) for j in range(X_train.shape[1])}
    feature_dicts.append(feat_dict)

# Get predictions from all 8 models using BATCH prediction (FAST!)
print("  Getting XGBoost predictions...")
xgb_predictions = xgb_model.predict_batch(feature_dicts)

print("  Getting XGBoost ET predictions...")
et_predictions = et_model.predict_batch(feature_dicts)

print("  Getting LightGBM predictions...")
lgbm_predictions = lgbm_model.predict_batch(feature_dicts)

print("  Getting CatBoost predictions...")
gb_predictions = gb_model.predict_batch(feature_dicts)

print("  Getting XGBoost Hist predictions...")
hist_predictions = hist_model.predict_batch(feature_dicts)

print("  Getting XGBoost DART predictions...")
dart_predictions = dart_model.predict_batch(feature_dicts)

print("  Getting XGBoost GBLinear predictions...")
gblinear_predictions = gblinear_model.predict_batch(feature_dicts)

print("  Getting XGBoost Approx predictions...")
approx_predictions = approx_model.predict_batch(feature_dicts)

# Format predictions for meta-learner
all_base_preds = []
for i in range(len(X_train)):
    base_preds = {
        'xgboost': xgb_predictions[i],
        'xgboost_et': et_predictions[i],
        'lightgbm': lgbm_predictions[i],
        'catboost': gb_predictions[i],
        'xgboost_hist': hist_predictions[i],
        'xgboost_dart': dart_predictions[i],
        'xgboost_gblinear': gblinear_predictions[i],
        'xgboost_approx': approx_predictions[i]
    }
    all_base_preds.append(base_preds)

print("[INFO] Training meta-learner on base model predictions...")
meta_learner.train(all_base_preds, y_train)
meta_learner.save()
print("  [OK] Meta-learner trained and saved to turbomode_models/")

# Evaluate meta-learner on test set
print(f"\nPHASE 4: EVALUATE META-LEARNER")
print("=" * 70)

print("[INFO] Evaluating meta-learner on test set...")
meta_correct = 0
meta_total = len(X_test)

for i in range(meta_total):
    # Convert to feature dict
    feat_dict = {f'feature_{j}': float(X_test[i, j]) for j in range(X_test.shape[1])}

    # Get base model predictions (8-model ensemble)
    base_preds = {
        'xgboost': xgb_model.predict(feat_dict),
        'xgboost_et': et_model.predict(feat_dict),
        'lightgbm': lgbm_model.predict(feat_dict),
        'catboost': gb_model.predict(feat_dict),
        'xgboost_hist': hist_model.predict(feat_dict),
        'xgboost_dart': dart_model.predict(feat_dict),
        'xgboost_gblinear': gblinear_model.predict(feat_dict),
        'xgboost_approx': approx_model.predict(feat_dict)
    }

    # Get meta-learner prediction
    ensemble_pred = meta_learner.predict(base_preds)

    # Map prediction to class (BINARY: 0=Buy, 1=Sell)
    pred_map = {'buy': 0, 'sell': 1}
    pred_class = pred_map[ensemble_pred['prediction']]

    if pred_class == y_test[i]:
        meta_correct += 1

    # Progress indicator
    if (i + 1) % 500 == 0:
        print(f"    Progress: {i+1}/{meta_total} ({(i+1)/meta_total*100:.1f}%)")

meta_test_accuracy = meta_correct / meta_total
print(f"\n  {'Meta-Learner (8 models)':25s} {meta_test_accuracy:.4f} ({meta_test_accuracy*100:.2f}%)")

print(f"\n{'=' * 70}")
print("FINAL RESULTS")
print(f"{'=' * 70}")

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}")
print(f"\nMeta-Learner Test Accuracy: {meta_test_accuracy:.4f} ({meta_test_accuracy*100:.2f}%)")

print(f"\n{'=' * 70}")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 70}\n")

print("[COMPLETE] Meta-learner retrained with 8-model ensemble (no duplicate CatBoost)!")
