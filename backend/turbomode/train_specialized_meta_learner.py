"""
Train Specialized Meta-Learner for Top 10 Stocks
Uses backtest data from only the most predictable stocks
Expected to achieve 75-80%+ accuracy on focused universe
"""

import sys
import os
import warnings

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
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import sqlite3
import json

print("\n" + "=" * 70)
print("TRAIN SPECIALIZED META-LEARNER FOR TOP 10 STOCKS")
print("=" * 70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Get script directory and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Define paths
TURBOMODE_MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "turbomode_models")
DB_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "advanced_ml_system.db")
RANKINGS_FILE = os.path.join(PROJECT_ROOT, "backend", "data", "stock_rankings.json")

# Load top 10 stocks from rankings
print(f"\nPHASE 1: LOAD TOP 10 STOCKS FROM RANKINGS")
print("=" * 70)

if not os.path.exists(RANKINGS_FILE):
    print(f"[ERROR] Rankings file not found: {RANKINGS_FILE}")
    print("[INFO] Please run: python adaptive_stock_ranker.py")
    sys.exit(1)

with open(RANKINGS_FILE, 'r') as f:
    rankings = json.load(f)

top_10_symbols = [stock['symbol'] for stock in rankings['top_10']]
print(f"[INFO] Top 10 stocks: {', '.join(top_10_symbols)}")

# Load training data from database (ONLY top 10 stocks)
print(f"\nPHASE 2: LOAD TRAINING DATA FOR TOP 10 STOCKS ONLY")
print("=" * 70)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Build SQL query to filter only top 10 stocks
placeholders = ','.join(['?' for _ in top_10_symbols])
query = f'''
    SELECT entry_features_json, outcome
    FROM trades
    WHERE trade_type = 'backtest'
    AND entry_features_json IS NOT NULL
    AND outcome IS NOT NULL
    AND symbol IN ({placeholders})
'''

cursor.execute(query, top_10_symbols)
rows = cursor.fetchall()
conn.close()

if not rows:
    print("[ERROR] No training data found for top 10 stocks!")
    sys.exit(1)

# Extract features and labels
X_list = []
y_list = []

for row in rows:
    features_json = row[0]
    outcome = row[1]

    # Parse features
    features_dict = json.loads(features_json)
    feature_array = np.array([features_dict[f'feature_{i}'] for i in range(179)])

    # Map outcome to label (0=buy, 1=sell)
    label = 0 if outcome == 'buy' else 1

    X_list.append(feature_array)
    y_list.append(label)

X = np.array(X_list)
y = np.array(y_list)

print(f"[DATA] Total samples from top 10 stocks: {len(X)}")
print(f"[DATA] Features: {X.shape[1]}")
print(f"[DATA] Buy samples: {np.sum(y == 0)}")
print(f"[DATA] Sell samples: {np.sum(y == 1)}")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"[DATA] Training samples: {len(X_train)}")
print(f"[DATA] Test samples: {len(X_test)}")

# Load existing base models (already trained on ALL stocks)
print(f"\nPHASE 3: LOAD EXISTING BASE MODELS")
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

print("[OK] All 8 base models loaded")

# Initialize SPECIALIZED meta-learner
meta_learner_specialized = MetaLearner(
    model_path=os.path.join(TURBOMODE_MODEL_PATH, "meta_learner_top10"),
    use_gpu=True
)

# Register base models
print(f"\nPHASE 4: TRAIN SPECIALIZED META-LEARNER")
print("=" * 70)

print("[INFO] Registering all 8 base models...")
meta_learner_specialized.register_base_model('xgboost', xgb_model)
meta_learner_specialized.register_base_model('xgboost_et', et_model)
meta_learner_specialized.register_base_model('lightgbm', lgbm_model)
meta_learner_specialized.register_base_model('catboost', gb_model)
meta_learner_specialized.register_base_model('xgboost_hist', hist_model)
meta_learner_specialized.register_base_model('xgboost_dart', dart_model)
meta_learner_specialized.register_base_model('xgboost_gblinear', gblinear_model)
meta_learner_specialized.register_base_model('xgboost_approx', approx_model)

print("[INFO] Generating base model predictions...")
# Convert X_train to feature dictionaries
feature_dicts = []
for i in range(len(X_train)):
    feat_dict = {f'feature_{j}': float(X_train[i, j]) for j in range(X_train.shape[1])}
    feature_dicts.append(feat_dict)

# Get predictions from all 8 models
print("  XGBoost...")
xgb_predictions = xgb_model.predict_batch(feature_dicts)
print("  XGBoost ET...")
et_predictions = et_model.predict_batch(feature_dicts)
print("  LightGBM...")
lgbm_predictions = lgbm_model.predict_batch(feature_dicts)
print("  CatBoost...")
gb_predictions = gb_model.predict_batch(feature_dicts)
print("  XGBoost Hist...")
hist_predictions = hist_model.predict_batch(feature_dicts)
print("  XGBoost DART...")
dart_predictions = dart_model.predict_batch(feature_dicts)
print("  XGBoost GBLinear...")
gblinear_predictions = gblinear_model.predict_batch(feature_dicts)
print("  XGBoost Approx...")
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

print("[INFO] Training specialized meta-learner...")
meta_learner_specialized.train(all_base_preds, y_train)
meta_learner_specialized.save()
print("  [OK] Specialized meta-learner saved")

# Evaluate specialized meta-learner
print(f"\nPHASE 5: EVALUATE SPECIALIZED META-LEARNER")
print("=" * 70)

meta_correct = 0
for i in range(len(X_test)):
    feat_dict = {f'feature_{j}': float(X_test[i, j]) for j in range(X_test.shape[1])}

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

    ensemble_pred = meta_learner_specialized.predict(base_preds)
    pred_map = {'buy': 0, 'sell': 1}
    pred_class = pred_map[ensemble_pred['prediction']]

    if pred_class == y_test[i]:
        meta_correct += 1

specialized_accuracy = meta_correct / len(X_test)

# Compare with general meta-learner
print("\nComparing with general meta-learner...")
meta_learner_general = MetaLearner(
    model_path=os.path.join(TURBOMODE_MODEL_PATH, "meta_learner"),
    use_gpu=True
)
meta_learner_general.register_base_model('xgboost', xgb_model)
meta_learner_general.register_base_model('xgboost_et', et_model)
meta_learner_general.register_base_model('lightgbm', lgbm_model)
meta_learner_general.register_base_model('catboost', gb_model)
meta_learner_general.register_base_model('xgboost_hist', hist_model)
meta_learner_general.register_base_model('xgboost_dart', dart_model)
meta_learner_general.register_base_model('xgboost_gblinear', gblinear_model)
meta_learner_general.register_base_model('xgboost_approx', approx_model)

general_correct = 0
for i in range(len(X_test)):
    feat_dict = {f'feature_{j}': float(X_test[i, j]) for j in range(X_test.shape[1])}

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

    ensemble_pred = meta_learner_general.predict(base_preds)
    pred_class = pred_map[ensemble_pred['prediction']]

    if pred_class == y_test[i]:
        general_correct += 1

general_accuracy = general_correct / len(X_test)

print(f"\n{'=' * 70}")
print("FINAL RESULTS")
print(f"{'=' * 70}")

print(f"\nTop 10 Stocks: {', '.join(top_10_symbols)}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

print(f"\n{'=' * 70}")
print("ACCURACY COMPARISON ON TOP 10 STOCKS")
print(f"{'=' * 70}")

print(f"\nGeneral Meta-Learner (all 80 stocks):     {general_accuracy:.4f} ({general_accuracy*100:.2f}%)")
print(f"Specialized Meta-Learner (top 10 stocks): {specialized_accuracy:.4f} ({specialized_accuracy*100:.2f}%)")

improvement = (specialized_accuracy - general_accuracy) * 100
if improvement > 0:
    print(f"\n🎯 Improvement: +{improvement:.2f} percentage points")
    print(f"✅ Specialized meta-learner is BETTER for top 10 stocks!")
else:
    print(f"\n⚠️ Improvement: {improvement:.2f} percentage points")
    print(f"⚠️ Specialized meta-learner did not improve significantly")

print(f"\n{'=' * 70}")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 70}\n")

print("[COMPLETE] Specialized meta-learner trained!")
print(f"[SAVED] Model saved to: meta_learner_top10/")
