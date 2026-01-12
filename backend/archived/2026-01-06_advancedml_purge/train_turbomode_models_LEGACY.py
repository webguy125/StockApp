"""
Train TurboMode Models - Autonomous System
Uses training data from turbomode.db (TurboMode's own database)
Saves models to backend/data/turbomode_models/

This creates independent models for TurboMode's overnight scanning system
ZERO DEPENDENCY ON SLIPSTREAM
"""

import sys
import os
import warnings

# Suppress annoying warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add backend to path
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

# GPU Models (Windows-compatible) - All XGBoost/CatBoost/LightGBM variants
from advanced_ml.models.xgboost_model import XGBoostModel  # XGBoost GPU (binary:logistic)
from advanced_ml.models.xgboost_et_model import XGBoostETModel  # XGBoost Extra Trees
from advanced_ml.models.lightgbm_model import LightGBMModel  # LightGBM GPU
from advanced_ml.models.catboost_model import CatBoostModel  # CatBoost GPU
from advanced_ml.models.xgboost_hist_model import XGBoostHistModel  # XGBoost Hist (NEW)
from advanced_ml.models.xgboost_dart_model import XGBoostDartModel  # XGBoost DART (NEW)
from advanced_ml.models.xgboost_gblinear_model import XGBoostGBLinearModel  # XGBoost GBLinear (NEW)
from advanced_ml.models.xgboost_approx_model import XGBoostApproxModel  # XGBoost Approx (NEW)
from advanced_ml.models.meta_learner import MetaLearner  # XGBoost GPU meta-learner
from advanced_ml.database.schema import AdvancedMLDatabase
from advanced_ml.backtesting.historical_backtest import HistoricalBacktest
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
import torch  # For GPU memory management

print("\n" + "=" * 70)
print("TRAIN TURBOMODE MODELS - AUTONOMOUS SYSTEM")
print("=" * 70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"TurboMode Database: backend/data/turbomode.db")
print(f"Output Directory: backend/data/turbomode_models/")
print(f"Configuration: Clean baseline (179 features, no events)")
print("=" * 70)

# Get script directory and project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Go up 2 levels from turbomode/

# DEBUG: Print path calculation
print(f"\n[DEBUG] __file__: {__file__}")
print(f"[DEBUG] SCRIPT_DIR: {SCRIPT_DIR}")
print(f"[DEBUG] PROJECT_ROOT: {PROJECT_ROOT}")

# Define TurboMode model directory (absolute path)
TURBOMODE_MODEL_PATH = os.path.join(PROJECT_ROOT, "backend", "data", "turbomode_models")

# Ensure directory exists
os.makedirs(TURBOMODE_MODEL_PATH, exist_ok=True)

# Initialize database connection (TurboMode autonomous database)
# TurboMode has its own training data - completely separate from Slipstream
db_path = os.path.join(PROJECT_ROOT, "backend", "data", "turbomode.db")
print(f"\n[DATABASE] TurboMode Database: {db_path}")
print(f"[DATABASE] File exists: {os.path.exists(db_path)}")
print(f"[DATABASE] File size: {os.path.getsize(db_path) / 1024 / 1024:.1f} MB" if os.path.exists(db_path) else "[DATABASE] FILE NOT FOUND!")
print(f"[INFO] Autonomous TurboMode training database")

db = AdvancedMLDatabase(db_path)
backtest = HistoricalBacktest(db_path)

# Load training data
print(f"\n{'=' * 70}")
print("PHASE 1: LOAD TRAINING DATA FROM SHARED DATABASE")
print(f"{'=' * 70}\n")

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

if X_train.shape[1] != 179:
    print(f"[WARNING] Expected 179 features, got {X_train.shape[1]}")

# Initialize TurboMode models with custom paths
print(f"\n{'=' * 70}")
print("PHASE 2: INITIALIZE TURBOMODE MODELS")
print(f"{'=' * 70}\n")

print(f"[INIT] Creating GPU models in {TURBOMODE_MODEL_PATH}/")
print(f"[GPU] 100% GPU-accelerated ensemble (5 XGBoost variants + 1 CatBoost + 1 LightGBM + 1 XGBoost ET)")

# All 8 base models are GPU-accelerated XGBoost/CatBoost/LightGBM variants
xgb_model = XGBoostModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "xgboost"), use_gpu=True)  # Binary logistic
et_model = XGBoostETModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "xgboost_et"))  # Extra Trees
lgbm_model = LightGBMModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "lightgbm"), use_gpu=True)  # LightGBM
gb_model = CatBoostModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "catboost"))  # CatBoost
hist_model = XGBoostHistModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "xgboost_hist"))  # Histogram (NEW)
dart_model = XGBoostDartModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "xgboost_dart"))  # DART dropout (NEW)
gblinear_model = XGBoostGBLinearModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "xgboost_gblinear"))  # GBLinear (NEW)
approx_model = XGBoostApproxModel(model_path=os.path.join(TURBOMODE_MODEL_PATH, "xgboost_approx"))  # Approx (NEW)
meta_learner = MetaLearner(model_path=os.path.join(TURBOMODE_MODEL_PATH, "meta_learner"), use_gpu=True)  # XGBoost GPU meta-learner

print("[OK] All model objects initialized")

# Train all 8 base models (all XGBoost/CatBoost/LightGBM variants)
print(f"\n{'=' * 70}")
print("PHASE 3: TRAIN ALL 8 GPU-ACCELERATED MODELS")
print(f"{'=' * 70}\n")

models_to_train = [
    ('XGBoost GPU', xgb_model),
    ('XGBoost ET GPU', et_model),
    ('LightGBM GPU', lgbm_model),
    ('CatBoost GPU', gb_model),
    ('XGBoost Hist GPU', hist_model),
    ('XGBoost DART GPU', dart_model),
    ('XGBoost GBLinear GPU', gblinear_model),
    ('XGBoost Approx GPU', approx_model)
]

for i, (name, model) in enumerate(models_to_train, 1):
    print(f"\n[{i}/8] Training {name}...")
    model.train(X_train, y_train)
    model.save()
    print(f"  [OK] {name} trained and saved to turbomode_models/")

    # Free GPU memory after each model to prevent segmentation fault
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"  [GPU] Memory cleared after {name}")

# Register base models with meta-learner
print(f"\n{'=' * 70}")
print("PHASE 4: TRAIN META-LEARNER")
print(f"{'=' * 70}\n")

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

# Evaluate all models
print(f"\n{'=' * 70}")
print("PHASE 5: EVALUATE ALL 8 BASE MODELS + META-LEARNER")
print(f"{'=' * 70}\n")

results = {}

# Evaluate base models
for name, model in models_to_train:
    metrics = model.evaluate(X_test, y_test)
    results[name.lower().replace(' ', '_')] = metrics
    print(f"  {name:25s} {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")

# Evaluate meta-learner manually on test set
print("\n[INFO] Evaluating meta-learner on test set...")
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
    if (i + 1) % 1000 == 0:
        print(f"    Progress: {i+1}/{meta_total} ({(i+1)/meta_total*100:.1f}%)")

meta_test_accuracy = meta_correct / meta_total
results['meta_learner'] = {'accuracy': meta_test_accuracy}
print(f"  {'Meta-Learner':25s} {meta_test_accuracy:.4f} ({meta_test_accuracy*100:.2f}%)")

# Find best model
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
results['best_model'] = best_model[0]
results['best_accuracy'] = best_model[1]['accuracy']

print(f"\n{'=' * 70}")
print("FINAL RESULTS")
print(f"{'=' * 70}\n")

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}")
print(f"\nBest Model: {best_model[0]} ({best_model[1]['accuracy']:.4f})\n")

# List model files created
print(f"{'=' * 70}")
print("TURBOMODE MODELS SAVED")
print(f"{'=' * 70}\n")

for root, dirs, files in os.walk(TURBOMODE_MODEL_PATH):
    level = root.replace(TURBOMODE_MODEL_PATH, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")

print(f"\n{'=' * 70}")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 70}\n")

print("[COMPLETE] TurboMode models trained and saved!")
print("[NOTE] These models are independent from Slipstream models")
print("[NEXT] Update overnight_scanner.py to use these models")
