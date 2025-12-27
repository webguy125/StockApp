"""
Train ALL Models Fresh - Skip Backtest
Uses existing 34,086 samples from overnight backtest
Trains all 8 base models + meta-learner
"""

import sys
sys.path.insert(0, 'backend')

from backend.advanced_ml.training.training_pipeline import TrainingPipeline
from backend.advanced_ml.config.core_symbols import CORE_SYMBOLS
from datetime import datetime
import json

print("\n" + "=" * 70)
print("TRAIN ALL MODELS FRESH - SKIP BACKTEST")
print("=" * 70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Database: backend/backend/data/advanced_ml_system.db (34,086 samples)")
print(f"Configuration: Clean baseline (179 features, no events)")
print("=" * 70)

# Initialize pipeline with absolute database path
import os
db_path = os.path.abspath("backend/backend/data/advanced_ml_system.db")
print(f"Using database: {db_path}")
pipeline = TrainingPipeline(db_path=db_path)

# Get symbols list for metadata
all_symbols = []
for sector, market_caps in CORE_SYMBOLS.items():
    for cap_category, symbol_list in market_caps.items():
        all_symbols.extend(symbol_list)

print(f"\n[SYMBOLS] Training on {len(all_symbols)} symbols worth of data")

# PHASE 1: Load Training Data (from existing database)
print(f"\n{'=' * 70}")
print("PHASE 1: LOAD TRAINING DATA FROM DATABASE")
print(f"{'=' * 70}\n")

X_train, X_test, y_train, y_test, sample_weight = pipeline.load_training_data(
    test_size=0.2,
    use_rare_event_archive=False,  # DISABLED
    use_regime_processing=False     # DISABLED
)

print(f"\n[DATA] Training samples: {len(X_train)}")
print(f"[DATA] Test samples: {len(X_test)}")
print(f"[DATA] Features: {X_train.shape[1]}")

if X_train.shape[1] != 179:
    print(f"\n[WARNING] Expected 179 features, got {X_train.shape[1]}")

# PHASE 2: Train ALL 8 Base Models
print(f"\n{'=' * 70}")
print("PHASE 2: TRAIN ALL 8 BASE MODELS")
print(f"{'=' * 70}\n")

models_to_train = [
    ('Random Forest', pipeline.rf_model),
    ('XGBoost', pipeline.xgb_model),
    ('LightGBM', pipeline.lgbm_model),
    ('Extra Trees', pipeline.et_model),
    ('Gradient Boosting', pipeline.gb_model),
    ('Neural Network', pipeline.nn_model),
    ('Logistic Regression', pipeline.lr_model),
    ('SVM', pipeline.svm_model)
]

for i, (name, model) in enumerate(models_to_train, 1):
    print(f"\n[{i}/8] Training {name}...")
    model.train(X_train, y_train, sample_weight=sample_weight)
    model.save()
    print(f"  [OK] {name} trained and saved")

# PHASE 3: Train Meta-Learner (Original Stacking)
print(f"\n{'=' * 70}")
print("PHASE 3: TRAIN META-LEARNER (ORIGINAL STACKING)")
print(f"{'=' * 70}\n")

print("[INFO] Using original MetaLearner with LogisticRegression stacking")
print("       This should achieve 90%+ accuracy by learning optimal")
print("       combination of all 8 base models\n")

pipeline.train_meta_learner(X_train, y_train)
pipeline.meta_learner.save()
print("  [OK] Meta-learner trained and saved")

# PHASE 4: Evaluate ALL Models
print(f"\n{'=' * 70}")
print("PHASE 4: EVALUATE ALL 9 MODELS")
print(f"{'=' * 70}\n")

results = pipeline.evaluate_models(X_test, y_test)

# PHASE 5: Display Results
print(f"\n{'=' * 70}")
print("FINAL RESULTS")
print(f"{'=' * 70}\n")

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}")
print(f"\nModel Performance on Test Set:\n")

# Sort by accuracy
model_scores = []
for model_name, metrics in results.items():
    if model_name not in ['best_model', 'best_accuracy']:
        acc = metrics.get('accuracy', 0)
        model_scores.append((model_name, acc))

model_scores.sort(key=lambda x: x[1], reverse=True)

for model_name, acc in model_scores:
    marker = "  ← BEST MODEL" if model_name == results.get('best_model') else ""
    print(f"  {model_name:25s} {acc:.4f} ({acc*100:.2f}%){marker}")

print(f"\n{'=' * 70}")
print("TARGET ASSESSMENT")
print(f"{'=' * 70}\n")

meta_acc = results.get('meta_learner', {}).get('accuracy', 0)
best_model = results.get('best_model', 'unknown')

if meta_acc >= 0.90 and best_model == 'meta_learner':
    print("✓ SUCCESS! Meta-learner achieved 90%+ and is the best model!")
    print("  BASELINE RESTORED!")
elif meta_acc >= 0.88:
    print(f"~ VERY CLOSE! Meta-learner at {meta_acc:.4f} ({meta_acc*100:.2f}%)")
    print(f"  Gap to 90%: {(0.90 - meta_acc)*100:.2f}%")
    if best_model != 'meta_learner':
        print(f"  Note: {best_model} is currently best model")
else:
    print(f"✗ Below target. Meta-learner at {meta_acc:.4f} ({meta_acc*100:.2f}%)")
    print(f"  Best model: {best_model} ({results.get('best_accuracy', 0):.4f})")

# Save results
results_file = 'backend/backend/data/training_results_final.json'
results_summary = {
    'timestamp': datetime.now().isoformat(),
    'feature_count': X_train.shape[1],
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'symbols': all_symbols,
    'model_results': results
}

with open(results_file, 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

print(f"\n[OK] Results saved to {results_file}")

print(f"\n{'=' * 70}")
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'=' * 70}\n")

print("[COMPLETE] All models trained and evaluated!")
