"""
Full Model Training - All 9 Models (8 base + meta-learner)
Uses existing 8,760 samples from 20 stocks
"""
import sys
sys.path.insert(0, 'backend')

from backend.advanced_ml.training.training_pipeline import TrainingPipeline
from datetime import datetime
import json
import numpy as np

print("\n" + "=" * 70)
print("FULL MODEL TRAINING - ALL 9 MODELS")
print("=" * 70)
print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Models: 8 base models + meta-learner = 9 total")
print(f"Features: 179 (technical only)")
print(f"Data: 8,760 samples from 20 stocks")
print("=" * 70)

# Initialize pipeline
print("\n[INIT] Initializing training pipeline...")
pipeline = TrainingPipeline()

# Load Training Data
print(f"\n{'=' * 70}")
print("PHASE 1: LOAD TRAINING DATA")
print(f"{'=' * 70}\n")

result = pipeline.load_training_data(
    test_size=0.2,
    use_rare_event_archive=False,
    use_regime_processing=False
)

if len(result) == 5:
    X_train, X_test, y_train, y_test, sample_weight = result
else:
    X_train, X_test, y_train, y_test = result
    sample_weight = None

print(f"\n[DATA] Training samples: {len(X_train)}")
print(f"[DATA] Test samples: {len(X_test)}")
print(f"[DATA] Features: {X_train.shape[1]}")

# Train All 8 Base Models
print(f"\n{'=' * 70}")
print("PHASE 2: TRAIN 8 BASE MODELS")
print(f"{'=' * 70}\n")

models = [
    ('Random Forest', pipeline.rf_model),
    ('XGBoost', pipeline.xgb_model),
    ('LightGBM', pipeline.lgbm_model),
    ('Extra Trees', pipeline.et_model),
    ('Gradient Boosting', pipeline.gb_model),
    ('Neural Network', pipeline.nn_model),
    ('Logistic Regression', pipeline.lr_model),
    ('SVM', pipeline.svm_model)
]

for i, (name, model) in enumerate(models, 1):
    print(f"\n[{i}/8] Training {name}...")
    start = datetime.now()
    model.train(X_train, y_train, sample_weight=sample_weight)
    model.save()
    duration = (datetime.now() - start).total_seconds()
    print(f"    [OK] Trained in {duration:.1f}s")

# Train Meta-Learner
print(f"\n{'=' * 70}")
print("PHASE 3: TRAIN META-LEARNER (Model #9)")
print(f"{'=' * 70}\n")

print("[TRAINING] Meta-Learner (ensemble of all 8 models)...")
start = datetime.now()

# Train meta-learner on training data
pipeline.train_meta_learner(X_train, y_train)
pipeline.meta_learner.save()

duration = (datetime.now() - start).total_seconds()
print(f"    [OK] Meta-Learner trained in {duration:.1f}s")

# Evaluate All Models
print(f"\n{'=' * 70}")
print("PHASE 4: EVALUATE ALL 9 MODELS")
print(f"{'=' * 70}\n")

from sklearn.metrics import accuracy_score, classification_report

results = {}

# Add meta-learner to evaluation
all_models = models + [('Meta-Learner', pipeline.meta_learner)]

for name, model in all_models:
    print(f"\n[EVALUATING] {name}...")

    # Convert to feature dicts
    test_dicts = [{f'feature_{j}': float(X_test[i, j]) for j in range(X_test.shape[1])}
                  for i in range(len(X_test))]

    # Get predictions
    predictions = model.predict_batch(test_dicts)

    # Extract labels
    label_map = {'buy': 0, 'hold': 1, 'sell': 2}
    pred_labels = []
    for pred in predictions:
        p = pred['prediction']
        if isinstance(p, (int, np.integer)):
            pred_labels.append(p)
        else:
            pred_labels.append(label_map.get(p, 1))

    # Test accuracy
    test_acc = accuracy_score(y_test, pred_labels)

    # Training accuracy (sample 1000)
    train_size = min(1000, len(X_train))
    train_dicts = [{f'feature_{j}': float(X_train[i, j]) for j in range(X_train.shape[1])}
                   for i in range(train_size)]
    train_preds = model.predict_batch(train_dicts)
    train_labels = []
    for pred in train_preds:
        p = pred['prediction']
        if isinstance(p, (int, np.integer)):
            train_labels.append(p)
        else:
            train_labels.append(label_map.get(p, 1))
    train_acc = accuracy_score(y_train[:train_size], train_labels)

    results[name] = {
        'test_accuracy': test_acc,
        'train_accuracy': train_acc,
        'gap': train_acc - test_acc
    }

    print(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test Accuracy:     {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Gap:               {results[name]['gap']:.4f} ({results[name]['gap']*100:.2f}%)")

# Final Summary
print(f"\n{'=' * 70}")
print("ALL 9 MODELS TRAINED!")
print(f"{'=' * 70}")
print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

print(f"\n{'=' * 70}")
print("FINAL RESULTS - ALL 9 MODELS")
print(f"{'=' * 70}\n")

# Sort by test accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)

for rank, (name, result) in enumerate(sorted_results, 1):
    status = "EXCELLENT" if result['test_accuracy'] > 0.85 else \
             "GOOD" if result['test_accuracy'] > 0.75 else \
             "OK" if result['test_accuracy'] > 0.65 else \
             "NEEDS WORK"

    print(f"{rank}. {name:20} Test: {result['test_accuracy']*100:5.2f}%  Gap: {result['gap']*100:5.2f}%  [{status}]")

# Save results
results_file = 'backend/data/quick_training_results.json'
with open(results_file, 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': X_train.shape[1],
        'results': results
    }, f, indent=2, default=str)

print(f"\n  Results saved: {results_file}")

# Final verdict
best_model = sorted_results[0]
best_acc = best_model[1]['test_accuracy']

print(f"\n{'=' * 70}")
print("BASELINE CHECK:")
print(f"{'=' * 70}")
print(f"\nBest Model: {best_model[0]}")
print(f"Test Accuracy: {best_acc*100:.2f}%")

if best_acc >= 0.90:
    print("\nSUCCESS! Test accuracy >= 90% - BASELINE RESTORED!")
elif best_acc >= 0.85:
    print("\nVERY GOOD! Test accuracy >= 85% - Close to baseline")
elif best_acc >= 0.80:
    print("\nGOOD! Test accuracy >= 80%")
else:
    print("\nNeeds improvement - consider more data")
print("=" * 70)

print("\n[SUCCESS] Full 9-model training completed!")
