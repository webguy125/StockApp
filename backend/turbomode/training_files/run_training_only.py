"""
Training-Only Script - Skip Data Collection
Uses existing data from database to train models quickly
"""
import sys
sys.path.insert(0, 'backend')

from backend.advanced_ml.training.training_pipeline import TrainingPipeline
from datetime import datetime
import json
import numpy as np

print("\n" + "=" * 70)
print("FAST TRAINING (USING EXISTING DATA)")
print("=" * 70)
print(f"\nStart Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Features: 179 (technical only, no events)")
print(f"Regime Processing: DISABLED")
print(f"\nSkipping data collection - using existing 8,760 samples")
print("=" * 70)

# Initialize pipeline
print("\n[INIT] Initializing training pipeline...")
pipeline = TrainingPipeline()

# PHASE 1: Load Training Data (from existing database)
print(f"\n{'=' * 70}")
print("PHASE 1: LOAD TRAINING DATA FROM DATABASE")
print(f"{'=' * 70}\n")

result = pipeline.load_training_data(
    test_size=0.2,
    use_rare_event_archive=False,
    use_regime_processing=False
)

# Handle different return values
if len(result) == 5:
    X_train, X_test, y_train, y_test, sample_weight = result
else:
    X_train, X_test, y_train, y_test = result
    sample_weight = None

print(f"\n[DATA] Training samples: {len(X_train)}")
print(f"[DATA] Test samples: {len(X_test)}")
print(f"[DATA] Features per sample: {X_train.shape[1]}")

if X_train.shape[1] != 179:
    print(f"\n[WARNING] Expected 179 features (technical only), got {X_train.shape[1]}")
    sys.exit(1)

# Show label distribution
from collections import Counter
train_dist = Counter(y_train)
total = len(y_train)
print(f"\n[LABELS] Training distribution:")
print(f"  Buy (0):  {train_dist[0]:5d} ({train_dist[0]/total*100:5.1f}%)")
print(f"  Hold (1): {train_dist[1]:5d} ({train_dist[1]/total*100:5.1f}%)")
print(f"  Sell (2): {train_dist[2]:5d} ({train_dist[2]/total*100:5.1f}%)")

# PHASE 2: Train Models
print(f"\n{'=' * 70}")
print("PHASE 2: TRAIN MODELS")
print(f"{'=' * 70}\n")

models_to_train = [
    ('XGBoost', pipeline.xgb_model),
    ('Random Forest', pipeline.rf_model),
    ('LightGBM', pipeline.lgbm_model)
]

for name, model in models_to_train:
    print(f"\n[TRAINING] {name}...")
    start = datetime.now()
    model.train(X_train, y_train, sample_weight=sample_weight)
    model.save()
    duration = (datetime.now() - start).total_seconds()
    print(f"    [OK] {name} trained in {duration:.1f}s")

# PHASE 3: Evaluate Models
print(f"\n{'=' * 70}")
print("PHASE 3: EVALUATE MODELS")
print(f"{'=' * 70}\n")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

results = {}

for name, model in models_to_train:
    print(f"\n[EVALUATING] {name}...")

    # Convert test data to feature dicts
    feature_dicts = [{f'feature_{j}': float(X_test[i, j]) for j in range(X_test.shape[1])}
                    for i in range(len(X_test))]

    # Get predictions
    predictions = model.predict_batch(feature_dicts)

    # Extract labels
    label_map = {'buy': 0, 'hold': 1, 'sell': 2}
    pred_labels = []
    for pred in predictions:
        p = pred['prediction']
        if isinstance(p, (int, np.integer)):
            pred_labels.append(p)
        else:
            pred_labels.append(label_map[p])

    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test, pred_labels)

    # Get training accuracy (sample 1000 for speed)
    train_sample_size = min(1000, len(X_train))
    train_feature_dicts = [{f'feature_{j}': float(X_train[i, j]) for j in range(X_train.shape[1])}
                           for i in range(train_sample_size)]
    train_preds = model.predict_batch(train_feature_dicts)
    train_labels = []
    for pred in train_preds:
        p = pred['prediction']
        if isinstance(p, (int, np.integer)):
            train_labels.append(p)
        else:
            train_labels.append(label_map[p])
    train_accuracy = accuracy_score(y_train[:train_sample_size], train_labels)

    # Per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, pred_labels, labels=[0, 1, 2], zero_division=0
    )

    results[name] = {
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'gap': train_accuracy - test_accuracy,
        'buy_precision': precision[0],
        'buy_recall': recall[0],
        'sell_precision': precision[2],
        'sell_recall': recall[2]
    }

    print(f"\n  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Test Accuracy:     {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  Gap:               {results[name]['gap']:.4f} ({results[name]['gap']*100:.2f}%)")

    print(f"\n  Per-Class Performance:")
    print(f"    Buy Precision:  {precision[0]:.4f} (when it says buy, is it right?)")
    print(f"    Buy Recall:     {recall[0]:.4f} (does it catch good buys?)")
    print(f"    Sell Precision: {precision[2]:.4f} (when it says sell, is it right?)")
    print(f"    Sell Recall:    {recall[2]:.4f} (does it catch good sells?)")

    # Confusion matrix
    cm = confusion_matrix(y_test, pred_labels, labels=[0, 1, 2])
    print(f"\n  Confusion Matrix:")
    print(f"           Pred: Buy  Hold  Sell")
    print(f"    True Buy:  {cm[0,0]:4d}  {cm[0,1]:4d}  {cm[0,2]:4d}")
    print(f"    True Hold: {cm[1,0]:4d}  {cm[1,1]:4d}  {cm[1,2]:4d}")
    print(f"    True Sell: {cm[2,0]:4d}  {cm[2,1]:4d}  {cm[2,2]:4d}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, pred_labels,
                               target_names=['buy', 'hold', 'sell'],
                               digits=4))

# Final Summary
print(f"\n{'=' * 70}")
print("TRAINING COMPLETE")
print(f"{'=' * 70}")
print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nResults Summary:")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Test samples: {len(X_test)}")
print(f"  - Features: {X_train.shape[1]}")

print(f"\nModel Performance:")
for name, result in results.items():
    status = "✓ EXCELLENT" if result['test_accuracy'] > 0.85 else \
             "✓ GOOD" if result['test_accuracy'] > 0.75 else \
             "~ OK" if result['test_accuracy'] > 0.65 else \
             "✗ NEEDS WORK"
    print(f"  {name:15} Test: {result['test_accuracy']*100:5.2f}%  Gap: {result['gap']*100:5.2f}%  {status}")
    print(f"                 Buy Precision: {result['buy_precision']*100:5.2f}%  Sell Precision: {result['sell_precision']*100:5.2f}%")

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
print("=" * 70)

# Final verdict
best_test_acc = max([r['test_accuracy'] for r in results.values()])
print(f"\n{'=' * 70}")
print("BASELINE ACCURACY CHECK:")
if best_test_acc >= 0.90:
    print("✓ SUCCESS! Test accuracy ≥ 90% - BASELINE RESTORED!")
elif best_test_acc >= 0.80:
    print("✓ GOOD! Test accuracy ≥ 80% - Close to baseline")
elif best_test_acc >= 0.70:
    print("~ ACCEPTABLE - Test accuracy ≥ 70%")
else:
    print("✗ PROBLEM - Test accuracy < 70% - Further debugging needed")
print(f"\nBest Model Test Accuracy: {best_test_acc*100:.2f}%")
print("=" * 70)

print("\n[SUCCESS] Fast training completed!")
