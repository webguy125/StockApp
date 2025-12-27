"""
Quick Script: Retrain Meta-Learner Only
Uses existing trained base models and training data
"""

import sys
sys.path.insert(0, 'backend')

from backend.advanced_ml.training.training_pipeline import TrainingPipeline
from datetime import datetime

print("\n" + "=" * 70)
print("RETRAIN META-LEARNER ONLY")
print("=" * 70)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Initialize pipeline
print("\n[1/5] Initializing training pipeline...")
pipeline = TrainingPipeline()

# Load training data from database
print("\n[2/5] Loading training data from database...")
X_train, X_test, y_train, y_test, sample_weight = pipeline.load_training_data(
    test_size=0.2,
    use_rare_event_archive=False,
    use_regime_processing=False
)

print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")
print(f"  Features: {X_train.shape[1]}")

# Load all existing base models
print("\n[3/5] Loading existing base models...")
models_to_load = [
    ('random_forest', pipeline.rf_model),
    ('xgboost', pipeline.xgb_model),
    ('lightgbm', pipeline.lgbm_model),
    ('extratrees', pipeline.et_model),
    ('gradientboost', pipeline.gb_model),
    ('neural_network', pipeline.nn_model),
    ('logistic_regression', pipeline.lr_model),
    ('svm', pipeline.svm_model)
]

for name, model in models_to_load:
    if not model.is_trained:
        print(f"  Loading {name}...")
        model.load()
    else:
        print(f"  {name} already loaded")

# Train meta-learner with original stacking approach
print("\n[4/5] Training meta-learner (original stacking approach)...")
print("  This uses Logistic Regression to learn optimal combination")
print("  of base model predictions (proper ensemble stacking)")
pipeline.train_meta_learner(X_train, y_train)
pipeline.meta_learner.save()
print("  [OK] Meta-learner trained and saved")

# Evaluate all models
print("\n[5/5] Evaluating all models...")
results = pipeline.evaluate_models(X_test, y_test)

# Print results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# Find best model
best_model = None
best_acc = 0
for model_name, metrics in results.items():
    acc = metrics.get('accuracy', 0)
    if acc > best_acc:
        best_acc = acc
        best_model = model_name

print(f"\nBest Model: {best_model} ({best_acc:.4f})")
print(f"\nTarget: 0.9000 (90%)")

if best_model == 'meta_learner' and best_acc >= 0.90:
    print("\n✓ SUCCESS! Meta-learner is best model and achieved 90%+ accuracy!")
    print("  BASELINE RESTORED!")
elif best_model == 'meta_learner':
    print(f"\n✓ Meta-learner is best model ({best_acc:.4f})")
    print(f"  Close to target (gap: {0.90 - best_acc:.4f})")
else:
    print(f"\n✗ Meta-learner NOT best model")
    print(f"  Meta-learner: {results.get('meta_learner', {}).get('accuracy', 0):.4f}")
    print(f"  Best: {best_model} ({best_acc:.4f})")

print("\n" + "=" * 70)
print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
