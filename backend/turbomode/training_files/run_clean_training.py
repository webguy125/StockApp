"""
Clean training without event features and regime balancing
To get back to baseline performance
"""
import sys
sys.path.append('backend')

from advanced_ml.training.training_pipeline import TrainingPipeline
from advanced_ml.features.feature_engineer import FeatureEngineer

# Initialize
pipeline = TrainingPipeline()

print("\n" + "=" * 70)
print("CLEAN BASELINE TRAINING (NO EVENT FEATURES, NO REGIME BALANCING)")
print("=" * 70)

# Load data WITHOUT regime processing
X_train, X_test, y_train, y_test, sample_weight = pipeline.load_training_data(
    test_size=0.2,
    use_rare_event_archive=False,  # Skip rare events
    use_regime_processing=False     # Skip regime balancing
)

print(f"\n[DATA] Training samples: {len(X_train)}")
print(f"[DATA] Test samples: {len(X_test)}")
print(f"[DATA] Features: {X_train.shape[1]}")

# Check if we still have event features
print(f"\n[CHECKING] Are event features included?")
# The feature engineer adds event features by default, we need to check

# Train models
print("\n" + "=" * 70)
print("TRAINING BASELINE MODELS")
print("=" * 70)

# Train just XGBoost for now to test
print("\n[TRAINING] XGBoost (baseline)...")
xgb_metrics = pipeline.xgb_model.train(X_train, y_train, validate=True)
pipeline.xgb_model.save()

print(f"\n[RESULTS]")
print(f"  Training Accuracy: {xgb_metrics['train_accuracy']:.4f}")
print(f"  CV Accuracy: {xgb_metrics.get('cv_mean', 'N/A')}")

# Evaluate on test set
from sklearn.metrics import accuracy_score
feature_dicts = [{f'feature_{j}': float(X_test[i, j]) for j in range(X_test.shape[1])} for i in range(len(X_test))]
predictions = pipeline.xgb_model.predict_batch(feature_dicts)
pred_labels = [0 if p['prediction'] == 'buy' else 1 if p['prediction'] == 'hold' else 2 for p in predictions]
test_acc = accuracy_score(y_test, pred_labels)

print(f"  Test Accuracy: {test_acc:.4f}")
print(f"\n{'=' * 70}")

if test_acc > 0.70:
    print("✓ GOOD! Test accuracy > 70% - baseline is healthy")
elif test_acc > 0.60:
    print("~ OK - Test accuracy > 60% but could be better")
else:
    print("✗ PROBLEM - Test accuracy < 60% - something is still wrong")

print("=" * 70)
