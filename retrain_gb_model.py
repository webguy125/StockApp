"""
Quick script to retrain only the Gradient Boosting model
Fixes pickle compatibility issue
"""

import sys
sys.path.insert(0, 'backend')

from advanced_ml.training.training_pipeline import TrainingPipeline
from advanced_ml.database.schema import AdvancedMLDatabase
import numpy as np

print("=" * 80)
print("RETRAINING GRADIENT BOOSTING MODEL ONLY")
print("=" * 80)

# Initialize pipeline
print("\n[1/4] Initializing ML pipeline...")
pipeline = TrainingPipeline(db_path="backend/backend/data/advanced_ml_system.db")

# Load training data from database using pipeline method
print("\n[2/4] Loading training data from ML database...")
result = pipeline.load_training_data(
    test_size=0.2,
    use_rare_event_archive=False,
    use_regime_processing=False
)

# Handle different return values (with/without sample_weight)
if len(result) == 5:
    X_train, X_test, y_train, y_test, sample_weight = result
else:
    X_train, X_test, y_train, y_test = result
    sample_weight = None

print(f"  X_train shape: {X_train.shape}")
print(f"  X_test shape: {X_test.shape}")
print(f"  y_train shape: {y_train.shape}")
print(f"  Classes: {np.unique(y_train)}")

# Train Gradient Boosting model only
print("\n[3/4] Training Gradient Boosting model...")
gb_metrics = pipeline.gb_model.train(X_train, y_train, validate=True, sample_weight=sample_weight)

print(f"\n  Training Accuracy: {gb_metrics['train_accuracy']:.4f}")
if 'cv_score' in gb_metrics:
    print(f"  Cross-Val Score: {gb_metrics['cv_score']:.4f}")

# Save model
print("\n[4/4] Saving model to disk...")
save_success = pipeline.gb_model.save()

if save_success:
    print("\n" + "=" * 80)
    print("GRADIENT BOOSTING MODEL RETRAINED SUCCESSFULLY")
    print("=" * 80)
    print(f"  Model saved to: backend/data/ml_models/gradientboost/")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Accuracy: {gb_metrics['train_accuracy']:.4f}")
    print("\nModel is now ready for TurboMode scanner!")
else:
    print("\nERROR: Failed to save model")
    sys.exit(1)
