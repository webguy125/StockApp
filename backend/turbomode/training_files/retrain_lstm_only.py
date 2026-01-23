"""
Retrain LSTM with New Anti-Overfitting Architecture
After fixing overfitting issues in lstm_model.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

import sqlite3
import numpy as np
import json
from advanced_ml.models.lstm_model import LSTMModel

# Load training data from database
db_path = "backend/data/advanced_ml_system.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

print("=" * 70)
print("RETRAIN LSTM WITH ANTI-OVERFITTING ARCHITECTURE")
print("=" * 70)
print(f"Database: {db_path}")
print()

# Load backtest samples
cursor.execute('''
    SELECT features, label
    FROM trades
    WHERE trade_type = "backtest"
    ORDER BY timestamp
''')

rows = cursor.fetchall()
print(f"[DATA] Loaded {len(rows)} samples from database")

# Parse features and labels
X_list = []
y_list = []

for features_blob, label in rows:
    try:
        features = json.loads(features_blob)
    except:
        import pickle
        features = pickle.loads(features_blob)

    # Extract feature values (exclude metadata)
    exclude_keys = ['feature_count', 'symbol', 'last_price', 'last_volume', 'timestamp', 'error']
    feature_values = []
    for key, value in sorted(features.items()):
        if key not in exclude_keys:
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                feature_values.append(float(value))

    X_list.append(feature_values)

    # Binary classification: buy=0, sell=1
    if label == 'buy':
        y_list.append(0)
    elif label == 'sell':
        y_list.append(1)

X = np.array(X_list)
y = np.array(y_list)

print(f"[DATA] Features shape: {X.shape}")
print(f"[DATA] Labels shape: {y.shape}")
print(f"[DATA] Feature count: {X.shape[1]}")
print()

# Split into train/test (80/20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"[DATA] Training samples: {X_train.shape[0]}")
print(f"[DATA] Test samples: {X_test.shape[0]}")
print()

# Initialize LSTM with new architecture
print("[INIT] Creating LSTM model with anti-overfitting architecture...")
lstm = LSTMModel(model_path="backend/data/turbomode_models/lstm", sequence_length=15)
print(f"  Hidden size: {lstm.hyperparameters['hidden_size']}")
print(f"  Learning rate: {lstm.hyperparameters['learning_rate']}")
print(f"  Dropout: 0.5 (LSTM and FC layers)")
print(f"  Early stopping patience: {lstm.hyperparameters['early_stopping_patience']}")
print()

# Train LSTM
print("[TRAIN] Training LSTM with new architecture...")
metrics = lstm.train(X_train, y_train, validate=False)

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"  Training samples: {metrics['n_samples']}")
print(f"  Training accuracy: {metrics['train_accuracy']:.4f}")
print(f"  Epochs trained: {metrics['epochs_trained']}")
print(f"  Final loss: {metrics['final_loss']:.4f}")
print()

# Evaluate on test set
print("[EVAL] Evaluating on test set...")
eval_metrics = lstm.evaluate(X_test, y_test)

print()
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)
print(f"  Test accuracy: {eval_metrics['accuracy']:.4f} ({eval_metrics['accuracy']*100:.2f}%)")
print(f"  Sequences evaluated: {eval_metrics['n_sequences_evaluated']}")
if 'buy_accuracy' in eval_metrics:
    print(f"  Buy accuracy: {eval_metrics['buy_accuracy']:.4f}")
if 'sell_accuracy' in eval_metrics:
    print(f"  Sell accuracy: {eval_metrics['sell_accuracy']:.4f}")
print()

# Compare to baseline
baseline_test_acc = 0.5087
improvement = (eval_metrics['accuracy'] - baseline_test_acc) * 100

print("=" * 70)
print("COMPARISON TO BASELINE")
print("=" * 70)
print(f"  OLD architecture (hidden=128, dropout=0.3):")
print(f"    Training: 84.63%")
print(f"    Test: 50.87%")
print(f"    Overfitting gap: -33.76%")
print()
print(f"  NEW architecture (hidden=64, dropout=0.5):")
print(f"    Training: {metrics['train_accuracy']*100:.2f}%")
print(f"    Test: {eval_metrics['accuracy']*100:.2f}%")
print(f"    Overfitting gap: {(metrics['train_accuracy'] - eval_metrics['accuracy'])*100:.2f}%")
print()
print(f"  IMPROVEMENT: {improvement:+.2f}% test accuracy")
print("=" * 70)
