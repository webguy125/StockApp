"""
Quick script to add validation set support to all 5 XGBoost variant models.
This follows the same pattern used in xgboost_model.py.
"""

import os
import re

# XGBoost variant models to update (excluding xgboost_model.py which is already updated)
models = [
    'xgboost_et_model.py',
    'xgboost_hist_model.py',
    'xgboost_dart_model.py',
    'xgboost_gblinear_model.py',
    'xgboost_approx_model.py'
]

base_path = r'C:\StockApp\backend\advanced_ml\models'

for model_file in models:
    file_path = os.path.join(base_path, model_file)

    if not os.path.exists(file_path):
        print(f"[SKIP] {model_file} not found")
        continue

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Pattern 1: Update function signature
    # From: def train(self, X: np.ndarray, y: np.ndarray, validate: bool = False,
    # To: def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, validate: bool = False,

    content = re.sub(
        r'def train\(self, X: np\.ndarray, y: np\.ndarray, validate: bool = False,',
        'def train(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None, validate: bool = False,',
        content
    )

    # Pattern 2: Update X parameter docs
    # Add X_val and y_val documentation
    content = re.sub(
        r'(\s+X: Feature matrix.*?\n\s+y: Target labels.*?\n)',
        r'\1            X_val: Validation features (optional, for early stopping)\n            y_val: Validation labels (optional, for early stopping)\n',
        content
    )

    # Pattern 3: Replace internal split with external validation set
    # From: X_train = self.scaler.fit_transform(X)
    #       split_idx = int(0.8 * len(X))
    #       X_val = X_scaled[split_idx:]
    #       ...

    pattern_old_split = r'''        # Initialize scaler and model
        self\.scaler = StandardScaler\(\)
        X_scaled = self\.scaler\.fit_transform\(X\)

        # Split for early stopping \(80/20\)
        split_idx = int\(0\.8 \* len\(X\)\)
        X_train = X_scaled\[:split_idx\]
        X_val = X_scaled\[split_idx:\]
        y_train = y\[:split_idx\]
        y_val = y\[split_idx:\]

        # Split sample weights if provided
        sample_weight_train = None
        if sample_weight is not None:
            sample_weight_train = sample_weight\[:split_idx\]'''

    pattern_new_split = '''        # Initialize scaler and model
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X)

        # Use provided validation set if available, otherwise create internal split
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_train = y
            print(f"  Validation samples: {X_val.shape[0]}")
        else:
            # Fallback: Split for early stopping (80/20)
            split_idx = int(0.8 * len(X))
            X_val_scaled = X_train[split_idx:]
            X_train = X_train[:split_idx]
            y_train = y[:split_idx]
            y_val = y[split_idx:]

            # Split sample weights if provided
            if sample_weight is not None:
                sample_weight = sample_weight[:split_idx]'''

    content = re.sub(pattern_old_split, pattern_new_split, content)

    # Pattern 4: Update fit calls to use X_val_scaled instead of X_val
    content = re.sub(
        r'eval_set=\[\(X_val, y_val\)\]',
        'eval_set=[(X_val_scaled, y_val)]',
        content
    )

    # Pattern 5: Update val_score calculation
    content = re.sub(
        r'val_score = self\.model\.score\(X_val, y_val\)',
        'val_score = self.model.score(X_val_scaled, y_val)',
        content
    )

    # Pattern 6: Update sample_weight_train references
    content = re.sub(
        r'sample_weight=sample_weight_train',
        'sample_weight=sample_weight',
        content
    )

    # Write updated content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[OK] Updated {model_file}")

print("\n[SUCCESS] All 5 XGBoost variant models updated with validation set support!")
