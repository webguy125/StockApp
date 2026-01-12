"""
Options ML Model Training Pipeline
Trains ensemble of XGBoost, LightGBM, CatBoost with hyperparameter tuning
Then trains meta-learner on out-of-fold predictions
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_curve
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import matplotlib.pyplot as plt
import shap

# Paths
INPUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'options_training', 'training_features.parquet')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'options_models', 'v1.0')
SCALER_FILE = os.path.join(MODEL_DIR, 'feature_scaler.pkl')
FEATURE_NAMES_FILE = os.path.join(MODEL_DIR, 'feature_names.json')

os.makedirs(MODEL_DIR, exist_ok=True)

print("="*80)
print("OPTIONS ML MODEL TRAINING PIPELINE")
print("="*80)

def load_data():
    """Load preprocessed features"""
    print(f"\n[INFO] Loading data from {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)

    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target']

    print(f"[OK] Loaded {len(df)} examples, {X.shape[1]} features")
    print(f"[OK] Positive class: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"[OK] Negative class: {len(y)-y.sum()} ({(1-y.mean())*100:.1f}%)")

    return X, y

def time_based_split(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Time-based split (no shuffling to prevent lookahead bias)"""

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1"

    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]

    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]

    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]

    print(f"\n[INFO] Time-based split:")
    print(f"  Train: {len(X_train)} ({len(X_train)/n*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/n*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/n*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with hyperparameter tuning"""

    print(f"\n[INFO] Training XGBoost with GridSearchCV...")

    # Hyperparameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        use_label_encoder=False
    )

    # GridSearchCV with 3-fold CV
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate on validation set
    y_val_pred = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)

    print(f"[OK] XGBoost best params: {grid_search.best_params_}")
    print(f"[OK] XGBoost Val AUC: {val_auc:.4f}")

    # Save model
    model_file = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(best_model, f)

    # Save metadata
    metadata = {
        'model_type': 'XGBoost',
        'best_params': grid_search.best_params_,
        'val_auc': float(val_auc),
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metadata_file = os.path.join(MODEL_DIR, 'xgboost_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return best_model, val_auc

def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM with hyperparameter tuning"""

    print(f"\n[INFO] Training LightGBM with GridSearchCV...")

    # Hyperparameter grid
    param_grid = {
        'num_leaves': [31, 50, 70],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_samples': [10, 20, 30]
    }

    base_model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        random_state=42,
        verbose=-1
    )

    # GridSearchCV with 3-fold CV
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate on validation set
    y_val_pred = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)

    print(f"[OK] LightGBM best params: {grid_search.best_params_}")
    print(f"[OK] LightGBM Val AUC: {val_auc:.4f}")

    # Save model
    model_file = os.path.join(MODEL_DIR, 'lightgbm_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(best_model, f)

    # Save metadata
    metadata = {
        'model_type': 'LightGBM',
        'best_params': grid_search.best_params_,
        'val_auc': float(val_auc),
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metadata_file = os.path.join(MODEL_DIR, 'lightgbm_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return best_model, val_auc

def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost with hyperparameter tuning"""

    print(f"\n[INFO] Training CatBoost with GridSearchCV...")

    # Hyperparameter grid
    param_grid = {
        'depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'iterations': [100, 200, 300],
        'l2_leaf_reg': [1, 3, 5]
    }

    base_model = cb.CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='AUC',
        random_state=42,
        verbose=0
    )

    # GridSearchCV with 3-fold CV
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate on validation set
    y_val_pred = best_model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_pred)

    print(f"[OK] CatBoost best params: {grid_search.best_params_}")
    print(f"[OK] CatBoost Val AUC: {val_auc:.4f}")

    # Save model
    model_file = os.path.join(MODEL_DIR, 'catboost_model.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(best_model, f)

    # Save metadata
    metadata = {
        'model_type': 'CatBoost',
        'best_params': grid_search.best_params_,
        'val_auc': float(val_auc),
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metadata_file = os.path.join(MODEL_DIR, 'catboost_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return best_model, val_auc

def generate_oof_predictions(models, X_train, y_train, n_folds=5):
    """Generate out-of-fold predictions for meta-learner"""

    print(f"\n[INFO] Generating out-of-fold predictions ({n_folds} folds)...")

    oof_preds = np.zeros((len(X_train), len(models)))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=False, random_state=42)

    for model_idx, (model_name, model) in enumerate(models.items()):
        print(f"  Processing {model_name}...")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]

            # Clone and train model on this fold
            if model_name == 'XGBoost':
                fold_model = xgb.XGBClassifier(**model.get_params())
            elif model_name == 'LightGBM':
                fold_model = lgb.LGBMClassifier(**model.get_params())
            else:  # CatBoost
                fold_model = cb.CatBoostClassifier(**model.get_params())

            fold_model.fit(X_fold_train, y_fold_train)

            # Predict on validation fold
            oof_preds[val_idx, model_idx] = fold_model.predict_proba(X_fold_val)[:, 1]

    print(f"[OK] Generated OOF predictions shape: {oof_preds.shape}")

    return oof_preds

def train_meta_learner(oof_preds, y_train, models, X_val, y_val, X_test, y_test):
    """Train meta-learner (stacking) on out-of-fold predictions"""

    print(f"\n[INFO] Training meta-learner (Logistic Regression)...")

    # Train meta-learner on OOF predictions
    meta_learner = LogisticRegression(random_state=42, max_iter=1000)
    meta_learner.fit(oof_preds, y_train)

    # Generate base model predictions for validation set
    val_base_preds = np.column_stack([
        models['XGBoost'].predict_proba(X_val)[:, 1],
        models['LightGBM'].predict_proba(X_val)[:, 1],
        models['CatBoost'].predict_proba(X_val)[:, 1]
    ])

    # Meta-learner predictions on validation
    y_val_meta_pred = meta_learner.predict_proba(val_base_preds)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_meta_pred)

    print(f"[OK] Meta-learner Val AUC: {val_auc:.4f}")
    print(f"[OK] Meta-learner weights: XGB={meta_learner.coef_[0][0]:.3f}, LGB={meta_learner.coef_[0][1]:.3f}, CAT={meta_learner.coef_[0][2]:.3f}")

    # Generate base model predictions for test set
    test_base_preds = np.column_stack([
        models['XGBoost'].predict_proba(X_test)[:, 1],
        models['LightGBM'].predict_proba(X_test)[:, 1],
        models['CatBoost'].predict_proba(X_test)[:, 1]
    ])

    # Meta-learner predictions on test
    y_test_meta_pred = meta_learner.predict_proba(test_base_preds)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_meta_pred)

    print(f"[OK] Meta-learner Test AUC: {test_auc:.4f}")

    # Save meta-learner
    meta_file = os.path.join(MODEL_DIR, 'meta_learner.pkl')
    with open(meta_file, 'wb') as f:
        pickle.dump(meta_learner, f)

    # Save metadata
    metadata = {
        'model_type': 'Meta-Learner (Stacking)',
        'base_models': ['XGBoost', 'LightGBM', 'CatBoost'],
        'weights': {
            'XGBoost': float(meta_learner.coef_[0][0]),
            'LightGBM': float(meta_learner.coef_[0][1]),
            'CatBoost': float(meta_learner.coef_[0][2])
        },
        'val_auc': float(val_auc),
        'test_auc': float(test_auc),
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    metadata_file = os.path.join(MODEL_DIR, 'meta_learner_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    return meta_learner, val_auc, test_auc, y_test_meta_pred

def evaluate_final_model(y_test, y_test_pred_proba, threshold=0.5):
    """Comprehensive evaluation of final ensemble"""

    print(f"\n{'='*80}")
    print("FINAL MODEL EVALUATION (Test Set)")
    print(f"{'='*80}")

    # Binary predictions
    y_test_pred = (y_test_pred_proba >= threshold).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_test_pred_proba)
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    print(f"AUC:       {auc:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Fail', 'Success']))

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Options ML Ensemble')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    roc_file = os.path.join(MODEL_DIR, 'roc_curve.png')
    plt.savefig(roc_file, dpi=150)
    print(f"\n[OK] Saved ROC curve to {roc_file}")
    plt.close()

    return {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def generate_feature_importance(model, feature_names):
    """Generate feature importance plot"""

    print(f"\n[INFO] Generating feature importance...")

    # Get feature importance from XGBoost model
    importance = model.feature_importances_

    # Sort features by importance
    indices = np.argsort(importance)[::-1]
    top_n = 20

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance[indices[:top_n]][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Most Important Features (XGBoost)')
    plt.tight_layout()

    importance_file = os.path.join(MODEL_DIR, 'feature_importance.png')
    plt.savefig(importance_file, dpi=150)
    print(f"[OK] Saved feature importance to {importance_file}")
    plt.close()

def generate_shap_values(model, X_test, feature_names):
    """Generate SHAP values for model interpretability"""

    print(f"\n[INFO] Generating SHAP values (this may take a few minutes)...")

    # Use a sample for faster computation
    sample_size = min(500, len(X_test))
    X_sample = X_test.iloc[:sample_size]

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()

    shap_file = os.path.join(MODEL_DIR, 'shap_summary.png')
    plt.savefig(shap_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Saved SHAP summary to {shap_file}")
    plt.close()

    # Feature importance (mean absolute SHAP)
    shap_importance = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(shap_importance)[::-1][:20]

    plt.figure(figsize=(10, 8))
    plt.barh(range(20), shap_importance[indices][::-1])
    plt.yticks(range(20), [feature_names[i] for i in indices][::-1])
    plt.xlabel('Mean |SHAP Value|')
    plt.title('Top 20 Features by SHAP Importance')
    plt.tight_layout()

    shap_importance_file = os.path.join(MODEL_DIR, 'shap_importance.png')
    plt.savefig(shap_importance_file, dpi=150)
    print(f"[OK] Saved SHAP importance to {shap_importance_file}")
    plt.close()

def main():
    """Main training pipeline"""

    # Load data
    X, y = load_data()

    # Load feature names
    with open(FEATURE_NAMES_FILE, 'r') as f:
        feature_names = json.load(f)

    # Time-based split
    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(X, y)

    # Train base models
    xgb_model, xgb_auc = train_xgboost(X_train, y_train, X_val, y_val)
    lgb_model, lgb_auc = train_lightgbm(X_train, y_train, X_val, y_val)
    cat_model, cat_auc = train_catboost(X_train, y_train, X_val, y_val)

    models = {
        'XGBoost': xgb_model,
        'LightGBM': lgb_model,
        'CatBoost': cat_model
    }

    # Generate out-of-fold predictions
    oof_preds = generate_oof_predictions(models, X_train, y_train, n_folds=5)

    # Train meta-learner
    meta_learner, val_auc, test_auc, y_test_pred = train_meta_learner(
        oof_preds, y_train, models, X_val, y_val, X_test, y_test
    )

    # Evaluate final ensemble
    metrics = evaluate_final_model(y_test, y_test_pred, threshold=0.5)

    # Generate visualizations
    generate_feature_importance(xgb_model, feature_names)
    generate_shap_values(xgb_model, X_test, feature_names)

    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE - MODEL SUMMARY")
    print(f"{'='*80}")
    print(f"Base Models Validation AUC:")
    print(f"  XGBoost:  {xgb_auc:.4f}")
    print(f"  LightGBM: {lgb_auc:.4f}")
    print(f"  CatBoost: {cat_auc:.4f}")
    print(f"\nMeta-Learner Performance:")
    print(f"  Validation AUC: {val_auc:.4f}")
    print(f"  Test AUC:       {test_auc:.4f}")
    print(f"\nFinal Test Metrics:")
    print(f"  AUC:       {metrics['auc']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"\nAll models saved to: {MODEL_DIR}")
    print(f"{'='*80}\n")

    # Check if we hit target AUC > 0.75
    if test_auc >= 0.75:
        print("[SUCCESS] Target AUC > 0.75 achieved!")
    else:
        print("[WARNING] Test AUC below target (0.75). Consider:")
        print("  - Collecting more training data")
        print("  - Adding sentiment features")
        print("  - Further hyperparameter tuning")

if __name__ == '__main__':
    main()
