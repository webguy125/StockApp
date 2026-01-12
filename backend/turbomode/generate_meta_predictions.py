"""
Generate Meta-Predictions Table for Meta-Learner Retraining

This script loads all training data, runs each sample through all 8 base models,
and stores the probability outputs in a meta_predictions table.

This table is required for retraining the meta-learner with override-aware features.
"""

import os
import sys
import sqlite3
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.turbomode.turbomode_training_loader import TurboModeTrainingDataLoader
from backend.turbomode.models.xgboost_model import XGBoostModel
from backend.turbomode.models.xgboost_et_model import XGBoostETModel
from backend.turbomode.models.lightgbm_model import LightGBMModel
from backend.turbomode.models.catboost_model import CatBoostModel
from backend.turbomode.models.xgboost_hist_model import XGBoostHistModel
from backend.turbomode.models.xgboost_dart_model import XGBoostDARTModel
from backend.turbomode.models.xgboost_gblinear_model import XGBoostGBLinearModel
from backend.turbomode.models.xgboost_approx_model import XGBoostApproxModel


def create_meta_predictions_table(db_path: str):
    """
    Create meta_predictions table in database.

    Schema:
        - sample_id: INTEGER PRIMARY KEY
        - label: INTEGER (0=SELL, 1=HOLD, 2=BUY)
        - xgboost_prob_down: REAL
        - xgboost_prob_neutral: REAL
        - xgboost_prob_up: REAL
        - ... (same for all 8 models)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop table if exists
    cursor.execute('DROP TABLE IF EXISTS meta_predictions')

    # Create table
    cursor.execute('''
        CREATE TABLE meta_predictions (
            sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
            label INTEGER NOT NULL,
            xgboost_prob_down REAL,
            xgboost_prob_neutral REAL,
            xgboost_prob_up REAL,
            xgboost_et_prob_down REAL,
            xgboost_et_prob_neutral REAL,
            xgboost_et_prob_up REAL,
            lightgbm_prob_down REAL,
            lightgbm_prob_neutral REAL,
            lightgbm_prob_up REAL,
            catboost_prob_down REAL,
            catboost_prob_neutral REAL,
            catboost_prob_up REAL,
            xgboost_hist_prob_down REAL,
            xgboost_hist_prob_neutral REAL,
            xgboost_hist_prob_up REAL,
            xgboost_dart_prob_down REAL,
            xgboost_dart_prob_neutral REAL,
            xgboost_dart_prob_up REAL,
            xgboost_gblinear_prob_down REAL,
            xgboost_gblinear_prob_neutral REAL,
            xgboost_gblinear_prob_up REAL,
            xgboost_approx_prob_down REAL,
            xgboost_approx_prob_neutral REAL,
            xgboost_approx_prob_up REAL
        )
    ''')

    conn.commit()
    conn.close()

    print('[OK] Created meta_predictions table')


def load_all_base_models(models_dir: str):
    """
    Load all 8 base models from disk.

    Returns:
        Dictionary mapping model names to model instances
    """
    print('\n[STEP 1] Loading base models...')

    models = {}
    model_classes = {
        'xgboost': XGBoostModel,
        'xgboost_et': XGBoostETModel,
        'lightgbm': LightGBMModel,
        'catboost': CatBoostModel,
        'xgboost_hist': XGBoostHistModel,
        'xgboost_dart': XGBoostDARTModel,
        'xgboost_gblinear': XGBoostGBLinearModel,
        'xgboost_approx': XGBoostApproxModel
    }

    for model_name, model_class in model_classes.items():
        model_path = os.path.join(models_dir, model_name)

        if not os.path.exists(model_path):
            print(f'  [ERROR] Model not found: {model_path}')
            return None

        try:
            model = model_class(model_path=model_path)
            model.load()
            models[model_name] = model
            print(f'  [OK] Loaded {model_name}')
        except Exception as e:
            print(f'  [ERROR] Failed to load {model_name}: {e}')
            return None

    print(f'\n[OK] Loaded {len(models)} base models')
    return models


def generate_meta_predictions(db_path: str, models_dir: str, batch_size: int = 5000):
    """
    Generate meta-predictions by running all training samples through base models.

    Args:
        db_path: Path to turbomode.db
        models_dir: Path to directory containing trained models
        batch_size: Number of samples to process before committing to database
    """
    print('=' * 80)
    print('GENERATING META-PREDICTIONS TABLE')
    print('=' * 80)

    # Load base models
    models = load_all_base_models(models_dir)
    if models is None:
        print('[ERROR] Failed to load base models')
        return False

    # Load training data
    print('\n[STEP 2] Loading training data...')
    loader = TurboModeTrainingDataLoader(db_path)
    X, y = loader.load_training_data(include_hold=True)

    if len(X) == 0:
        print('[ERROR] No training data found')
        return False

    print(f'[OK] Loaded {len(X):,} training samples')

    # Create meta_predictions table
    print('\n[STEP 3] Creating meta_predictions table...')
    create_meta_predictions_table(db_path)

    # Generate predictions
    print('\n[STEP 4] Generating base model predictions...')
    print(f'Processing {len(X):,} samples in batches of {batch_size}')

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    model_names = ['xgboost', 'xgboost_et', 'lightgbm', 'catboost',
                   'xgboost_hist', 'xgboost_dart', 'xgboost_gblinear', 'xgboost_approx']

    # Process in batches (OPTIMIZED: vectorized batch predictions)
    for start_idx in tqdm(range(0, len(X), batch_size), desc='Batch progress'):
        end_idx = min(start_idx + batch_size, len(X))
        batch_X = X[start_idx:end_idx]
        batch_y = y[start_idx:end_idx]
        batch_size_actual = len(batch_X)

        # Get predictions from all models for entire batch (VECTORIZED)
        model_predictions = {}
        for model_name in model_names:
            model = models[model_name]

            try:
                # Get probability predictions for entire batch at once (FAST!)
                # Returns shape: (batch_size, 3) where 3 = [prob_down, prob_neutral, prob_up]
                probs_batch = model.predict_proba(batch_X)

                # Validate shape
                if probs_batch.shape != (batch_size_actual, 3):
                    print(f'\n[WARNING] {model_name} returned shape {probs_batch.shape}, expected ({batch_size_actual}, 3)')
                    probs_batch = np.tile([0.0, 1.0, 0.0], (batch_size_actual, 1))

                model_predictions[model_name] = probs_batch

            except Exception as e:
                print(f'\n[ERROR] {model_name} batch prediction failed: {e}')
                # Default to neutral for all samples in batch
                model_predictions[model_name] = np.tile([0.0, 1.0, 0.0], (batch_size_actual, 1))

        # Assemble rows for database insertion
        batch_predictions = []
        for i in range(batch_size_actual):
            label = int(batch_y[i])
            row_data = [label]

            # Append predictions from all 8 models
            for model_name in model_names:
                probs = model_predictions[model_name][i]
                row_data.extend([float(probs[0]), float(probs[1]), float(probs[2])])

            batch_predictions.append(tuple(row_data))

        # Bulk insert batch
        cursor.executemany('''
            INSERT INTO meta_predictions (
                label,
                xgboost_prob_down, xgboost_prob_neutral, xgboost_prob_up,
                xgboost_et_prob_down, xgboost_et_prob_neutral, xgboost_et_prob_up,
                lightgbm_prob_down, lightgbm_prob_neutral, lightgbm_prob_up,
                catboost_prob_down, catboost_prob_neutral, catboost_prob_up,
                xgboost_hist_prob_down, xgboost_hist_prob_neutral, xgboost_hist_prob_up,
                xgboost_dart_prob_down, xgboost_dart_prob_neutral, xgboost_dart_prob_up,
                xgboost_gblinear_prob_down, xgboost_gblinear_prob_neutral, xgboost_gblinear_prob_up,
                xgboost_approx_prob_down, xgboost_approx_prob_neutral, xgboost_approx_prob_up
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', batch_predictions)

        conn.commit()

    conn.close()

    # Verify
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM meta_predictions')
    count = cursor.fetchone()[0]
    conn.close()

    print(f'\n[OK] Generated {count:,} meta-predictions')
    print('=' * 80)
    print('META-PREDICTIONS TABLE GENERATION COMPLETE')
    print('=' * 80)

    return True


if __name__ == '__main__':
    # Paths
    db_path = Path(__file__).parent.parent / 'data' / 'turbomode.db'
    models_dir = Path(__file__).parent.parent / 'data' / 'turbomode_models'

    print(f'\nDatabase: {db_path}')
    print(f'Models directory: {models_dir}')
    print('\n' + '=' * 80)

    # Generate meta-predictions
    success = generate_meta_predictions(str(db_path), str(models_dir), batch_size=1000)

    if success:
        print('\n[SUCCESS] Meta-predictions table ready for meta-learner retraining')
        print('\nNext step: Run retrain_meta_with_override_features.py')
    else:
        print('\n[FAILED] Meta-predictions generation failed')
