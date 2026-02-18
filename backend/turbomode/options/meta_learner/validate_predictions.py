#!/usr/bin/env python
"""
Validate Options Meta-Learner Predictions Against Recent Data

This script:
1. Loads recent market data (last 30 days)
2. Generates predictions using all 5 trained models
3. Compares predictions with actual outcomes
4. Calculates performance metrics
5. Generates validation report
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIONS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
TURBOMODE_DIR = os.path.abspath(os.path.join(OPTIONS_DIR, '..'))
BACKEND_DIR = os.path.abspath(os.path.join(TURBOMODE_DIR, '..'))
STOCKAPP_DIR = os.path.abspath(os.path.join(BACKEND_DIR, '..'))

MASTER_MARKET_DB = os.path.join(STOCKAPP_DIR, 'master_market_data', 'master_market_data.db')
OPTIONS_UNIVERSE_DB = os.path.join(OPTIONS_DIR, 'Data', 'options_universe.db')
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')


def load_recent_ohlcv(days=30):
    """Load recent OHLCV data from master_market_data.db"""
    logger.info(f"Loading OHLCV data for last {days} days")

    cutoff_date = datetime.now() - timedelta(days=days)
    cutoff_timestamp = int(cutoff_date.timestamp())

    query = f"""
        SELECT symbol, timestamp as date, close, volume
        FROM ohlcv
        WHERE timestamp >= {cutoff_timestamp}
        ORDER BY symbol, timestamp
    """

    conn = sqlite3.connect(MASTER_MARKET_DB)
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['date'] = pd.to_datetime(df['date'], unit='s')

    logger.info(f"Loaded {len(df)} rows for {df['symbol'].nunique()} symbols")
    return df


def load_options_features():
    """Load options features from options_universe.db"""
    logger.info("Loading options features")

    conn = sqlite3.connect(OPTIONS_UNIVERSE_DB)
    df_opt = pd.read_sql_query("SELECT * FROM option_features_daily", conn)
    conn.close()

    df_opt['date'] = pd.to_datetime(df_opt['date'])

    logger.info(f"Loaded {len(df_opt)} options feature rows")
    return df_opt


def compute_features(df):
    """Compute price-based features"""
    df = df.copy()
    df = df.sort_values(['symbol', 'date'])

    for symbol in df['symbol'].unique():
        mask = df['symbol'] == symbol
        symbol_data = df.loc[mask].copy()

        symbol_data['close_1d_ago'] = symbol_data['close'].shift(1)
        symbol_data['close_2d_ago'] = symbol_data['close'].shift(2)
        symbol_data['close_5d_ago'] = symbol_data['close'].shift(5)
        symbol_data['close_10d_ago'] = symbol_data['close'].shift(10)
        symbol_data['close_20d_ago'] = symbol_data['close'].shift(20)

        symbol_data['return_1d'] = (symbol_data['close'] / symbol_data['close_1d_ago']) - 1
        symbol_data['return_2d'] = (symbol_data['close'] / symbol_data['close_2d_ago']) - 1
        symbol_data['return_5d'] = (symbol_data['close'] / symbol_data['close_5d_ago']) - 1
        symbol_data['return_10d'] = (symbol_data['close'] / symbol_data['close_10d_ago']) - 1
        symbol_data['return_20d'] = (symbol_data['close'] / symbol_data['close_20d_ago']) - 1

        symbol_data['realized_vol_5d'] = symbol_data['return_1d'].rolling(5).std()
        symbol_data['realized_vol_10d'] = symbol_data['return_1d'].rolling(10).std()
        symbol_data['realized_vol_20d'] = symbol_data['return_1d'].rolling(20).std()

        symbol_data['volume_5d_avg'] = symbol_data['volume'].rolling(5).mean()
        symbol_data['volume_ratio'] = symbol_data['volume'] / symbol_data['volume_5d_avg']

        avg_vol_20d = symbol_data['return_1d'].rolling(20).std().mean()
        symbol_data['vol_regime'] = np.where(
            symbol_data['realized_vol_20d'] > avg_vol_20d * 1.2, 1, 0
        )

        df.loc[mask] = symbol_data

    return df


def compute_labels(df, horizons):
    """Compute actual outcomes for validation"""
    df = df.copy()

    for horizon_name, horizon_days in horizons.items():
        col_name = f'direction_{horizon_name}'
        future_price = df.groupby('symbol')['close'].shift(-horizon_days)
        current_price = df['close']

        df[col_name] = np.where(
            current_price > 0,
            (future_price / current_price) - 1,
            np.nan
        )

        df[col_name] = df[col_name].replace([np.inf, -np.inf], np.nan)

    volatility_horizon_1 = 2
    df['actual_volatility_H1'] = df.groupby('symbol')['return_1d'].shift(-volatility_horizon_1).rolling(volatility_horizon_1).std().shift(-volatility_horizon_1)

    volatility_horizon_2 = 10
    df['actual_volatility_H2'] = df.groupby('symbol')['return_1d'].shift(-volatility_horizon_2).rolling(volatility_horizon_2).std().shift(-volatility_horizon_2)

    return df


def load_models():
    """Load all trained models"""
    logger.info("Loading trained models")

    models = {}
    model_files = {
        'direction_h1': 'direction_h1.pkl',
        'direction_h2': 'direction_h2.pkl',
        'volatility_h1': 'volatility_h1.pkl',
        'volatility_h2': 'volatility_h2.pkl',
        'strategy': 'strategy.pkl'
    }

    for name, filename in model_files.items():
        filepath = os.path.join(MODELS_DIR, filename)
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                models[name] = pickle.load(f)
            logger.info(f"Loaded {name} model")
        else:
            logger.warning(f"Model not found: {filepath}")

    metadata_path = os.path.join(MODELS_DIR, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info("Loaded model metadata")
    else:
        metadata = None
        logger.warning("Metadata not found")

    return models, metadata


def generate_predictions(df, models, feature_cols):
    """Generate predictions using trained models"""
    logger.info("Generating predictions")

    X = df[feature_cols].copy()

    predictions = {}

    if 'direction_h1' in models:
        predictions['pred_direction_h1'] = models['direction_h1'].predict(X)
        predictions['pred_direction_h1_proba'] = models['direction_h1'].predict_proba(X)[:, 1]

    if 'direction_h2' in models:
        predictions['pred_direction_h2'] = models['direction_h2'].predict(X)
        predictions['pred_direction_h2_proba'] = models['direction_h2'].predict_proba(X)[:, 1]

    if 'volatility_h1' in models:
        predictions['pred_volatility_h1'] = models['volatility_h1'].predict(X)

    if 'volatility_h2' in models:
        predictions['pred_volatility_h2'] = models['volatility_h2'].predict(X)

    if 'strategy' in models:
        predictions['pred_strategy'] = models['strategy'].predict(X)

    for key, values in predictions.items():
        df[key] = values

    return df


def calculate_metrics(df):
    """Calculate performance metrics"""
    logger.info("Calculating performance metrics")

    metrics = {}

    df_valid_h1 = df[df['direction_H1'].notna()].copy()
    df_valid_h1['actual_direction_h1'] = (df_valid_h1['direction_H1'] > 0).astype(int)

    if len(df_valid_h1) > 0 and 'pred_direction_h1' in df_valid_h1.columns:
        metrics['direction_h1_accuracy'] = accuracy_score(
            df_valid_h1['actual_direction_h1'],
            df_valid_h1['pred_direction_h1']
        )
        metrics['direction_h1_precision'] = precision_score(
            df_valid_h1['actual_direction_h1'],
            df_valid_h1['pred_direction_h1'],
            zero_division=0
        )
        metrics['direction_h1_recall'] = recall_score(
            df_valid_h1['actual_direction_h1'],
            df_valid_h1['pred_direction_h1'],
            zero_division=0
        )
        metrics['direction_h1_f1'] = f1_score(
            df_valid_h1['actual_direction_h1'],
            df_valid_h1['pred_direction_h1'],
            zero_division=0
        )
        metrics['direction_h1_samples'] = len(df_valid_h1)

    df_valid_h2 = df[df['direction_H2'].notna()].copy()
    df_valid_h2['actual_direction_h2'] = (df_valid_h2['direction_H2'] > 0).astype(int)

    if len(df_valid_h2) > 0 and 'pred_direction_h2' in df_valid_h2.columns:
        metrics['direction_h2_accuracy'] = accuracy_score(
            df_valid_h2['actual_direction_h2'],
            df_valid_h2['pred_direction_h2']
        )
        metrics['direction_h2_precision'] = precision_score(
            df_valid_h2['actual_direction_h2'],
            df_valid_h2['pred_direction_h2'],
            zero_division=0
        )
        metrics['direction_h2_recall'] = recall_score(
            df_valid_h2['actual_direction_h2'],
            df_valid_h2['pred_direction_h2'],
            zero_division=0
        )
        metrics['direction_h2_f1'] = f1_score(
            df_valid_h2['actual_direction_h2'],
            df_valid_h2['pred_direction_h2'],
            zero_division=0
        )
        metrics['direction_h2_samples'] = len(df_valid_h2)

    df_valid_vol_h1 = df[df['actual_volatility_H1'].notna()].copy()
    if len(df_valid_vol_h1) > 0 and 'pred_volatility_h1' in df_valid_vol_h1.columns:
        metrics['volatility_h1_mae'] = mean_absolute_error(
            df_valid_vol_h1['actual_volatility_H1'],
            df_valid_vol_h1['pred_volatility_h1']
        )
        metrics['volatility_h1_rmse'] = np.sqrt(mean_squared_error(
            df_valid_vol_h1['actual_volatility_H1'],
            df_valid_vol_h1['pred_volatility_h1']
        ))
        metrics['volatility_h1_samples'] = len(df_valid_vol_h1)

    df_valid_vol_h2 = df[df['actual_volatility_H2'].notna()].copy()
    if len(df_valid_vol_h2) > 0 and 'pred_volatility_h2' in df_valid_vol_h2.columns:
        metrics['volatility_h2_mae'] = mean_absolute_error(
            df_valid_vol_h2['actual_volatility_H2'],
            df_valid_vol_h2['pred_volatility_h2']
        )
        metrics['volatility_h2_rmse'] = np.sqrt(mean_squared_error(
            df_valid_vol_h2['actual_volatility_H2'],
            df_valid_vol_h2['pred_volatility_h2']
        ))
        metrics['volatility_h2_samples'] = len(df_valid_vol_h2)

    return metrics


def save_validation_report(metrics, output_path):
    """Save validation report to JSON"""
    report = {
        'validation_date': datetime.now().isoformat(),
        'metrics': metrics,
        'summary': {
            'direction_h1_meets_threshold': metrics.get('direction_h1_accuracy', 0) >= 0.60,
            'direction_h2_meets_threshold': metrics.get('direction_h2_accuracy', 0) >= 0.60,
            'overall_status': 'PASS' if all([
                metrics.get('direction_h1_accuracy', 0) >= 0.60,
                metrics.get('direction_h2_accuracy', 0) >= 0.60
            ]) else 'REVIEW_NEEDED'
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f"Validation report saved to {output_path}")
    return report


def print_validation_summary(metrics):
    """Print validation summary to console"""
    print("\n" + "="*80)
    print("OPTIONS META-LEARNER VALIDATION REPORT")
    print("="*80)
    print(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("DIRECTION MODELS (2-day horizon)")
    print("-" * 80)
    if 'direction_h1_accuracy' in metrics:
        print(f"  Accuracy:  {metrics['direction_h1_accuracy']:.1%}")
        print(f"  Precision: {metrics['direction_h1_precision']:.1%}")
        print(f"  Recall:    {metrics['direction_h1_recall']:.1%}")
        print(f"  F1 Score:  {metrics['direction_h1_f1']:.3f}")
        print(f"  Samples:   {metrics['direction_h1_samples']}")
        status = "PASS" if metrics['direction_h1_accuracy'] >= 0.60 else "REVIEW"
        print(f"  Status:    {status} (threshold: 60%)")
    else:
        print("  No data available")

    print()
    print("DIRECTION MODELS (10-day horizon)")
    print("-" * 80)
    if 'direction_h2_accuracy' in metrics:
        print(f"  Accuracy:  {metrics['direction_h2_accuracy']:.1%}")
        print(f"  Precision: {metrics['direction_h2_precision']:.1%}")
        print(f"  Recall:    {metrics['direction_h2_recall']:.1%}")
        print(f"  F1 Score:  {metrics['direction_h2_f1']:.3f}")
        print(f"  Samples:   {metrics['direction_h2_samples']}")
        status = "PASS" if metrics['direction_h2_accuracy'] >= 0.60 else "REVIEW"
        print(f"  Status:    {status} (threshold: 60%)")
    else:
        print("  No data available")

    print()
    print("VOLATILITY MODELS (2-day horizon)")
    print("-" * 80)
    if 'volatility_h1_mae' in metrics:
        print(f"  MAE:       {metrics['volatility_h1_mae']:.6f}")
        print(f"  RMSE:      {metrics['volatility_h1_rmse']:.6f}")
        print(f"  Samples:   {metrics['volatility_h1_samples']}")
    else:
        print("  No data available")

    print()
    print("VOLATILITY MODELS (10-day horizon)")
    print("-" * 80)
    if 'volatility_h2_mae' in metrics:
        print(f"  MAE:       {metrics['volatility_h2_mae']:.6f}")
        print(f"  RMSE:      {metrics['volatility_h2_rmse']:.6f}")
        print(f"  Samples:   {metrics['volatility_h2_samples']}")
    else:
        print("  No data available")

    print()
    print("="*80)

    overall_pass = all([
        metrics.get('direction_h1_accuracy', 0) >= 0.60,
        metrics.get('direction_h2_accuracy', 0) >= 0.60
    ])

    if overall_pass:
        print("OVERALL STATUS: PASS")
        print("Models meet minimum accuracy thresholds")
    else:
        print("OVERALL STATUS: REVIEW NEEDED")
        print("One or more models below accuracy threshold")

    print("="*80)
    print()


def main():
    """Main validation function"""
    logger.info("Starting options meta-learner validation")

    models, metadata = load_models()

    if not models:
        logger.error("No models found. Please train models first.")
        return

    df_ohlcv = load_recent_ohlcv(days=60)

    df = compute_features(df_ohlcv)

    horizons = {'H1': 2, 'H2': 10}
    df = compute_labels(df, horizons)

    df_opt = load_options_features()
    df = df.merge(df_opt, on=['symbol', 'date'], how='left')

    df = df.replace([np.inf, -np.inf], np.nan)

    if metadata and 'feature_cols' in metadata:
        feature_cols = [col for col in metadata['feature_cols'] if col in df.columns]
    else:
        exclude_cols = ['symbol', 'date', 'direction_H1', 'direction_H2',
                        'actual_volatility_H1', 'actual_volatility_H2',
                        'pred_direction_h1', 'pred_direction_h2',
                        'pred_volatility_h1', 'pred_volatility_h2',
                        'pred_strategy', 'id', 'created_at',
                        'actual_direction_h1', 'actual_direction_h2',
                        'pred_direction_h1_proba', 'pred_direction_h2_proba']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

    logger.info(f"Using {len(feature_cols)} features for prediction")

    df = generate_predictions(df, models, feature_cols)

    metrics = calculate_metrics(df)

    output_path = os.path.join(SCRIPT_DIR, 'validation_report.json')
    report = save_validation_report(metrics, output_path)

    print_validation_summary(metrics)

    logger.info("Validation complete")


if __name__ == '__main__':
    main()
